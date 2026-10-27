"""
McPAT 1.3 interface for uncore blocks: shared L2 (l2), L3 (l3), mesh NoC routers (noc),
and memory controllers (mc).

Each query fills templates/<class>.xml (a Xeon.xml derivative with cores and every
other block disabled) and runs McPAT at -print_level 5. Per-access energy uses McPAT's
Runtime Dynamic power with a difference method: one run with all access stats at 0
(base) and one run per operation with n_accesses of it, all over total_cycles, so
background terms (e.g. the memory controller idle term) cancel:

    energy per access = (RtDyn_op - RtDyn_base) [W] * total_cycles / (frequency [MHz] * 1e6) / n_accesses

The NoC is modeled as routers only (mesh, no global links), because McPAT derives link
length from the chip area, which is zero without cores. Link and wire energy is NOT
included. One NoC access is one flit traversing one router (one hop); NoC area and leakage
cover the whole mesh (horizontal_nodes x vertical_nodes routers).

L2/L3 energy is for hits only. McPAT sizes the cache output port at half a line
(sharedcache.cc:121), so l2/l3 numbers are not directly comparable to cacti7 sram.
MC energy is per 64 B line (template block_size) and includes the PHY, which McPAT
always models for the default MC type (memoryctrl.cc:547). The returned MC frequency is
the system clock; the MC itself runs at mc_clock. The McPAT core is not supported.
"""

import subprocess, os, copy, platform, hashlib, math, re, fcntl
import xml.etree.ElementTree as ET

from collections import OrderedDict

from loguru import logger


interface_dir = os.path.dirname(os.path.abspath(__file__))
run_dir = os.path.join(interface_dir, "include/mcpat")
template_dir = os.path.join(interface_dir, "templates")

total_cycles = 100000
n_accesses = 100000
run_timeout_seconds = 120
# McPAT has technology data for 22, 32, 45, 65, 90, 180 nm and interpolates between them.
min_technology_nm = 22
max_technology_nm = 180

zero = dict.fromkeys(
    ("read_accesses", "write_accesses", "read_misses", "write_misses"), 0
)
cache_runs = {
    "base": zero,
    "read": dict(zero, read_accesses=n_accesses),
    "write": dict(zero, write_accesses=n_accesses),
}

# component: McPAT component id of the block; section: report header of the block total;
# clock: block clock param set to the query frequency; config: cache config param;
# params: optional query keys written to the block param of the same name (default: template value);
# runs: access stats per run; read/write: run that prices each operation (default: the run of that name).
classes = {
    "l2": {
        "component": "system.L20",
        "section": "Total L2s:",
        "clock": "clockrate",
        "config": "L2_config",
        "params": (),
        "runs": cache_runs,
    },
    "l3": {
        "component": "system.L30",
        "section": "Total L3s:",
        "clock": "clockrate",
        "config": "L3_config",
        "params": (),
        "runs": cache_runs,
    },
    "noc": {
        "component": "system.NoC0",
        "section": "Total NoCs (Network/Bus):",
        "clock": "clockrate",
        "config": None,
        "params": (
            "horizontal_nodes",
            "vertical_nodes",
            "flit_bits",
            "input_ports",
            "output_ports",
            "virtual_channel_per_port",
            "input_buffer_entries_per_vc",
        ),
        "runs": {
            "base": {"total_accesses": 0},
            "access": {"total_accesses": n_accesses},
        },
        # McPAT prices every NoC access the same, so read and write share one energy.
        "read": "access",
        "write": "access",
    },
    "mc": {
        "component": "system.mc",
        "section": "Total MCs:",
        # The MC runs at mc_clock (DIMM IO bus clock), a query param; the query frequency sets the system clock.
        "clock": None,
        "config": None,
        "params": (
            "mc_clock",
            "peak_transfer_rate",
            "memory_channels_per_mc",
            "databus_width",
        ),
        "runs": {
            "base": {"memory_reads": 0, "memory_writes": 0},
            "read": {"memory_reads": n_accesses, "memory_writes": 0},
            "write": {"memory_reads": 0, "memory_writes": n_accesses},
        },
    },
}

# cache size keys, shared with the cacti7 sram query
cache_keys = ("width", "depth", "bank", "associativity")

# report field -> (label, unit) as printed by McPAT
report_fields = {
    "area": ("Area", "mm^2"),
    "subthreshold_leakage": ("Subthreshold Leakage", "W"),
    "gate_leakage": ("Gate Leakage", "W"),
    "runtime_dynamic": ("Runtime Dynamic", "W"),
}


def mcpat_signature(
    query_class: str, technology, frequency, knobs: OrderedDict, template_file: str
):
    """
    Identify a query by everything that shapes its reports, including the template contents
    and the run lengths. Reports are named after this, so differing inputs get their own files.
    """
    with open(template_file, "rb") as template:
        template_bytes = template.read()
    knobs = sorted((str(key), str(value)) for key, value in knobs.items())
    payload = (
        repr(
            (
                query_class,
                str(technology),
                str(frequency),
                knobs,
                total_cycles,
                n_accesses,
            )
        ).encode()
        + template_bytes
    )
    return hashlib.sha256(payload).hexdigest()[:12]


def find_component(root, component_id: str):
    for component in root.iter("component"):
        if component.get("id") == component_id:
            return component
    raise KeyError(f"No component <{component_id}> in McPAT template.")


def find_entry(component, tag: str, name: str):
    """
    Return the direct <tag name=...> child of a component; nested components are not searched.
    """
    entries = [entry for entry in component.findall(tag) if entry.get("name") == name]
    if len(entries) != 1:
        raise KeyError(
            f'Expected one <{tag} name="{name}"> in component <{component.get("id")}>, found {len(entries)}.'
        )
    return entries[0]


def mcpat_write_xml(
    query_class: str,
    technology,
    frequency: int,
    knobs: OrderedDict,
    stats: dict,
    template_file: str,
    xml_file: str,
):
    spec = classes[query_class]
    tree = ET.parse(template_file)
    system = find_component(tree.getroot(), "system")
    block = find_component(tree.getroot(), spec["component"])

    find_entry(system, "param", "core_tech_node").set("value", str(technology))
    find_entry(system, "param", "target_core_clockrate").set("value", str(frequency))
    find_entry(system, "stat", "total_cycles").set("value", str(total_cycles))
    # The core is disabled; its clock only appears in the report header.
    core = find_component(tree.getroot(), "system.core0")
    find_entry(core, "param", "clock_rate").set("value", str(frequency))
    if spec["clock"] is not None:
        find_entry(block, "param", spec["clock"]).set("value", str(frequency))

    if spec["config"] is not None:
        # Cache size uses the cacti7 sram keys: width bits per line, depth lines per bank, bank banks.
        width = int(knobs["width"])
        if width % 8:
            raise ValueError(f"Cache width <{width}> must be a multiple of 8 bits.")
        config = find_entry(block, "param", spec["config"])
        # fields: capacity (B), block size (B), associativity, banks, throughput, latency, ...
        fields = config.get("value").split(",")
        fields[0] = str(int(knobs["bank"]) * int(knobs["depth"]) * width // 8)
        fields[1] = str(width // 8)
        fields[3] = str(int(knobs["bank"]))
        if "associativity" in knobs:
            fields[2] = str(int(knobs["associativity"]))
        config.set("value", ",".join(fields))

    for key in spec["params"]:
        if key in knobs:
            find_entry(block, "param", key).set("value", str(knobs[key]))

    for key, value in stats.items():
        find_entry(block, "stat", key).set("value", str(value))

    tree.write(xml_file)


def mcpat_binary():
    """
    McPAT binaries are stored per host as 'mcpat-<system>-<machine>' (e.g. mcpat-Linux-x86_64);
    build the one for this host from source via make when it is absent.
    """
    binary = "mcpat-" + platform.system() + "-" + platform.machine()
    binary_file = os.path.join(run_dir, binary)
    if os.path.exists(binary_file):
        return binary
    # Serialize builds across processes; another process may finish the build while we wait.
    with open(os.path.join(run_dir, ".build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(binary_file):
            logger.info(
                "No McPAT binary for <" + binary + ">; building from source via make."
            )
            subprocess.run(["make", "all"], cwd=run_dir, check=True)
            os.replace(os.path.join(run_dir, "mcpat"), binary_file)
            subprocess.run(["make", "clean_opt"], cwd=run_dir, check=True)
    return binary


def mcpat_run(xml_file: str, report_file: str):
    binary = mcpat_binary()
    with open(report_file, "w") as report:
        subprocess.run(
            ["./" + binary, "-infile", xml_file, "-print_level", "5"],
            cwd=run_dir,
            stdout=report,
            check=True,
            timeout=run_timeout_seconds,
        )


def parse_report_section(report_file: str, section: str):
    """
    Read the fields of one block total from a McPAT report. The block starts at its header
    line and ends at the next blank line.
    """
    with open(report_file, "r") as report:
        text = report.read()
    start = text.find(section)
    if start < 0:
        raise RuntimeError(f"No <{section}> section in McPAT report <{report_file}>.")
    block = re.split(r"\n\s*\n", text[start:], maxsplit=1)[0]

    values = {}
    for field, (label, unit) in report_fields.items():
        match = re.search(
            r"^\s*" + re.escape(label) + r" = (\S+) " + re.escape(unit) + r"\s*$",
            block,
            re.M,
        )
        if match is None:
            raise RuntimeError(
                f"No <{label}> in <{section}> of McPAT report <{report_file}>."
            )
        value = float(match.group(1))
        if not math.isfinite(value):
            raise RuntimeError(
                f"<{label}> is <{match.group(1)}> in <{section}> of McPAT report <{report_file}>."
            )
        values[field] = value
    return values


def query(
    name: str, interface: str, query: OrderedDict, input_dir=None, output_dir=None
):
    query = copy.deepcopy(query)
    query_class = query["class"].lower()
    query_technology = query["technology"]
    query_frequency = query["frequency"]
    del query["class"]
    del query["technology"]
    del query["frequency"]

    assert os.path.isdir(output_dir), logger.error(
        f"Invalid output dir <{output_dir}>."
    )
    if query_class not in classes:
        raise ValueError(
            f"Invalid <{query_class}> in McPAT; valid values: {list(classes)}"
        )
    if not min_technology_nm <= float(query_technology) <= max_technology_nm:
        raise ValueError(
            f"McPAT supports technology {min_technology_nm} to {max_technology_nm} nm, got <{query_technology}>."
        )
    if float(query_frequency) != int(query_frequency) or float(query_frequency) <= 0:
        raise ValueError(
            f"McPAT reads clock rates as positive integer MHz, got <{query_frequency}>."
        )
    frequency = int(query_frequency)

    spec = classes[query_class]
    accepted = (cache_keys if spec["config"] is not None else ()) + spec["params"]
    unknown = [key for key in query if key not in accepted]
    if unknown:
        raise ValueError(
            f"Unknown keys {unknown} for McPAT class <{query_class}>; valid keys: {list(accepted)}"
        )
    if os.sep in name or (os.altsep and os.altsep in name) or ".." in name:
        raise ValueError(f"Invalid module name <{name}> for McPAT output files.")
    template_file = os.path.join(template_dir, query_class + ".xml")
    signature = mcpat_signature(
        query_class, query_technology, frequency, query, template_file
    )
    prefix = os.path.join(
        os.path.abspath(output_dir),
        name + "." + query_class + ".tech" + str(query_technology) + "." + signature,
    )

    sections = {}
    for run, stats in spec["runs"].items():
        xml_file = prefix + "." + run + ".mcpat.xml"
        report_file = prefix + "." + run + ".mcpat.rpt"
        flag_file = xml_file + ".flag"
        done = os.path.exists(flag_file)
        if not done:
            mcpat_write_xml(
                query_class,
                query_technology,
                frequency,
                query,
                stats,
                template_file,
                xml_file,
            )
            mcpat_run(xml_file, report_file)
        sections[run] = parse_report_section(report_file, spec["section"])
        if not done:
            with open(flag_file, "w") as flag:
                flag.write(flag_file)

    execution_time_s = total_cycles / (frequency * 1e6)

    def energy_per_access_nJ(run):
        delta_W = sections[run]["runtime_dynamic"] - sections["base"]["runtime_dynamic"]
        return delta_W * execution_time_s / n_accesses * 1e9

    base = sections["base"]
    energy_per_rd_nJ = energy_per_access_nJ(spec.get("read", "read"))
    energy_per_wr_nJ = energy_per_access_nJ(spec.get("write", "write"))
    # Subthreshold Leakage is the value without power gating.
    leakage_power_mW = (base["subthreshold_leakage"] + base["gate_leakage"]) * 1e3
    total_area_mm2 = base["area"]

    output_dict = OrderedDict()
    output_dict["technology"] = OrderedDict({"value": query_technology, "unit": "nm"})
    output_dict["frequency"] = OrderedDict({"value": query_frequency, "unit": "MHz"})
    output_dict["dynamic_energy"] = OrderedDict(
        {
            "read": OrderedDict({"value": energy_per_rd_nJ, "unit": "nJ"}),
            "write": OrderedDict({"value": energy_per_wr_nJ, "unit": "nJ"}),
        }
    )
    output_dict["leakage_power"] = OrderedDict(
        {"value": leakage_power_mW, "unit": "mW"}
    )
    output_dict["area"] = OrderedDict({"value": total_area_mm2, "unit": "mm^2"})

    return output_dict
