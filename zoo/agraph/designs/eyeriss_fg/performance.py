from collections import OrderedDict
from math import ceil
from archx.utils import get_prod

# off-chip fmap RLC ratio (JSSC 2017 Sec. VI: 1.2x-1.9x); filters are not RLC coded
RLC_RATIO = 1.5
# psum spad entries inside pe (pe.v PSUM_SPAD_WIDTH)
PSUM_SPAD_DEPTH = 24
# ifmap spad entries inside pe (pe.v IFMAP_SPAD_WIDTH); it holds one filter row's window
IFMAP_SPAD_DEPTH = 12


def _sram_bytes(architecture_dict, module):
    query = architecture_dict[module]["query"]
    instances = get_prod(architecture_dict[module]["instance"])
    return query["width"] * query["depth"] * query["bank"] * instances // 8


def _layer(workload_dict, index):
    configuration = workload_dict["configuration"]
    H, W, C = configuration["conv_ifmap"][index]
    R, S = configuration["conv_filter"][index]
    E, F = configuration["conv_ofmap"][index]
    M = configuration["conv_num_filter"][index]
    g = configuration["conv_group"][index]
    U = configuration["conv_stride"][index]
    batch = configuration["batch"]

    layer = OrderedDict()
    layer["H"], layer["W"], layer["C"] = H, W, C
    layer["R"], layer["S"] = R, S
    layer["E"], layer["F"] = E, F
    layer["M"] = M
    # per-group filter depth: AlexNet conv2/4/5 split the channels over 2 groups
    layer["c"] = C // g
    # filters per group: an ifmap word meets only the filters of its own group
    layer["m_group"] = M // g
    layer["batch"] = batch
    # filters are shared by the images processed in the same pass
    layer["filter_loads"] = ceil(batch / configuration["images_per_pass"][index])
    layer["ifmap_zero_ratio"] = configuration["ifmap_zero_ratio"][index]
    layer["macs"] = R * S * layer["c"] * M * E * F * batch
    layer["bitwidth"] = configuration["bitwidth"]
    layer["U"] = U
    # JSSC 2017 Table III: the RS dataflow mapping parameters of this layer
    layer["map_m"] = configuration["map_m"][index]
    layer["map_n"] = configuration["images_per_pass"][index]
    layer["map_e"] = configuration["map_e"][index]
    layer["map_p"] = configuration["map_p"][index]
    layer["map_q"] = configuration["map_q"][index]
    layer["map_r"] = configuration["map_r"][index]
    layer["map_t"] = configuration["map_t"][index]
    layer["map_glb_ifmap_kb"] = configuration["map_glb_ifmap_kb"][index]
    layer["map_glb_psum_kb"] = configuration["map_glb_psum_kb"][index]
    return layer


def _segments(width, pe_cols):
    """Fold one PE-set row of `width` PEs into segments no wider than the array.

    JSSC 2017 Sec. IV-A: a set wider than 14 "is divided into separated segments
    that are mapped to different parts of the PE array". The CONV2 set of width
    27 becomes 14 + 13.
    """
    count = ceil(width / pe_cols)
    base, extra = divmod(width, count)
    return [base + 1] * extra + [base] * (count - extra)


def _mapping(architecture_dict, layer):
    """Map the layer with the JSSC 2017 Table III parameters.

    The mapping is read from the paper, not derived from the storage capacity;
    the capacity is only checked against it below.
    """
    pe_rows, pe_cols = (
        architecture_dict["multiplier"]["instance"][0],
        architecture_dict["multiplier"]["instance"][1],
    )
    filter_depth = architecture_dict["filter_spad"]["query"]["depth"]
    byte = layer["bitwidth"] // 8

    m, n, e = layer["map_m"], layer["map_n"], layer["map_e"]
    p, q, r, t = layer["map_p"], layer["map_q"], layer["map_r"], layer["map_t"]
    R, S, E, F, U = layer["R"], layer["S"], layer["E"], layer["F"], layer["U"]

    # one PE set is R filter rows x e ofmap rows; a row too wide for the array is segmented
    # (_segments picks ceil(e / pe_cols) segments, so no segment is ever wider than the array)
    segments = _segments(e, pe_cols)
    used_pe = R * e * r * t
    # the r * t sets have to tile the array: one set is R * len(segments) rows x max(segments) cols
    set_rows, set_cols = R * len(segments), max(segments)
    sets_that_fit = (pe_rows // set_rows) * (pe_cols // set_cols)
    assert (
        r * t <= sets_that_fit
    ), f"{r * t} PE sets of {set_rows}x{set_cols} do not tile a {pe_rows}x{pe_cols} array"
    assert (
        used_pe <= pe_rows * pe_cols
    ), f"mapping needs {used_pe} PEs, array has {pe_rows * pe_cols}"

    # spad capacity: a PE holds p filters x q channels of one S-long filter row,
    # p psums, and the S-entry ifmap window
    assert (
        p * q * S <= filter_depth
    ), f"filter spad needs {p * q * S} of {filter_depth} entries"
    # filter_noc_transfer stalls the pass, which holds only if the next pass's filters can't preload
    assert (
        2 * p * q * S > filter_depth
    ), f"filter spad can double buffer {p * q * S} entries, so the pass ramp-up would hide"
    assert p <= PSUM_SPAD_DEPTH, f"psum spad needs {p} of {PSUM_SPAD_DEPTH} entries"
    assert S <= IFMAP_SPAD_DEPTH, f"ifmap spad needs {S} of {IFMAP_SPAD_DEPTH} entries"

    # strip mining: the set covers e of the E ofmap rows at a time (Sec. IV-A)
    strips = ceil(E / e)
    ifmap_rows_per_sweep = 0
    remaining = E
    for _ in range(strips):
        rows = min(e, remaining)
        remaining -= rows
        # a strip of `rows` ofmap rows reads (rows - 1) * U + R ifmap rows
        ifmap_rows_per_sweep += min((rows - 1) * U + R, layer["H"])

    # GLB capacity: n * q * r 2-D ifmaps and n * m 2-D psums have to fit the ifmap/psum banks (Sec. IV-A)
    glb_ifmap_bytes = n * q * r * min((e - 1) * U + R, layer["H"]) * layer["W"] * byte
    glb_psum_bytes = n * m * e * F * byte
    glb_fmap_capacity = _sram_bytes(architecture_dict, "ifmap_psum_glb")
    assert (
        glb_ifmap_bytes + glb_psum_bytes <= glb_fmap_capacity
    ), f"ifmap_psum_glb needs {glb_ifmap_bytes + glb_psum_bytes} of {glb_fmap_capacity} B"
    # the GLB preloads the next pass's filters into its separate filter banks (Sec. V-A)
    glb_filter_bytes = p * t * q * r * R * S * byte
    glb_filter_capacity = _sram_bytes(architecture_dict, "filt_glb")
    assert (
        glb_filter_bytes <= glb_filter_capacity
    ), f"filt_glb needs {glb_filter_bytes} of {glb_filter_capacity} B"

    # Table III also publishes the two fmap allocations per layer. The psum column is
    # reproduced exactly by n * m * e * F words, so it is checked here, to the 0.05 KB the
    # table's one printed decimal is rounded to.
    table_psum_bytes = layer["map_glb_psum_kb"] * 1024
    assert (
        abs(glb_psum_bytes - table_psum_bytes) <= 0.05 * 1024
    ), f"psum allocation {glb_psum_bytes} B is not Table III's {layer['map_glb_psum_kb']} KB"
    # layer['map_glb_ifmap_kb'] is deliberately NOT checked: Table III sizes the ifmap
    # allocation on the *padded* ifmap, so it sits 1.334x (CONV2) and 1.325x (CONV3-5) above
    # what the unpadded expression above gives; only CONV1, which needs no padding, matches.
    # Padding the ifmaps is ISSUES.md issue 3; until then the column is reference data only.

    mapping = OrderedDict()
    mapping["used_pe"] = used_pe
    mapping["strips"] = strips
    mapping["ifmap_rows_per_sweep"] = ifmap_rows_per_sweep
    # one pass covers p * t filters of one group and q * r channels
    mapping["pass_m"] = ceil(layer["m_group"] / (p * t))
    mapping["pass_n"] = ceil(layer["c"] / (q * r))
    return mapping


def _traffic(architecture_dict, layer, mapping):
    bitwidth = layer["bitwidth"]
    fmap_glb_width = architecture_dict["ifmap_psum_glb"]["query"]["width"]
    filt_glb_width = architecture_dict["filt_glb"]["query"]["width"]
    dram_width = architecture_dict["dram"]["query"]["width"]
    # data width of each network, taken from its X-bus (the Y-bus adds the col tag)
    ifmap_noc_width = architecture_dict["bus_gin_ifmap_x"]["query"]["width"]
    filter_noc_width = architecture_dict["bus_gin_filt_x"]["query"]["width"]
    psum_gin_width = architecture_dict["bus_gin_psum_x"]["query"]["width"]
    psum_gon_width = architecture_dict["bus_gon_psum_x"]["query"]["width"]

    ifmap_words = layer["H"] * layer["W"] * layer["C"] * layer["batch"]
    filter_words = (
        layer["R"] * layer["S"] * layer["c"] * layer["M"] * layer["filter_loads"]
    )
    ofmap_words = layer["E"] * layer["F"] * layer["M"] * layer["batch"]

    # the fmaps are RLC coded, the filters are not
    dram_read_words = ceil(ifmap_words / RLC_RATIO) + filter_words
    dram_write_words = ceil(ofmap_words / RLC_RATIO)

    # each pass reads the ifmap rows of its strip out of the GLB, and every pass
    # over the filters of a group reads them again
    glb_ifmap_words = (
        mapping["ifmap_rows_per_sweep"]
        * layer["W"]
        * layer["C"]
        * layer["batch"]
        * mapping["pass_m"]
    )
    # the filters of each strip are loaded separately
    glb_filter_words = filter_words * mapping["strips"]
    # psums bounce through the GLB between the channel passes
    glb_psum_words = ofmap_words * (mapping["pass_n"] - 1)

    traffic = OrderedDict()
    # counted in dram_width-bit link transfers
    traffic["dram_read"] = ceil(dram_read_words * bitwidth / dram_width)
    traffic["dram_write"] = ceil(dram_write_words * bitwidth / dram_width)
    # counted in words of each GLB's width
    traffic["ifmap_psum_glb_read"] = ceil(
        (glb_ifmap_words + glb_psum_words) * bitwidth / fmap_glb_width
    )
    traffic["ifmap_psum_glb_write"] = ceil(
        (ofmap_words + glb_psum_words) * bitwidth / fmap_glb_width
    )
    traffic["filt_glb_read"] = ceil(glb_filter_words * bitwidth / filt_glb_width)
    # counted in transfers of each network's data width
    traffic["ifmap_noc"] = ceil(glb_ifmap_words * bitwidth / ifmap_noc_width)
    traffic["filter_noc"] = ceil(glb_filter_words * bitwidth / filter_noc_width)
    traffic["psum_gin"] = ceil(glb_psum_words * bitwidth / psum_gin_width)
    traffic["psum_gon"] = ceil(
        (glb_psum_words + ofmap_words) * bitwidth / psum_gon_width
    )
    # each filter row is written into the filter spad of every PE in its set row, and a set
    # row is map_e PEs wide: the segments of a row partition e, so they sum back to it
    traffic["filter_spad_write"] = glb_filter_words * layer["map_e"]
    return traffic


def _pe_words(layer):
    """Words moved through the PE fifos, summed over all PEs (pe_ctrl.v).

    Per row of F outputs, a PE loads q * (S + (F - 1) * U) ifmaps (S for the first
    window, U for each next one) for p * q * S * F MACs, and writes p * F psums out.
    """
    p, q = layer["map_p"], layer["map_q"]
    S, F, U = layer["S"], layer["F"], layer["U"]
    ofmap_words = layer["E"] * F * layer["M"] * layer["batch"]

    words = OrderedDict()
    words["ifmap"] = layer["macs"] * (S + (F - 1) * U) / (p * S * F)
    words["psum_out"] = layer["macs"] / (q * S)
    # a psum written out is accumulated by the PE above or leaves on the GON, and every GON
    # word but the final ofmap comes back on the GIN to be accumulated
    words["psum_acc"] = words["psum_out"] - ofmap_words
    return words


def _conv(architecture_dict, workload_dict, index):
    performance_dict = OrderedDict()

    layer = _layer(workload_dict, index)
    mapping = _mapping(architecture_dict, layer)
    traffic = _traffic(architecture_dict, layer, mapping)
    pe_words = _pe_words(layer)

    pe_total = get_prod(architecture_dict["multiplier"]["instance"])

    # one pe_mac event fires pe_total MACs; latency scaled by pe_total / used_pe
    mac_events = layer["macs"] / pe_total
    utilization_factor = pe_total / mapping["used_pe"]
    # a zero ifmap gates the filter spad read and the MAC datapath, but the cycle is still spent
    zero = layer["ifmap_zero_ratio"]

    pe_mac_dict = OrderedDict(
        {
            "count": mac_events * (1 - zero),
            "aggregation": "sequential",
            "factor": {
                "cycle_count": utilization_factor,
                "runtime": utilization_factor,
            },
        }
    )
    pe_gated_dict = OrderedDict(
        {
            "count": mac_events * zero,
            "aggregation": "sequential",
            "factor": {
                "cycle_count": utilization_factor,
                "runtime": utilization_factor,
            },
        }
    )
    # one ifmap spad and zero buffer read per map_p MAC slots, gated or not
    pe_ifmap_read_dict = OrderedDict(
        {"count": mac_events / layer["map_p"], "aggregation": "sequential"}
    )
    # the GLB double buffers, so only the off-chip traffic and the filter loads add cycles
    dram_read_dict = OrderedDict(
        {"count": traffic["dram_read"], "aggregation": "sequential"}
    )
    dram_write_dict = OrderedDict(
        {"count": traffic["dram_write"], "aggregation": "sequential"}
    )
    ifmap_psum_glb_read_dict = OrderedDict(
        {"count": traffic["ifmap_psum_glb_read"], "aggregation": "sequential"}
    )
    ifmap_psum_glb_write_dict = OrderedDict(
        {"count": traffic["ifmap_psum_glb_write"], "aggregation": "sequential"}
    )
    filt_glb_read_dict = OrderedDict(
        {"count": traffic["filt_glb_read"], "aggregation": "sequential"}
    )
    # one non-zero MAC reads a weight from the filter SRAM
    spad_read_dict = OrderedDict(
        {"count": layer["macs"] * (1 - zero), "aggregation": "sequential"}
    )
    spad_write_dict = OrderedDict(
        {"count": traffic["filter_spad_write"], "aggregation": "sequential"}
    )
    ifmap_noc_dict = OrderedDict(
        {"count": traffic["ifmap_noc"], "aggregation": "sequential"}
    )
    filter_noc_dict = OrderedDict(
        {"count": traffic["filter_noc"], "aggregation": "sequential"}
    )
    psum_gin_dict = OrderedDict(
        {"count": traffic["psum_gin"], "aggregation": "sequential"}
    )
    psum_gon_dict = OrderedDict(
        {"count": traffic["psum_gon"], "aggregation": "sequential"}
    )
    # PE fifo traffic, in 16b words; it adds no cycle, so the cycle count stays one per MAC as in eyeriss
    pe_ifmap_load_dict = OrderedDict(
        {"count": pe_words["ifmap"], "aggregation": "sequential"}
    )
    pe_filter_load_dict = OrderedDict(
        {"count": traffic["filter_spad_write"], "aggregation": "sequential"}
    )
    pe_psum_acc_dict = OrderedDict(
        {"count": pe_words["psum_acc"], "aggregation": "sequential"}
    )
    pe_psum_out_dict = OrderedDict(
        {"count": pe_words["psum_out"], "aggregation": "sequential"}
    )

    performance_dict["subevent"] = OrderedDict(
        {
            "dram_read": dram_read_dict,
            "dram_write": dram_write_dict,
            "ifmap_psum_glb_read": ifmap_psum_glb_read_dict,
            "ifmap_psum_glb_write": ifmap_psum_glb_write_dict,
            "filt_glb_read": filt_glb_read_dict,
            "spad_read": spad_read_dict,
            "spad_write": spad_write_dict,
            "ifmap_noc_transfer": ifmap_noc_dict,
            "filter_noc_transfer": filter_noc_dict,
            "psum_gin_transfer": psum_gin_dict,
            "psum_gon_transfer": psum_gon_dict,
            "pe_mac": pe_mac_dict,
            "pe_gated": pe_gated_dict,
            "pe_ifmap_read": pe_ifmap_read_dict,
            "pe_ifmap_load": pe_ifmap_load_dict,
            "pe_filter_load": pe_filter_load_dict,
            "pe_psum_acc": pe_psum_acc_dict,
            "pe_psum_out": pe_psum_out_dict,
        }
    )
    return performance_dict


def alexnet(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    layer_dict = OrderedDict({"count": 1, "aggregation": "sequential"})
    performance_dict["subevent"] = OrderedDict(
        {
            "conv1": layer_dict,
            "conv2": OrderedDict(layer_dict),
            "conv3": OrderedDict(layer_dict),
            "conv4": OrderedDict(layer_dict),
            "conv5": OrderedDict(layer_dict),
        }
    )
    return performance_dict


def conv1(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 0)


def conv2(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 1)


def conv3(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 2)


def conv4(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 3)


def conv5(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 4)


def dram_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict["dram"]["query"]["frequency"]

    # the RLC decoder is clocked once per transfer, uncoded filters included
    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    dram_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "read"})}
    )
    rlc_dict = OrderedDict({"count": 1})
    performance_dict["subevent"] = OrderedDict(
        {"dram": dram_dict, "rlc_decoder": rlc_dict}
    )

    return performance_dict


def dram_write(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict["dram"]["query"]["frequency"]

    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    dram_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "write"})}
    )
    relu_dict = OrderedDict({"count": 1})
    rlc_dict = OrderedDict({"count": 1})
    performance_dict["subevent"] = OrderedDict(
        {"dram": dram_dict, "relu": relu_dict, "rlc_encoder": rlc_dict}
    )

    return performance_dict


def ifmap_psum_glb_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # overlapped with the MAC cycles, so it costs energy but no extra cycle
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    glb_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "read"})}
    )
    performance_dict["subevent"] = OrderedDict({"ifmap_psum_glb": glb_dict})

    return performance_dict


def ifmap_psum_glb_write(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    glb_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "write"})}
    )
    performance_dict["subevent"] = OrderedDict({"ifmap_psum_glb": glb_dict})

    return performance_dict


def filt_glb_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # the read feeds filter_noc_transfer, which carries the cycle
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    glb_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "read"})}
    )
    performance_dict["subevent"] = OrderedDict({"filt_glb": glb_dict})

    return performance_dict


def spad_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # the ifmap and psum spad accesses are charged by the pe_* events
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    filter_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "read"})}
    )
    performance_dict["subevent"] = OrderedDict({"filter_spad": filter_dict})

    return performance_dict


def spad_write(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # the load's cycles are charged on filter_noc_transfer
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    filter_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "write"})}
    )
    performance_dict["subevent"] = OrderedDict({"filter_spad": filter_dict})

    return performance_dict


def ifmap_noc_transfer(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        {
            "bus_gin_ifmap_y": OrderedDict({"count": 1}),
            "bus_gin_ifmap_x": OrderedDict({"count": 1}),
        }
    )

    return performance_dict


def filter_noc_transfer(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict["bus_gin_filt_y"]["query"]["frequency"]

    # pass ramp-up (Sec. VI-A): the filter spad can't hold two passes' filters, so each pass
    # waits for its filters, one single-cycle multicast per transfer
    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    performance_dict["subevent"] = OrderedDict(
        {
            "bus_gin_filt_y": OrderedDict({"count": 1}),
            "bus_gin_filt_x": OrderedDict({"count": 1}),
        }
    )

    return performance_dict


def psum_gin_transfer(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        {
            "bus_gin_psum_y": OrderedDict({"count": 1}),
            "bus_gin_psum_x": OrderedDict({"count": 1}),
        }
    )

    return performance_dict


def psum_gon_transfer(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        {
            "bus_gon_psum_x": OrderedDict({"count": 1}),
            "bus_gon_psum_y": OrderedDict({"count": 1}),
        }
    )

    return performance_dict


def _pe_cycle(architecture_dict, modules):
    """One PE-array cycle: each of the PEs uses every module in `modules` once."""
    performance_dict = OrderedDict()
    frequency = architecture_dict["multiplier"]["query"]["frequency"]
    pe_total = get_prod(architecture_dict["multiplier"]["instance"])

    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    performance_dict["subevent"] = OrderedDict(
        (module, OrderedDict({"count": pe_total})) for module in modules
    )
    return performance_dict


def _pe_word(modules):
    """One 16b word through a PE: every module in `modules` is used `modules[module]` times."""
    performance_dict = OrderedDict()

    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        (module, OrderedDict({"count": count})) for module, count in modules.items()
    )
    return performance_dict


def pe_mac(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # S1 reads the psum spad (the ifmap spad read is pe_ifmap_read), S2 multiplies and
    # muxes the psum reset, S3 adds and writes the psum spad back (pe.v, psum_acc.v)
    return _pe_cycle(
        architecture_dict,
        [
            "ifmap_reg",
            "filt_reg",
            "multiplier",
            "mult_gate_mux",
            "psum_spad",
            "psum_spad_reg",
            "psum_op_reg",
            "psum_rst_mux",
            "psum_sel_mux",
            "adder",
            "pe_control",
        ],
    )


def pe_gated(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # a zero ifmap skips the filter spad read and the MAC datapath (JSSC'17 Sec. V-C): the filt
    # reg, the multiplier and the adder are gated. The psum spad read address holds and the
    # write-back is skipped, so the psum spad and the psum regs hold too (pe.v); the spad write
    # that clears a first accumulation is not charged. Both operand regs (ifmap reg, filt reg)
    # hold on a zero ifmap (o_op_en = r_v1 & r_mac1 & ~r_zero1, pe_ctrl.v:333); an incoming
    # psum is its own accumulate op (psum_sel, psum_acc.v) charged by pe_psum_acc, so it is
    # never gated here
    return _pe_cycle(
        architecture_dict,
        [
            "mult_gate_mux",
            "psum_rst_mux",
            "psum_sel_mux",
            "pe_control",
        ],
    )


def pe_ifmap_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # u is the innermost loop and the ifmap/zero buffer address depends only on k, j, so one
    # read serves map_p MACs (pe_ctrl.v:201-221)
    pe_total = get_prod(architecture_dict["multiplier"]["instance"])
    return _pe_word({"ifmap_spad": pe_total, "zero_buffer": pe_total})


def pe_ifmap_load(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # the fifo read is written into the ifmap spad and its zero flag into the zero buffer (pe.v:213-238)
    return _pe_word({"pe_ifmap_fifo": 1, "ifmap_spad": 1, "zero_buffer": 1})


def pe_filter_load(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # one fifo read moves out_width / reg_width filter words
    fifo = architecture_dict["pe_filt_fifo"]["query"]
    return _pe_word({"pe_filt_fifo": fifo["reg_width"] / fifo["out_width"]})


def pe_psum_acc(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # an S_ACC op (pe_ctrl.v) adds the fifo psum to the psum spad entry; the input mux
    # switches once per packed word
    words_per_pack = (
        architecture_dict["psum_in_mux"]["query"]["width"]
        // architecture_dict["multiplier"]["query"]["width"]
    )
    return _pe_word(
        {
            "psum_in_mux": 1 / words_per_pack,
            "pe_psum_ififo": 1,
            "psum_spad": 1,
            "psum_spad_reg": 1,
            "psum_op_reg": 1,
            "psum_rst_mux": 1,
            "psum_sel_mux": 1,
            "adder": 1,
        }
    )


def pe_psum_out(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _pe_word({"pe_psum_ofifo": 1})
