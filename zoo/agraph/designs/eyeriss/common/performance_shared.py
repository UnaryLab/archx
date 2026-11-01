"""Performance model parts shared by the cg and fg Eyeriss variants: layer, mapping, traffic and every non-PE event."""

from collections import OrderedDict
from math import ceil
from archx.utils import get_prod

# RLC (JSSC 2017 Sec. IV-B, Fig. 8): a 64b word packs three (5b run, 16b level) pairs, and a
# run stops at 31 zeros
RLC_PAIRS_PER_WORD = 3
RLC_MAX_RUN = 31
# psum spad entries inside pe (pe.v PSUM_SPAD_WIDTH)
PSUM_SPAD_DEPTH = 24
# ifmap spad entries inside pe (pe.v IFMAP_SPAD_WIDTH); it holds q sliding windows of S values
IFMAP_SPAD_DEPTH = 12


def _sram_bytes(architecture_dict, module):
    query = architecture_dict[module]["query"]
    instances = get_prod(architecture_dict[module]["instance"])
    return query["width"] * query["depth"] * query["bank"] * instances // 8


def _rlc_ratio(zero_ratio, bitwidth, word_width):
    """RLC compression ratio of an fmap whose zeros fall independently at `zero_ratio`.

    A pair codes the run of zeros before a value plus the value, and a run stops at
    RLC_MAX_RUN zeros, so a pair covers sum(z**k for k <= RLC_MAX_RUN) values on average.
    """
    values_per_pair = sum(zero_ratio**k for k in range(RLC_MAX_RUN + 1))
    return values_per_pair * bitwidth * RLC_PAIRS_PER_WORD / word_width


def _inside_taps(out, taps, stride, pad, size):
    """Share of the out * taps filter taps along one dimension that miss the padding border."""
    inside = sum(
        pad <= x * stride + i < size - pad for x in range(out) for i in range(taps)
    )
    return inside / (out * taps)


def _layer(workload_dict, index):
    configuration = workload_dict["configuration"]
    H, W, C = configuration["conv_ifmap"][index]
    R, S = configuration["conv_filter"][index]
    E, F = configuration["conv_ofmap"][index]
    M = configuration["conv_num_filter"][index]
    g = configuration["conv_group"][index]
    U = configuration["conv_stride"][index]
    P = configuration["conv_pad"][index]
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
    zero = configuration["ifmap_zero_ratio"][index]
    # the padding border adds zeros: the share of the padded ifmap that is zero, which RLC codes,
    layer["padded_zero_ratio"] = 1 - (1 - zero) * (H - 2 * P) * (W - 2 * P) / (H * W)
    # and the share of the MACs that see a zero ifmap, which the zero buffer gates
    layer["mac_zero_ratio"] = 1 - (1 - zero) * _inside_taps(
        E, R, U, P, H
    ) * _inside_taps(F, S, U, P, W)
    layer["ofmap_zero_ratio"] = configuration["ofmap_zero_ratio"][index]
    # every fmap in DRAM is RLC coded except the input image of the first layer (Sec. IV-B)
    layer["ifmap_rlc"] = index > 0
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


def _mapping(architecture_dict, layer, pe_module):
    """Map the layer with the JSSC 2017 Table III parameters.

    The mapping is read from the paper, not derived from the storage capacity;
    the capacity is only checked against it below.
    """
    pe_rows, pe_cols = (
        architecture_dict[pe_module]["instance"][0],
        architecture_dict[pe_module]["instance"][1],
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

    # spad capacity (Sec. IV-A): a PE holds p filters x q channels of one S-long filter row,
    # p psums, and q S-entry ifmap windows
    assert (
        p * q * S <= filter_depth
    ), f"filter spad needs {p * q * S} of {filter_depth} entries"
    # filter_noc_transfer stalls the pass, which holds only if the next pass's filters can't preload
    assert (
        2 * p * q * S > filter_depth
    ), f"filter spad can double buffer {p * q * S} entries, so the pass ramp-up would hide"
    assert p <= PSUM_SPAD_DEPTH, f"psum spad needs {p} of {PSUM_SPAD_DEPTH} entries"
    assert (
        q * S <= IFMAP_SPAD_DEPTH
    ), f"ifmap spad needs {q * S} of {IFMAP_SPAD_DEPTH} entries"

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

    # Table III also publishes the two fmap allocations per layer, and the expressions above
    # reproduce both, to the 0.05 KB the table's one printed decimal is rounded to
    table_ifmap_bytes = layer["map_glb_ifmap_kb"] * 1024
    assert (
        abs(glb_ifmap_bytes - table_ifmap_bytes) <= 0.05 * 1024
    ), f"ifmap allocation {glb_ifmap_bytes} B is not Table III's {layer['map_glb_ifmap_kb']} KB"
    table_psum_bytes = layer["map_glb_psum_kb"] * 1024
    assert (
        abs(glb_psum_bytes - table_psum_bytes) <= 0.05 * 1024
    ), f"psum allocation {glb_psum_bytes} B is not Table III's {layer['map_glb_psum_kb']} KB"

    mapping = OrderedDict()
    mapping["used_pe"] = used_pe
    mapping["strips"] = strips
    mapping["ifmap_rows_per_sweep"] = ifmap_rows_per_sweep
    # one pass covers p * t filters of one group and q * r channels
    mapping["pass_m"] = ceil(layer["m_group"] / (p * t))
    mapping["pass_n"] = ceil(layer["c"] / (q * r))
    # the GLB holds the psums of only m ofmap channels, so each m-filter chunk of a group
    # streams the group's ifmaps in from DRAM again (Sec. IV-A, Fig. 7)
    mapping["ifmap_fetches"] = ceil(layer["m_group"] / m)
    return mapping


def _traffic(architecture_dict, layer, mapping):
    bitwidth = layer["bitwidth"]
    fmap_glb_width = architecture_dict["ifmap_psum_glb"]["query"]["width"]
    filt_glb_width = architecture_dict["filt_glb"]["query"]["width"]
    dram_width = architecture_dict["dram"]["query"]["width"]
    relu_width = architecture_dict["relu"]["query"]["pack_width"]
    # data width of each network, taken from its X-bus (the Y-bus adds the col tag)
    ifmap_noc_width = architecture_dict["bus_gin_ifmap_x"]["query"]["width"]
    filter_noc_width = architecture_dict["bus_gin_filt_x"]["query"]["width"]
    psum_gin_width = architecture_dict["bus_gin_psum_x"]["query"]["width"]
    psum_gon_width = architecture_dict["bus_gon_psum_x"]["query"]["width"]

    filter_words = (
        layer["R"] * layer["S"] * layer["c"] * layer["M"] * layer["filter_loads"]
    )
    ofmap_words = layer["E"] * layer["F"] * layer["M"] * layer["batch"]

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

    # the GLB holds one strip of n * q * r ifmap channels, so every ifmap fetch streams the
    # strips in from DRAM, rows shared by neighboring strips included
    dram_ifmap_words = (
        mapping["ifmap_rows_per_sweep"]
        * layer["W"]
        * layer["C"]
        * layer["batch"]
        * mapping["ifmap_fetches"]
    )
    # the filter GLB only holds the next pass's filters (Sec. V-A), so every filter read
    # out of it was first fetched from DRAM
    dram_filter_words = glb_filter_words
    # the fmaps are RLC coded, the filters are not; one RLC word is one link transfer
    ifmap_ratio = (
        _rlc_ratio(layer["padded_zero_ratio"], bitwidth, dram_width)
        if layer["ifmap_rlc"]
        else 1
    )
    ofmap_ratio = _rlc_ratio(layer["ofmap_zero_ratio"], bitwidth, dram_width)
    dram_ifmap_coded_words = ceil(dram_ifmap_words / ifmap_ratio)
    dram_read_words = dram_ifmap_coded_words + dram_filter_words
    dram_write_words = ceil(ofmap_words / ofmap_ratio)

    traffic = OrderedDict()
    # counted in dram_width-bit link transfers
    traffic["dram_read"] = ceil(dram_read_words * bitwidth / dram_width)
    # the ifmap part of dram_read; the rest are filter reads
    traffic["dram_ifmap_read"] = ceil(dram_ifmap_coded_words * bitwidth / dram_width)
    traffic["dram_write"] = ceil(dram_write_words * bitwidth / dram_width)
    # the RLC decoder writes one decoded 16b value into the GLB per cycle, and only coded
    # ifmaps pass through it (Fig. 2): the filters and the first layer's image skip it
    traffic["rlc_decode"] = dram_ifmap_words if layer["ifmap_rlc"] else 0
    # every ofmap leaves through ReLU, one packed word at a time, and then the RLC encoder,
    # which takes one 16b value per cycle (Fig. 2)
    traffic["ofmap_relu"] = ceil(ofmap_words * bitwidth / relu_width)
    traffic["rlc_encode"] = ofmap_words
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


def conv_subevents(architecture_dict, workload_dict, index, pe_module):
    """Return (subevent, layer, mapping, traffic, mac_events) of conv layer `index`.

    subevent holds the non-PE events plus pe_mac and pe_gated; `pe_module` is the module
    whose instance gives the PE array shape.
    """
    layer = _layer(workload_dict, index)
    mapping = _mapping(architecture_dict, layer, pe_module)
    traffic = _traffic(architecture_dict, layer, mapping)

    pe_total = get_prod(architecture_dict[pe_module]["instance"])

    # one pe_mac event fires pe_total MACs; latency scaled by pe_total / used_pe
    mac_events = layer["macs"] / pe_total
    utilization_factor = pe_total / mapping["used_pe"]
    # a zero ifmap gates the filter spad read and the MAC datapath, but the cycle is still spent
    zero = layer["mac_zero_ratio"]

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
    # the GLB double buffers, so only the ifmap loads, the ofmap dumps and the filter loads
    # into the PEs add cycles. The GLB preloads the next pass's filters while the PE array
    # works (Sec. V-A), so only the ifmap share of the DRAM reads stops processing (Sec. VI-A)
    ifmap_share = traffic["dram_ifmap_read"] / traffic["dram_read"]
    # the preload hides only if the link brings the filters in before the MACs finish
    filter_read_ms = (
        (traffic["dram_read"] - traffic["dram_ifmap_read"])
        / 1000
        / architecture_dict["dram"]["query"]["frequency"]
    )
    mac_ms = (
        layer["macs"]
        / mapping["used_pe"]
        / 1000
        / architecture_dict[pe_module]["query"]["frequency"]
    )
    assert (
        filter_read_ms <= mac_ms
    ), f"filter preload takes {filter_read_ms:.2f} ms, longer than the {mac_ms:.2f} ms of MACs"
    dram_read_dict = OrderedDict(
        {
            "count": traffic["dram_read"],
            "aggregation": "sequential",
            "factor": {"cycle_count": ifmap_share, "runtime": ifmap_share},
        }
    )
    dram_write_dict = OrderedDict(
        {"count": traffic["dram_write"], "aggregation": "sequential"}
    )
    rlc_decode_dict = OrderedDict(
        {"count": traffic["rlc_decode"], "aggregation": "sequential"}
    )
    ofmap_relu_dict = OrderedDict(
        {"count": traffic["ofmap_relu"], "aggregation": "sequential"}
    )
    rlc_encode_dict = OrderedDict(
        {"count": traffic["rlc_encode"], "aggregation": "sequential"}
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

    subevent = OrderedDict(
        {
            "dram_read": dram_read_dict,
            "dram_write": dram_write_dict,
            "rlc_decode": rlc_decode_dict,
            "ofmap_relu": ofmap_relu_dict,
            "rlc_encode": rlc_encode_dict,
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
        }
    )
    return subevent, layer, mapping, traffic, mac_events


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


def dram_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict["dram"]["query"]["frequency"]

    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    dram_dict = OrderedDict(
        {"count": 1, "operation": OrderedDict({"dynamic_energy": "read"})}
    )
    performance_dict["subevent"] = OrderedDict({"dram": dram_dict})

    return performance_dict


def rlc_decode(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # one decoded value; it overlaps with the dram_read transfers, which carry the cycles
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        {"rlc_decoder": OrderedDict({"count": 1})}
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
    performance_dict["subevent"] = OrderedDict({"dram": dram_dict})

    return performance_dict


def ofmap_relu(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # one packed ofmap word; it overlaps with the dram_write transfers, which carry the cycles
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict({"relu": OrderedDict({"count": 1})})

    return performance_dict


def rlc_encode(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    performance_dict = OrderedDict()

    # one ofmap value; it overlaps with the dram_write transfers, which carry the cycles
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        {"rlc_encoder": OrderedDict({"count": 1})}
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

    # the ifmap and psum spad accesses are charged by the variant's PE events
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
