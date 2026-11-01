from collections import OrderedDict
from archx.utils import get_prod

# archx looks up each event's function by name in this file, so the shared event functions are imported here
from zoo.agraph.designs.eyeriss.common.performance_shared import (
    alexnet,
    conv_subevents,
    dram_read,
    dram_write,
    filt_glb_read,
    filter_noc_transfer,
    ifmap_noc_transfer,
    ifmap_psum_glb_read,
    ifmap_psum_glb_write,
    ofmap_relu,
    psum_gin_transfer,
    psum_gon_transfer,
    rlc_decode,
    rlc_encode,
    spad_read,
    spad_write,
)

PE_MODULE = "multiplier"
# the register-file spads, whose csv has one power report per read/write action
RW_SPADS = ("ifmap_spad", "psum_spad", "zero_buffer")


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

    subevent, layer, mapping, traffic, mac_events = conv_subevents(
        architecture_dict, workload_dict, index, PE_MODULE
    )
    pe_words = _pe_words(layer)
    # a word count is summed over all PEs, and the used_pe active PEs run in parallel, so
    # a word that takes a cycle of its PE adds 1 / used_pe array cycles
    pe_word_factor = {
        "cycle_count": 1 / mapping["used_pe"],
        "runtime": 1 / mapping["used_pe"],
    }

    # one ifmap spad and zero buffer read per map_p MAC slots, gated or not
    pe_ifmap_read_dict = OrderedDict(
        {"count": mac_events / layer["map_p"], "aggregation": "sequential"}
    )
    # PE fifo traffic, in 16b words. An ifmap load and a psum accumulation each take a cycle
    # of their PE (S_LD_IFM, S_ACC in pe_ctrl.v); the filter load cycles are carried by
    # filter_noc_transfer, and a psum out is written by the op that made it
    pe_ifmap_load_dict = OrderedDict(
        {
            "count": pe_words["ifmap"],
            "aggregation": "sequential",
            "factor": dict(pe_word_factor),
        }
    )
    pe_filter_load_dict = OrderedDict(
        {"count": traffic["filter_spad_write"], "aggregation": "sequential"}
    )
    pe_psum_acc_dict = OrderedDict(
        {
            "count": pe_words["psum_acc"],
            "aggregation": "sequential",
            "factor": dict(pe_word_factor),
        }
    )
    pe_psum_out_dict = OrderedDict(
        {"count": pe_words["psum_out"], "aggregation": "sequential"}
    )
    # every non-zero MAC and every accumulated psum reads its psum spad entry and writes it back
    psum_spad_accesses = (
        layer["macs"] * (1 - layer["mac_zero_ratio"]) + pe_words["psum_acc"]
    )
    pe_psum_spad_read_dict = OrderedDict(
        {"count": psum_spad_accesses, "aggregation": "sequential"}
    )
    pe_psum_spad_write_dict = OrderedDict(
        {"count": psum_spad_accesses, "aggregation": "sequential"}
    )
    subevent["pe_ifmap_read"] = pe_ifmap_read_dict
    subevent["pe_ifmap_load"] = pe_ifmap_load_dict
    subevent["pe_filter_load"] = pe_filter_load_dict
    subevent["pe_psum_acc"] = pe_psum_acc_dict
    subevent["pe_psum_out"] = pe_psum_out_dict
    subevent["pe_psum_spad_read"] = pe_psum_spad_read_dict
    subevent["pe_psum_spad_write"] = pe_psum_spad_write_dict

    performance_dict["subevent"] = subevent
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


def _pe_cycle(architecture_dict, modules):
    """One PE-array cycle: each of the PEs uses every module in `modules` once."""
    performance_dict = OrderedDict()
    frequency = architecture_dict[PE_MODULE]["query"]["frequency"]
    pe_total = get_prod(architecture_dict[PE_MODULE]["instance"])

    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    performance_dict["subevent"] = OrderedDict(
        (module, OrderedDict({"count": pe_total})) for module in modules
    )
    return performance_dict


def _pe_word(modules, spad_action=None, frequency=None):
    """One 16b word through a PE: every module in `modules` is used `modules[module]` times.

    The RW_SPADS among them are charged with `spad_action`, "read" or "write". Given the PE
    clock `frequency`, the word takes one cycle of its PE; otherwise it adds no cycle.
    """
    performance_dict = OrderedDict()
    cycles = 0 if frequency is None else 1
    runtime = 0 if frequency is None else 1 / 1000 / frequency

    performance_dict["cycle_count"] = OrderedDict({"value": cycles, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": runtime, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict()
    for module, count in modules.items():
        subevent = OrderedDict({"count": count})
        if module in RW_SPADS:
            subevent["operation"] = OrderedDict({"dynamic_energy": spad_action})
        performance_dict["subevent"][module] = subevent
    return performance_dict


def pe_mac(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # S1 reads the psum spad, S2 multiplies and muxes the psum reset, S3 adds and writes the
    # psum spad back (pe.v, psum_acc.v); the spad accesses are pe_ifmap_read and
    # pe_psum_spad_read/write
    return _pe_cycle(
        architecture_dict,
        [
            "ifmap_reg",
            "filt_reg",
            PE_MODULE,
            "mult_stage_reg",
            "mult_gate_mux",
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
    pe_total = get_prod(architecture_dict[PE_MODULE]["instance"])
    return _pe_word({"ifmap_spad": pe_total, "zero_buffer": pe_total}, "read")


def pe_ifmap_load(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # the fifo read is written into the ifmap spad and its zero flag into the zero buffer (pe.v:213-238).
    # S_LD_IFM reads one word per cycle and issues no MAC meanwhile (pe_ctrl.v)
    return _pe_word(
        {"pe_ifmap_fifo": 1, "ifmap_spad": 1, "zero_buffer": 1},
        "write",
        architecture_dict[PE_MODULE]["query"]["frequency"],
    )


def pe_filter_load(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # one fifo read moves out_width / reg_width filter words
    fifo = architecture_dict["pe_filt_fifo"]["query"]
    return _pe_word({"pe_filt_fifo": fifo["reg_width"] / fifo["out_width"]})


def pe_psum_acc(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    # an S_ACC op (pe_ctrl.v) adds the fifo psum to the psum spad entry (its read and write
    # are pe_psum_spad_read/write); the input mux switches once per packed word. The op takes
    # a PE cycle: the adder is shared with the MACs (Fig. 12), and S_ACC issues no MAC
    words_per_pack = (
        architecture_dict["psum_in_mux"]["query"]["width"]
        // architecture_dict[PE_MODULE]["query"]["width"]
    )
    return _pe_word(
        {
            "psum_in_mux": 1 / words_per_pack,
            "pe_psum_ififo": 1,
            "psum_spad_reg": 1,
            "psum_op_reg": 1,
            "psum_rst_mux": 1,
            "psum_sel_mux": 1,
            "adder": 1,
        },
        frequency=architecture_dict[PE_MODULE]["query"]["frequency"],
    )


def pe_psum_out(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _pe_word({"pe_psum_ofifo": 1})


def pe_psum_spad_read(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _pe_word({"psum_spad": 1}, "read")


def pe_psum_spad_write(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _pe_word({"psum_spad": 1}, "write")
