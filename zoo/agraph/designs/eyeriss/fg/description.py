from archx.programming.graph.agraph import AGraph

from zoo.agraph.designs.eyeriss.common.description_shared import (
    BITWIDTH,
    DESIGN_PATH,
    PACK_WIDTH,
    PE_ARRAY,
    add_attributes,
    add_metrics,
    add_shared_events,
    add_shared_modules,
    add_workload,
)

PERFORMANCE = f"{DESIGN_PATH}/fg/performance.py"


def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    add_attributes(architecture)

    # fmt: off
    # pe.v split into its submodules, one of each per PE; the "pe" tag sums them into the pe row
    # the four 4-entry PE fifos (pe.v:193, pe.v:241, psum_acc.v:51, psum_acc.v:113), each a fifo_pack synthesized at its in/out widths
    architecture.add_module(name="pe_ifmap_fifo", instance=PE_ARRAY, tag=["pe", "onchip", "fifo"], query={"class": "pe_ifmap_fifo", "in_width": BITWIDTH, "reg_width": BITWIDTH, "out_width": BITWIDTH, "depth": 4})
    architecture.add_module(name="pe_filt_fifo", instance=PE_ARRAY, tag=["pe", "onchip", "fifo"], query={"class": "pe_filt_fifo", "in_width": PACK_WIDTH, "reg_width": BITWIDTH, "out_width": BITWIDTH, "depth": 4})
    architecture.add_module(name="pe_psum_ififo", instance=PE_ARRAY, tag=["pe", "onchip", "fifo"], query={"class": "pe_psum_ififo", "in_width": PACK_WIDTH, "reg_width": BITWIDTH, "out_width": BITWIDTH, "depth": 4})
    architecture.add_module(name="pe_psum_ofifo", instance=PE_ARRAY, tag=["pe", "onchip", "fifo"], query={"class": "pe_psum_ofifo", "in_width": BITWIDTH, "reg_width": BITWIDTH, "out_width": PACK_WIDTH, "depth": 4})
    # register-file spads: ifmap 12x16b, psum 24x16b; the zero buffer keeps a 1b zero flag per ifmap spad entry.
    # Each has a read and a write power report, picked by the event's operation
    architecture.add_module(name="ifmap_spad", instance=PE_ARRAY, tag=["pe", "onchip", "spad", "reg_file", "ifmap"], query={"class": "ifmap_spad", "width": BITWIDTH, "depth": 12})
    architecture.add_module(name="psum_spad", instance=PE_ARRAY, tag=["pe", "onchip", "spad", "reg_file", "psum"], query={"class": "psum_spad", "width": BITWIDTH, "depth": 24})
    architecture.add_module(name="zero_buffer", instance=PE_ARRAY, tag=["pe", "onchip", "spad", "reg_file", "zero_buffer"], query={"class": "zero_buffer", "width": BITWIDTH, "depth": 12})
    # 2-stage pipelined multiplier (pe.v:285): both stages, without the pipeline register between them (mult_stage_reg)
    architecture.add_module(name="multiplier", instance=PE_ARRAY, tag=["pe", "onchip", "mac"], query={"class": "multiplier_fixed_point", "width": BITWIDTH})
    # the multiplier's stage-1 register, which holds the full 2*WIDTH-bit product (r_prod, multiplier_fixed_point.v:24)
    architecture.add_module(name="mult_stage_reg", instance=PE_ARRAY, tag=["pe", "onchip", "control"], query={"class": "register", "width": 2 * BITWIDTH})
    # psum adder (psum_acc.v:104)
    architecture.add_module(name="adder", instance=PE_ARRAY, tag=["pe", "onchip", "mac"], query={"class": "adder_fixed_point", "width": BITWIDTH})
    # multiplier operand regs (pe.v:259) and psum regs (psum_acc.v:85)
    architecture.add_module(name=["ifmap_reg"], instance=PE_ARRAY, tag=["pe", "onchip", "spad", "ifmap"], query={"class": "register", "width": BITWIDTH})
    architecture.add_module(name=["filt_reg"], instance=PE_ARRAY, tag=["pe", "onchip", "spad", "filt"], query={"class": "register", "width": BITWIDTH})
    architecture.add_module(name=["psum_spad_reg"], instance=PE_ARRAY, tag=["pe", "onchip", "spad", "psum"], query={"class": "register", "width": BITWIDTH})
    architecture.add_module(name=["psum_op_reg"], instance=PE_ARRAY, tag=["pe", "onchip", "control"], query={"class": "register", "width": BITWIDTH})
    # zero-ifmap product gate (pe.v:299), psum reset (psum_acc.v:95), adder input select fifo/multiplier (psum_acc.v:101);
    # Fig. 12 doesn't draw the product gate, but the held operands still give a product, so the RTL needs it
    architecture.add_module(name=["mult_gate_mux", "psum_rst_mux", "psum_sel_mux"], instance=PE_ARRAY, tag=["pe", "onchip", "control"], query={"class": "mux_2_to_1", "width": BITWIDTH})
    # psum input source, local network (PE below) or psum GIN, on the packed word (pe.v:132); it also stands in for
    # the local network's 64b PE-to-PE bus (Sec. V-B3), which has no module of its own
    architecture.add_module(name="psum_in_mux", instance=PE_ARRAY, tag=["pe", "onchip", "control"], query={"class": "mux_2_to_1", "width": PACK_WIDTH})
    # pe_ctrl and the PE clock gate (pe.v:108, pe.v:143)
    architecture.add_module(name="pe_control", instance=PE_ARRAY, tag=["pe", "onchip", "control"], query={"class": "pe_control"})
    # fmt: on
    add_shared_modules(architecture, ["pe", "memory", "onchip", "spad", "filt"])

    add_shared_events(
        event,
        PERFORMANCE,
        [
            "pe_ifmap_read",
            "pe_ifmap_load",
            "pe_filter_load",
            "pe_psum_acc",
            "pe_psum_out",
            "pe_psum_spad_read",
            "pe_psum_spad_write",
        ],
    )
    # fmt: off
    # one PE cycle of a non-zero MAC: every datapath stage switches (its psum spad read and write are pe_psum_spad_read/write)
    event.add_event(name="pe_mac", subevent=["ifmap_reg", "filt_reg", "multiplier", "mult_stage_reg", "mult_gate_mux", "psum_spad_reg", "psum_op_reg", "psum_rst_mux", "psum_sel_mux", "adder", "pe_control"], performance=PERFORMANCE)
    # one PE cycle of a zero-ifmap MAC: the ifmap reg, filt reg, multiplier, adder, psum spad and psum regs are gated, the muxes and control still switch (the ifmap spad and zero buffer read is pe_ifmap_read)
    event.add_event(name="pe_gated", subevent=["mult_gate_mux", "psum_rst_mux", "psum_sel_mux", "pe_control"], performance=PERFORMANCE)
    # one ifmap spad and zero buffer read per PE, shared by map_p MACs
    event.add_event(name="pe_ifmap_read", subevent=["ifmap_spad", "zero_buffer"], performance=PERFORMANCE)
    # one ifmap word into a PE: ifmap fifo, then the ifmap spad and zero buffer writes
    event.add_event(name="pe_ifmap_load", subevent=["pe_ifmap_fifo", "ifmap_spad", "zero_buffer"], performance=PERFORMANCE)
    # one filter word into a PE through the filt fifo (the filter spad write is spad_write)
    event.add_event(name="pe_filter_load", subevent=["pe_filt_fifo"], performance=PERFORMANCE)
    # one psum word from the PE below or the psum GIN, added to the psum spad entry (its read and write are pe_psum_spad_read/write)
    event.add_event(name="pe_psum_acc", subevent=["psum_in_mux", "pe_psum_ififo", "psum_spad_reg", "psum_op_reg", "psum_rst_mux", "psum_sel_mux", "adder"], performance=PERFORMANCE)
    # one psum word out of a PE, to the PE above or the psum GON
    event.add_event(name="pe_psum_out", subevent=["pe_psum_ofifo"], performance=PERFORMANCE)
    # one psum spad entry read and written back, by a non-zero MAC or an accumulated psum. An event charges a module
    # with one operation only, so read and write are two events; they sit beside pe_mac, not under it, because an
    # event with child events drops its own cycle count
    event.add_event(name="pe_psum_spad_read", subevent=["psum_spad"], performance=PERFORMANCE)
    event.add_event(name="pe_psum_spad_write", subevent=["psum_spad"], performance=PERFORMANCE)
    # fmt: on

    add_metrics(metric)
    add_workload(workload)

    return agraph.generate()
