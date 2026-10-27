from archx.programming.graph.agraph import AGraph

PERFORMANCE = "zoo/agraph/designs/eyeriss_fg/performance.py"


def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ################ Architecture ################
    ##############################################
    # RTL modules use the NanGate45 syn/PnR csv tables, scaled to 65 nm by the interface; cacti7 modules set their own interface
    architecture.add_attributes(
        technology=65, frequency=200, interface="eyeriss_45nm-65nm"
    )

    # 'class' is the RTL module name in A-Graph_CMOS_PnR/verilog_modules
    pe_array = [12, 14]  # 12x14 = 168 PEs (12 X-buses of 14 PEs each)
    bitwidth = 16
    pack_width = 64  # packed 4x16b word of the PE fifos, filter/psum buses and RLC
    link_width = 64
    sram_device_type = "itrs-lop"  # cacti7 device type for the on-chip SRAMs

    # fmt: off
    # pe.v split into its submodules, one of each per PE; the "pe" tag sums them into the pe row
    # the four 8-entry PE fifos (pe.v:193, pe.v:241, psum_acc.v:51, psum_acc.v:113), each a fifo_pack synthesized at its in/out widths
    architecture.add_module(name="pe_ifmap_fifo", instance=pe_array, tag=["pe", "onchip", "fifo"], query={"class": "fifo_pack", "in_width": bitwidth, "reg_width": bitwidth, "out_width": bitwidth, "depth": 8})
    architecture.add_module(name="pe_filt_fifo", instance=pe_array, tag=["pe", "onchip", "fifo"], query={"class": "fifo_pack_64_64", "in_width": pack_width, "reg_width": bitwidth, "out_width": pack_width, "depth": 8})
    architecture.add_module(name="pe_psum_ififo", instance=pe_array, tag=["pe", "onchip", "fifo"], query={"class": "fifo_pack_64_16", "in_width": pack_width, "reg_width": bitwidth, "out_width": bitwidth, "depth": 8})
    architecture.add_module(name="pe_psum_ofifo", instance=pe_array, tag=["pe", "onchip", "fifo"], query={"class": "fifo_pack_16_64", "in_width": bitwidth, "reg_width": bitwidth, "out_width": pack_width, "depth": 8})
    # register-file spads: ifmap 12x16b, psum 24x16b; the zero buffer keeps a 1b zero flag per ifmap spad entry
    architecture.add_module(name="ifmap_spad", instance=pe_array, tag=["pe", "onchip", "spad", "reg_file", "ifmap"], query={"class": "pe_ifmap_spad"})
    architecture.add_module(name="psum_spad", instance=pe_array, tag=["pe", "onchip", "spad", "reg_file", "psum"], query={"class": "pe_psum_spad"})
    architecture.add_module(name="zero_buffer", instance=pe_array, tag=["pe", "onchip", "spad", "reg_file", "zero_buffer"], query={"class": "pe_zero_buffer"})
    # 2-stage pipelined multiplier (pe.v:285): both stages and the pipeline register between them
    architecture.add_module(name="multiplier", instance=pe_array, tag=["pe", "onchip", "mac"], query={"class": "multiplier_fixed_point", "width": bitwidth})
    # psum adder (psum_acc.v:104)
    architecture.add_module(name="adder", instance=pe_array, tag=["pe", "onchip", "mac"], query={"class": "adder_fixed_point", "width": bitwidth})
    # multiplier operand regs (pe.v:259) and psum regs (psum_acc.v:85)
    architecture.add_module(name=["ifmap_reg"], instance=pe_array, tag=["pe", "onchip", "spad", "ifmap"], query={"class": "register", "width": bitwidth})
    architecture.add_module(name=["filt_reg"], instance=pe_array, tag=["pe", "onchip", "spad", "filt"], query={"class": "register", "width": bitwidth})
    architecture.add_module(name=["psum_spad_reg"], instance=pe_array, tag=["pe", "onchip", "spad", "psum"], query={"class": "register", "width": bitwidth})
    architecture.add_module(name=["psum_op_reg"], instance=pe_array, tag=["pe", "onchip", "control"], query={"class": "register", "width": bitwidth})
    # zero-ifmap product gate (pe.v:299), psum reset (psum_acc.v:95), adder input select fifo/multiplier (psum_acc.v:101)
    architecture.add_module(name=["mult_gate_mux", "psum_rst_mux", "psum_sel_mux"], instance=pe_array, tag=["pe", "onchip", "control"], query={"class": "mux_2_to_1", "width": bitwidth})
    # psum input source, local network (PE below) or psum GIN, on the packed word (pe.v:132)
    architecture.add_module(name="psum_in_mux", instance=pe_array, tag=["pe", "onchip", "control"], query={"class": "mux_2_to_1", "width": pack_width})
    # pe_ctrl and the PE clock gate (pe.v:108, pe.v:143)
    architecture.add_module(name="pe_control", instance=pe_array, tag=["pe", "onchip", "control"], query={"class": "pe_control"})

    architecture.add_module(name="filter_spad", instance=pe_array, tag=["pe", "memory", "onchip", "spad", "filt"], query={"class": "sram", "interface": "cacti7", "width": bitwidth, "depth": 224, "bank": 1, "voltage": 1.0, "device_type": sram_device_type})

    architecture.add_module(name="ifmap_psum_glb", instance=[25], tag=["memory", "glb", "onchip"], query={"class": "sram", "interface": "cacti7", "width": link_width, "depth": 512, "bank": 1, "voltage": 1.0, "device_type": sram_device_type})
    architecture.add_module(name="filt_glb", instance=[2], tag=["memory", "glb", "onchip"], query={"class": "sram", "interface": "cacti7", "width": link_width, "depth": 512, "bank": 1, "voltage": 1.0, "device_type": sram_device_type})

    # each network: one Y-bus multicasts to 12 X-buses, each X-bus to 14 PEs
    architecture.add_module(name="bus_gin_ifmap_y", instance=[1], tag=["noc", "onchip", "ifmap_noc"], query={"class": "bus_gin_ifmap_y", "width": 16, "n_dest": 12})
    architecture.add_module(name="bus_gin_ifmap_x", instance=[12], tag=["noc", "onchip", "ifmap_noc"], query={"class": "bus_gin_ifmap_x", "width": 16, "n_dest": 14})
    architecture.add_module(name="bus_gin_filt_y", instance=[1], tag=["noc", "onchip", "filt_noc"], query={"class": "bus_gin_filt_y", "width": 64, "n_dest": 12})
    architecture.add_module(name="bus_gin_filt_x", instance=[12], tag=["noc", "onchip", "filt_noc"], query={"class": "bus_gin_filt_x", "width": 64, "n_dest": 14})
    architecture.add_module(name="bus_gin_psum_y", instance=[1], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gin_psum_y", "width": 64, "n_dest": 12})
    architecture.add_module(name="bus_gin_psum_x", instance=[12], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gin_psum_x", "width": 64, "n_dest": 14})
    architecture.add_module(name="bus_gon_psum_y", instance=[1], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gon_psum_y", "width": 64, "n_dest": 12})
    architecture.add_module(name="bus_gon_psum_x", instance=[12], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gon_psum_x", "width": 64, "n_dest": 14})

    architecture.add_module(name="rlc_decoder", instance=[1], tag=["rlc_decoder", "codec", "onchip"], query={"class": "rlc_decoder", "width": bitwidth, "pack_width": pack_width, "depth": 8})
    architecture.add_module(name="rlc_encoder", instance=[1], tag=["rlc_encoder", "codec", "onchip"], query={"class": "rlc_encoder", "width": bitwidth, "pack_width": pack_width, "depth": 8})
    architecture.add_module(name="relu", instance=[1], tag=["relu", "codec", "onchip"], query={"class": "relu", "width": bitwidth, "pack_width": pack_width})
    # size in bits (1 GB); width is the link width the performance model reads, cacti7 ignores it
    architecture.add_module(name="dram", instance=[1], tag=["memory"], query={"class": "dram", "interface": "cacti7", "size": 8589934592, "width": link_width, "frequency": 60})
    # fmt: on

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(
        name="alexnet",
        subevent=["conv1", "conv2", "conv3", "conv4", "conv5"],
        performance=PERFORMANCE,
    )
    event.add_event(
        name=["conv1", "conv2", "conv3", "conv4", "conv5"],
        subevent=[
            "dram_read",
            "dram_write",
            "ifmap_psum_glb_read",
            "ifmap_psum_glb_write",
            "filt_glb_read",
            "spad_read",
            "spad_write",
            "ifmap_noc_transfer",
            "filter_noc_transfer",
            "psum_gin_transfer",
            "psum_gon_transfer",
            "pe_mac",
            "pe_gated",
            "pe_ifmap_read",
            "pe_ifmap_load",
            "pe_filter_load",
            "pe_psum_acc",
            "pe_psum_out",
        ],
        performance=PERFORMANCE,
    )

    event.add_event(
        name="dram_read", subevent=["dram", "rlc_decoder"], performance=PERFORMANCE
    )
    event.add_event(
        name="dram_write",
        subevent=["dram", "relu", "rlc_encoder"],
        performance=PERFORMANCE,
    )
    event.add_event(
        name="ifmap_psum_glb_read", subevent=["ifmap_psum_glb"], performance=PERFORMANCE
    )
    event.add_event(
        name="ifmap_psum_glb_write",
        subevent=["ifmap_psum_glb"],
        performance=PERFORMANCE,
    )
    event.add_event(
        name="filt_glb_read", subevent=["filt_glb"], performance=PERFORMANCE
    )
    # filter_spad is the only SRAM spad; the ifmap/psum spads are PE register files charged by the pe_* events
    event.add_event(name="spad_read", subevent=["filter_spad"], performance=PERFORMANCE)
    event.add_event(
        name="spad_write", subevent=["filter_spad"], performance=PERFORMANCE
    )
    event.add_event(
        name="ifmap_noc_transfer",
        subevent=["bus_gin_ifmap_y", "bus_gin_ifmap_x"],
        performance=PERFORMANCE,
    )
    event.add_event(
        name="filter_noc_transfer",
        subevent=["bus_gin_filt_y", "bus_gin_filt_x"],
        performance=PERFORMANCE,
    )
    event.add_event(
        name="psum_gin_transfer",
        subevent=["bus_gin_psum_y", "bus_gin_psum_x"],
        performance=PERFORMANCE,
    )
    event.add_event(
        name="psum_gon_transfer",
        subevent=["bus_gon_psum_x", "bus_gon_psum_y"],
        performance=PERFORMANCE,
    )
    # fmt: off
    # one PE cycle of a non-zero MAC: every datapath stage switches
    event.add_event(name="pe_mac", subevent=["ifmap_reg", "filt_reg", "multiplier", "mult_gate_mux", "psum_spad", "psum_spad_reg", "psum_op_reg", "psum_rst_mux", "psum_sel_mux", "adder", "pe_control"], performance=PERFORMANCE)
    # one PE cycle of a zero-ifmap MAC: the ifmap reg, filt reg, multiplier, adder, psum spad and psum regs are gated, the muxes and control still switch (the ifmap spad and zero buffer read is pe_ifmap_read)
    event.add_event(name="pe_gated", subevent=["mult_gate_mux", "psum_rst_mux", "psum_sel_mux", "pe_control"], performance=PERFORMANCE)
    # one ifmap spad and zero buffer read per PE, shared by map_p MACs
    event.add_event(name="pe_ifmap_read", subevent=["ifmap_spad", "zero_buffer"], performance=PERFORMANCE)
    # one ifmap word into a PE: ifmap fifo, then the ifmap spad and zero buffer writes
    event.add_event(name="pe_ifmap_load", subevent=["pe_ifmap_fifo", "ifmap_spad", "zero_buffer"], performance=PERFORMANCE)
    # one filter word into a PE through the filt fifo (the filter spad write is spad_write)
    event.add_event(name="pe_filter_load", subevent=["pe_filt_fifo"], performance=PERFORMANCE)
    # one psum word from the PE below or the psum GIN, added to the psum spad entry
    event.add_event(name="pe_psum_acc", subevent=["psum_in_mux", "pe_psum_ififo", "psum_spad", "psum_spad_reg", "psum_op_reg", "psum_rst_mux", "psum_sel_mux", "adder"], performance=PERFORMANCE)
    # one psum word out of a PE, to the PE above or the psum GON
    event.add_event(name="pe_psum_out", subevent=["pe_psum_ofifo"], performance=PERFORMANCE)
    # fmt: on

    ##############################################
    ###############    Metric    #################
    ##############################################
    metric.add_metric(name="area", unit="mm^2", aggregation="module")
    metric.add_metric(name="leakage_power", unit="mW", aggregation="module")
    metric.add_metric(name="dynamic_energy", unit="nJ", aggregation="summation")
    metric.add_metric(name="cycle_count", unit="cycles", aggregation="specified")
    metric.add_metric(name="runtime", unit="ms", aggregation="specified")

    ##############################################
    ###############   Workload   #################
    ##############################################
    # per-layer lists conv1..conv5; the stride is folded into conv_ofmap
    alexnet = workload.add_configuration(name="alexnet")
    alexnet.add_parameter(
        parameter_name="conv_ifmap",
        parameter_value=[
            [227, 227, 3],
            [27, 27, 96],
            [13, 13, 256],
            [13, 13, 384],
            [13, 13, 384],
        ],
        sweep=False,
    )
    alexnet.add_parameter(
        parameter_name="conv_filter",
        parameter_value=[[11, 11], [5, 5], [3, 3], [3, 3], [3, 3]],
        sweep=False,
    )
    alexnet.add_parameter(
        parameter_name="conv_num_filter",
        parameter_value=[96, 256, 384, 384, 256],
        sweep=False,
    )
    alexnet.add_parameter(
        parameter_name="conv_ofmap",
        parameter_value=[[55, 55], [27, 27], [13, 13], [13, 13], [13, 13]],
        sweep=False,
    )
    alexnet.add_parameter(
        parameter_name="conv_group", parameter_value=[1, 2, 1, 2, 2], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="conv_stride", parameter_value=[4, 1, 1, 1, 1], sweep=False
    )
    alexnet.add_parameter(parameter_name="bitwidth", parameter_value=16, sweep=False)
    # JSSC 2017 Table III (n) and Table V (zeros in ifmaps)
    alexnet.add_parameter(parameter_name="batch", parameter_value=4, sweep=False)
    alexnet.add_parameter(
        parameter_name="images_per_pass", parameter_value=[1, 1, 4, 4, 4], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="ifmap_zero_ratio",
        parameter_value=[0.0001, 0.387, 0.725, 0.793, 0.776],
        sweep=False,
    )

    # JSSC 2017 Table III, RS dataflow mapping parameters (Table II gives the meanings):
    #   m: ofmap channels held in the GLB            n: ifmaps processed by one pass (images_per_pass above)
    #   e: width of the PE set, strip-mined if needed   p: filters processed by a PE set
    #   q: channels processed by a PE set            r: PE sets covering different channels
    #   t: PE sets covering different filters
    # CONV1 is the strip-mined layer: its 11x55 set is mapped as 11x7 (Section IV-A).
    alexnet.add_parameter(
        parameter_name="map_m", parameter_value=[96, 64, 64, 64, 64], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="map_e", parameter_value=[7, 27, 13, 13, 13], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="map_p", parameter_value=[16, 16, 16, 16, 16], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="map_q", parameter_value=[1, 2, 4, 3, 3], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="map_r", parameter_value=[1, 1, 1, 2, 2], sweep=False
    )
    alexnet.add_parameter(
        parameter_name="map_t", parameter_value=[2, 1, 4, 2, 2], sweep=False
    )
    # Table III 'Global Buffer Allocation' columns, in KB. The psum column is cross-checked
    # against the mapping in performance.py; the ifmap column is reference data only, because
    # Table III sizes it on the padded ifmap and the model does not pad yet (ISSUES.md issue 3).
    alexnet.add_parameter(
        parameter_name="map_glb_ifmap_kb",
        parameter_value=[15.5, 3.8, 7.0, 10.5, 10.5],
        sweep=False,
    )
    alexnet.add_parameter(
        parameter_name="map_glb_psum_kb",
        parameter_value=[72.2, 91.1, 84.5, 84.5, 84.5],
        sweep=False,
    )

    return agraph.generate()
