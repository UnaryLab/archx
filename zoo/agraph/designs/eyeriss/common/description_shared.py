"""Description parts shared by the cg and fg Eyeriss variants: everything outside the PE."""

# repo-relative: archx resolves the performance file paths from the archx base dir
DESIGN_PATH = "zoo/agraph/designs/eyeriss"

# 'class' is the RTL module name in A-Graph_CMOS_PnR/verilog_modules
PE_ARRAY = [12, 14]  # 12x14 = 168 PEs (12 X-buses of 14 PEs each)
BITWIDTH = 16
PACK_WIDTH = 64  # packed 4x16b word of the PE fifos, filter/psum buses and RLC
LINK_WIDTH = 64
# GIN col IDs (Sec. V-B1): 4b to address the 14 columns, 5b on the ifmap GIN for up to 32
# ifmap rows on a diagonal
COL_ID_WIDTH = 4
IFMAP_COL_ID_WIDTH = 5
SRAM_DEVICE_TYPE = "itrs-lop"  # cacti7 device type for the on-chip SRAMs

# conv subevents outside the PE; each variant appends its own PE events
CONV_SUBEVENTS = [
    "dram_read",
    "dram_write",
    "rlc_decode",
    "ofmap_relu",
    "rlc_encode",
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
]


def add_attributes(architecture):
    # RTL modules use the NanGate45 syn/PnR csv tables, scaled to 65 nm by the interface; cacti7 modules set their own interface
    architecture.add_attributes(
        technology=65, frequency=200, interface="eyeriss_45nm-65nm"
    )


def add_shared_modules(architecture, filter_spad_tags):
    """Add every module after the PE modules, from filter_spad to dram.

    Call this after the variant's PE modules: count.csv rows follow module insertion order.
    """
    # fmt: off
    architecture.add_module(name="filter_spad", instance=PE_ARRAY, tag=filter_spad_tags, query={"class": "sram", "interface": "cacti7", "width": BITWIDTH, "depth": 224, "bank": 1, "voltage": 1.01, "device_type": SRAM_DEVICE_TYPE})

    architecture.add_module(name="ifmap_psum_glb", instance=[25], tag=["memory", "glb", "onchip"], query={"class": "sram", "interface": "cacti7", "width": LINK_WIDTH, "depth": 512, "bank": 1, "voltage": 1.0, "device_type": SRAM_DEVICE_TYPE})
    architecture.add_module(name="filt_glb", instance=[2], tag=["memory", "glb", "onchip"], query={"class": "sram", "interface": "cacti7", "width": LINK_WIDTH, "depth": 512, "bank": 1, "voltage": 1.0, "device_type": SRAM_DEVICE_TYPE})

    # each network: one Y-bus multicasts to 12 X-buses, each X-bus to 14 PEs. A GIN Y-bus carries
    # the col tag along with the data (Fig. 10), so its width adds the col ID, scaled linearly from
    # the data-width PnR row. The GON Y-bus carries only psum data to the GLB
    architecture.add_module(name="bus_gin_ifmap_y", instance=[1], tag=["noc", "onchip", "ifmap_noc"], query={"class": "bus_gin_ifmap_y", "width": 16 + IFMAP_COL_ID_WIDTH, "n_dest": 12})
    architecture.add_module(name="bus_gin_ifmap_x", instance=[12], tag=["noc", "onchip", "ifmap_noc"], query={"class": "bus_gin_ifmap_x", "width": 16, "n_dest": 14})
    architecture.add_module(name="bus_gin_filt_y", instance=[1], tag=["noc", "onchip", "filt_noc"], query={"class": "bus_gin_filt_y", "width": 64 + COL_ID_WIDTH, "n_dest": 12})
    architecture.add_module(name="bus_gin_filt_x", instance=[12], tag=["noc", "onchip", "filt_noc"], query={"class": "bus_gin_filt_x", "width": 64, "n_dest": 14})
    architecture.add_module(name="bus_gin_psum_y", instance=[1], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gin_psum_y", "width": 64 + COL_ID_WIDTH, "n_dest": 12})
    architecture.add_module(name="bus_gin_psum_x", instance=[12], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gin_psum_x", "width": 64, "n_dest": 14})
    architecture.add_module(name="bus_gon_psum_y", instance=[1], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gon_psum_y", "width": 64, "n_dest": 12})
    architecture.add_module(name="bus_gon_psum_x", instance=[12], tag=["noc", "onchip", "psum_noc"], query={"class": "bus_gon_psum_x", "width": 64, "n_dest": 14})

    architecture.add_module(name="rlc_decoder", instance=[1], tag=["rlc_decoder", "codec", "onchip"], query={"class": "rlc_decoder", "width": BITWIDTH, "pack_width": PACK_WIDTH, "depth": 4})
    architecture.add_module(name="rlc_encoder", instance=[1], tag=["rlc_encoder", "codec", "onchip"], query={"class": "rlc_encoder", "width": BITWIDTH, "pack_width": PACK_WIDTH, "depth": 4})
    architecture.add_module(name="relu", instance=[1], tag=["relu", "codec", "onchip"], query={"class": "relu", "width": BITWIDTH, "pack_width": PACK_WIDTH})
    # size in bits (1 GB); width is the link width the performance model reads, cacti7 ignores it
    architecture.add_module(name="dram", instance=[1], tag=["memory"], query={"class": "dram", "interface": "cacti7", "size": 8589934592, "width": LINK_WIDTH, "frequency": 60})
    # fmt: on


def add_shared_events(event, performance, extra_pe_events=()):
    """Add alexnet, conv1..conv5 and every non-PE event; the variant adds its PE events after.

    extra_pe_events are the variant's conv subevents beyond CONV_SUBEVENTS.
    """
    event.add_event(
        name="alexnet",
        subevent=["conv1", "conv2", "conv3", "conv4", "conv5"],
        performance=performance,
    )
    event.add_event(
        name=["conv1", "conv2", "conv3", "conv4", "conv5"],
        subevent=CONV_SUBEVENTS + list(extra_pe_events),
        performance=performance,
    )

    event.add_event(name="dram_read", subevent=["dram"], performance=performance)
    # one value out of the RLC decoder; filters and the first layer's image bypass it
    event.add_event(
        name="rlc_decode", subevent=["rlc_decoder"], performance=performance
    )
    event.add_event(name="dram_write", subevent=["dram"], performance=performance)
    # the ofmap path to DRAM (Fig. 2): ReLU on one packed word, then one value into the RLC encoder
    event.add_event(name="ofmap_relu", subevent=["relu"], performance=performance)
    event.add_event(
        name="rlc_encode", subevent=["rlc_encoder"], performance=performance
    )
    event.add_event(
        name="ifmap_psum_glb_read", subevent=["ifmap_psum_glb"], performance=performance
    )
    event.add_event(
        name="ifmap_psum_glb_write",
        subevent=["ifmap_psum_glb"],
        performance=performance,
    )
    event.add_event(
        name="filt_glb_read", subevent=["filt_glb"], performance=performance
    )
    # filter_spad is the only SRAM spad; the other spads are charged by the variant's PE events
    event.add_event(name="spad_read", subevent=["filter_spad"], performance=performance)
    event.add_event(
        name="spad_write", subevent=["filter_spad"], performance=performance
    )
    event.add_event(
        name="ifmap_noc_transfer",
        subevent=["bus_gin_ifmap_y", "bus_gin_ifmap_x"],
        performance=performance,
    )
    event.add_event(
        name="filter_noc_transfer",
        subevent=["bus_gin_filt_y", "bus_gin_filt_x"],
        performance=performance,
    )
    event.add_event(
        name="psum_gin_transfer",
        subevent=["bus_gin_psum_y", "bus_gin_psum_x"],
        performance=performance,
    )
    event.add_event(
        name="psum_gon_transfer",
        subevent=["bus_gon_psum_x", "bus_gon_psum_y"],
        performance=performance,
    )


def add_metrics(metric):
    metric.add_metric(name="area", unit="mm^2", aggregation="module")
    metric.add_metric(name="leakage_power", unit="mW", aggregation="module")
    metric.add_metric(name="dynamic_energy", unit="nJ", aggregation="summation")
    metric.add_metric(name="cycle_count", unit="cycles", aggregation="specified")
    metric.add_metric(name="runtime", unit="ms", aggregation="specified")


def add_workload(workload):
    # per-layer lists conv1..conv5; the stride is folded into conv_ofmap
    alexnet = workload.add_configuration(name="alexnet")
    # the padded ifmaps of JSSC 2017 Table III, which the GLB and the PEs process
    alexnet.add_parameter(
        parameter_name="conv_ifmap",
        parameter_value=[
            [227, 227, 3],
            [31, 31, 96],
            [15, 15, 256],
            [15, 15, 384],
            [15, 15, 384],
        ],
        sweep=False,
    )
    # the zero border on each side of the padded ifmap
    alexnet.add_parameter(
        parameter_name="conv_pad", parameter_value=[0, 2, 1, 1, 1], sweep=False
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
    # zeros in the ifmap values, without the padding border (Sec. IV-B gives them as
    # activation statistics); the model adds the border's zeros on top
    alexnet.add_parameter(
        parameter_name="ifmap_zero_ratio",
        parameter_value=[0.0001, 0.387, 0.725, 0.793, 0.776],
        sweep=False,
    )
    # zeros in each layer's ReLU output, which sets its RLC ratio in DRAM. The paper gives
    # none, so each is the next layer's ifmap zeros; pooling follows CONV1 and CONV2, so
    # theirs are lower bounds. CONV5 has no next conv layer and reuses its own ifmap zeros
    alexnet.add_parameter(
        parameter_name="ofmap_zero_ratio",
        parameter_value=[0.387, 0.725, 0.793, 0.776, 0.776],
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
    # Table III 'Global Buffer Allocation' columns, in KB; both are cross-checked against the
    # mapping in performance_shared.py
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
