from archx.programming.graph.agraph import AGraph

PERFORMANCE = "zoo/agraph/designs/tnn/performance.py"

# Sources:
#   Smith = J. E. Smith, arXiv 2011.13844 (MNIST TNN prototypes ECVT, ECCVT, ECCCVT).
#   TNN7  = Nair et al., arXiv 2205.07410 (TNN7 macro library, Table III MNIST PPA).
#   RTL paths are relative to TNNGen/tnn_hw/rtl/src/, the TNN7 source read by tnn_hw/syn_tcl/run_opt_*.tcl.
#   prabhu: = branch origin/prabhu of A-Graph/auto_syn_pnr, the RTL behind the tnn7 interface reports.
#
# TNN7 Table III targets (TNN7 rows; post-synthesis Genus/Joules, TT corner, 0.7 V, 25 C; "derived using
# synaptic count scaling" from single columns, TNN7 Sec. IV-B), kept here for later validation:
#   2-layer (ECVT):    2.25 mW, 41.38 ns,  3.09 mm^2
#   3-layer (ECCVT):  7.57 mW,  66.16 ns, 10.42 mm^2
#   4-layer (ECCCVT): 17.89 mW, 91.58 ns, 24.63 mm^2
# These areas are physical; the tnn7 CSV areas are the layout areas of the 4x-scaled ASAP7 LEF (about 16x
# physical) and are not scaled down here.
# The Table III synapse totals (389K / 1,310K / 3,096K, Smith Table 3) include 2 voters per final layer
# column (324K / 692K / 1,240K synapses), which TNN7 scales as C columns. Voters are not modeled here
# (voting is each column's WTA), so the modeled column synapses are 65K / 618K / 1,857K.

# Smith Table 3: C1 has one column per 3x3 window of the 28x28 image (26x26 = 676), with p = 8 inputs
# (the 4 corner pixels of the window plus their negatives) and q = 12 neurons. Smith lists 12 and 20
# "inputs" for C2 and C3, the previous layer's q; the synapse totals 553K and 1,239K require
# p = 4 * 12 = 48 and p = 4 * 20 = 80, i.e. the 4 corner columns of a 3x3 window of columns.
LAYERS = {
    "c1": {"cols": 676, "p": 8, "q": 12},  # 26x26 columns
    "c2": {"cols": 576, "p": 48, "q": 20},  # 24x24 columns
    "c3": {"cols": 484, "p": 80, "q": 32},  # 22x22 columns
}
# Neuron threshold theta per layer, Smith Table 4. The RTL default is THRESHOLD = 13 (column65x2.sv:40),
# which the single-column TNN7 runs use; set every entry to 13 to model that instead (changes only C1:
# accumulator width M goes from 4 to 5 bits).
THRESHOLD = {"c1": 4, "c2": 8, "c3": 8}
# Smith ECVT, ECCVT, ECCCVT = TNN7 2-, 3-, 4-layer designs (TNN7 Sec. IV-B)
DESIGNS = [["c1"], ["c1", "c2"], ["c1", "c2", "c3"]]

# aclk periods per gclk (gamma) period: tnn_hw/syn_tcl/chip.sdc:1-2 sets aclk 10000 and gclk 150000.
# The fsm_weight_update report's gclk period is 2.64 us (379 kHz), not 15 aclk periods; its energy per
# gclk event, P / f_report, is still self-consistent.
GAMMA_CYCLES = 15


def clog2(x):
    # $clog2 in SystemVerilog
    return (x - 1).bit_length()


def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ################ Architecture ################
    ##############################################
    # aclk = 100 kHz, "based on real-time operation requirement" (TNN7 Sec. IV)
    architecture.add_attributes(technology=7, frequency=0.1, interface="tnn7")

    # One set of modules per layer; the three designs differ only in which layers exist, chosen by the
    # event sweep below. Instances are [cols, p, q] per synapse, [cols, q] per neuron, [cols] per column.
    # Per-column counts follow the p x q column RTL (column65x2.sv), with
    #   L = clog2(p) (pac.sv:26), M = max(L + 1, clog2(theta) + 1) accumulator bits (pac.sv:30).
    # Function tags: synapse (fsm_synapse.sv), stdp (stdp.sv), neuron (neuron_body.sv), wta (wta.sv),
    # io (pulse/edge conversion and the gclk reset pulse, column65x2.sv:62-81,107-118).
    for layer, shape in LAYERS.items():
        cols, p, q = shape["cols"], shape["p"], shape["q"]
        L = clog2(p)
        M = max(L + 1, clog2(THRESHOLD[layer]) + 1)
        # pac adder tree: stage i = 0..L-2 has 2^L / 2^(i+2) adders of i+1 bits (pac.sv:56-66),
        # 2^L - L - 1 full-adder bits per neuron (inputs zero-padded to 2^L, pac.sv:45)
        tree_bits = 2**L - L - 1

        # fmt: off
        # per synapse: fsm_synapse = weight update + readout + 3-bit weight register + 1-bit gclk delay register (fsm_synapse.sv:11-15)
        architecture.add_module(name=f"{layer}_syn_readout", instance=[cols, p, q], tag=["tnn", layer, "synapse"], query={"class": "fsm_output"})
        architecture.add_module(name=f"{layer}_syn_weight_update", instance=[cols, p, q], tag=["tnn", layer, "synapse"], query={"class": "fsm_weight_update"})
        architecture.add_module(name=f"{layer}_syn_reg", instance=[cols, p, q], tag=["tnn", layer, "synapse"], query={"class": "tnn_reg", "width": 4})

        # per synapse: stdp = case gen + stabilize func + incdec (stdp.sv:42-54; one stdp per synapse, column65x2.sv:120-145)
        architecture.add_module(name=f"{layer}_stdp_case_gen", instance=[cols, p, q], tag=["tnn", layer, "stdp"], query={"class": "stdp_case_gen"})
        architecture.add_module(name=f"{layer}_stdp_flogic", instance=[cols, p, q], tag=["tnn", layer, "stdp"], query={"class": "flogic"})
        architecture.add_module(name=f"{layer}_stdp_incdec", instance=[cols, p, q], tag=["tnn", layer, "stdp"], query={"class": "incdec"})

        # per neuron: neuron_body = pac + spike gen (neuron_body.sv:30-40)
        # tnn_adder is a 4-bit adder (prabhu:src/adder/adder.v:1-5); width = TNN7 add_inv full-adder bits (adder.sv:26-35)
        architecture.add_module(name=f"{layer}_pac_tree", instance=[cols, q], tag=["tnn", layer, "neuron"], query={"class": "tnn_adder", "width": tree_bits})
        architecture.add_module(name=f"{layer}_pac_acc", instance=[cols, q], tag=["tnn", layer, "neuron"], query={"class": "tnn_adder", "width": M})  # pac.sv:76-80
        architecture.add_module(name=f"{layer}_pac_acc_reg", instance=[cols, q], tag=["tnn", layer, "neuron"], query={"class": "tnn_reg", "width": M})  # pac.sv:82-87
        # threshold load mux to the constant -theta: one 2-input gate per bit plus the (out | grst) OR2 (pac.sv:91);
        # no mux or OR2 report, tnn_and_gate is the stand-in
        architecture.add_module(name=f"{layer}_pac_mux", instance=[cols, q, M + 1], tag=["tnn", layer, "neuron"], query={"class": "tnn_and_gate"})
        architecture.add_module(name=f"{layer}_spike_gen", instance=[cols, q], tag=["tnn", layer, "neuron"], query={"class": "fsm_simple"})
        # 3-bit spike gen state register (fsm_simple.sv:15); the fsm_simple report has only the macro (prabhu:src/fsm_simple/rtl_list.tcl)
        architecture.add_module(name=f"{layer}_spike_gen_reg", instance=[cols, q], tag=["tnn", layer, "neuron"], query={"class": "tnn_reg", "width": 3})

        # per column: wta = first-spike OR + pulse2edge, q less_equal, q-1 tie-break AND2 (wta.sv:29-60)
        architecture.add_module(name=f"{layer}_wta_pulse2edge", instance=[cols], tag=["tnn", layer, "wta"], query={"class": "pulse2edge_96", "width": 1})
        architecture.add_module(name=f"{layer}_wta_less_equal", instance=[cols, q], tag=["tnn", layer, "wta"], query={"class": "less_equal"})
        architecture.add_module(name=f"{layer}_wta_and", instance=[cols, q - 1], tag=["tnn", layer, "wta"], query={"class": "tnn_and_gate"})
        # OR2: q-1 for |ec_spikes (wta.sv:29) + q-2 shared prefix ORs for the tie-break (wta.sv:57); no OR2 report, tnn_and_gate is the stand-in
        architecture.add_module(name=f"{layer}_wta_or", instance=[cols, 2 * q - 3], tag=["tnn", layer, "wta"], query={"class": "tnn_and_gate"})

        # pulse2edge_96 is 96 pulse2edge lanes (prabhu:src/pulse2edge_96/pulse2edge_96.sv:9); width 1 = one lane
        architecture.add_module(name=f"{layer}_in_pulse2edge", instance=[cols, p], tag=["tnn", layer, "io"], query={"class": "pulse2edge_96", "width": 1})  # column65x2.sv:70-81
        architecture.add_module(name=f"{layer}_out_pulse2edge", instance=[cols, q], tag=["tnn", layer, "io"], query={"class": "pulse2edge_96", "width": 1})  # column65x2.sv:107-118
        # gclk -> 2-cycle grst pulse: edge2pulse, 1-bit stretch register, OR2 (column65x2.sv:62-68)
        architecture.add_module(name=f"{layer}_edge2pulse", instance=[cols], tag=["tnn", layer, "io"], query={"class": "edge2pulse"})
        architecture.add_module(name=f"{layer}_grst_reg", instance=[cols], tag=["tnn", layer, "io"], query={"class": "tnn_reg", "width": 1})
        architecture.add_module(name=f"{layer}_grst_or", instance=[cols], tag=["tnn", layer, "io"], query={"class": "tnn_and_gate"})
        # fmt: on

    ##############################################
    ###############    Event    ##################
    ##############################################
    # Event sweep: each candidate subevent list of "tnn" is one design, and AGraph keeps only the modules
    # reachable from that design's event graph (agraph.py save_solutions), so absent layers have no
    # modules at all instead of zero-instance modules. Runs land in runs/mnist/arch_<i>/event_<i>/config_0,
    # with i = 0, 1, 2 for the 2-, 3-, 4-layer designs.
    # One "tnn" event is one input: one gamma cycle per layer (Smith: "each layer of the TNN is a single pipeline stage").
    event.add_event(name="tnn", subevent=DESIGNS, performance=PERFORMANCE)
    for layer in LAYERS:
        event.add_event(
            name=layer,
            subevent=[f"{layer}_aclk", f"{layer}_gclk"],
            performance=PERFORMANCE,
        )
        # one unit-clock (aclk) cycle; every register in the column is clocked by aclk
        event.add_event(
            name=f"{layer}_aclk",
            subevent=[
                f"{layer}_syn_readout",
                f"{layer}_syn_reg",
                f"{layer}_stdp_case_gen",
                f"{layer}_stdp_flogic",
                f"{layer}_stdp_incdec",
                f"{layer}_pac_tree",
                f"{layer}_pac_acc",
                f"{layer}_pac_acc_reg",
                f"{layer}_pac_mux",
                f"{layer}_spike_gen",
                f"{layer}_spike_gen_reg",
                f"{layer}_wta_pulse2edge",
                f"{layer}_wta_less_equal",
                f"{layer}_wta_and",
                f"{layer}_wta_or",
                f"{layer}_in_pulse2edge",
                f"{layer}_out_pulse2edge",
                f"{layer}_edge2pulse",
                f"{layer}_grst_reg",
                f"{layer}_grst_or",
            ],
            performance=PERFORMANCE,
        )
        # one gamma-clock (gclk) weight update: tinc/tdec fire only on the first aclk after gclk rises
        # (gclk & ~nxt_gclk, fsm_weight_update.sv:9-10)
        event.add_event(
            name=f"{layer}_gclk",
            subevent=[f"{layer}_syn_weight_update"],
            performance=PERFORMANCE,
        )

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
    # computation time is "derived from the critical path delay and the gamma period" (TNN7 Sec. IV)
    mnist = workload.add_configuration(name="mnist")
    mnist.add_parameter(parameter_name="gamma_cycles", parameter_value=GAMMA_CYCLES)

    return agraph.generate()
