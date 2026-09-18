from archx.programming.graph.agraph import AGraph
from copy import deepcopy
import math
from chiplet4ai.common.performance.utils import nominal_sram_bits
from chiplet4ai.results.query.utils import CORE_ARRAY_SIZES, FIG_BATCH_SIZE

# FIG_2'S DESIGN POINT, PER MODEL: (array shape, batch). fig_2 plots required DRAM
# bandwidth against SRAM CAPACITY, holding the array and batch fixed so the curve
# carries capacity alone.
#
# THIS IS fig_4's avg_band POINT -- the SMALLEST array within `SELECTION_MARGIN` of the
# `average_bandwidth` maximum over fig_6's grid at 1000 MHz and batch `FIG_BATCH_SIZE`
# (both in results/query/utils.py), not the strict argmax -- so fig_2's capacity curve
# and fig_4's bandwidth bars describe the same machine. It is deliberately the most
# memory-hungry configuration rather than the fastest: a capacity sweep is most informative where capacity is under the most
# pressure, which is also why three of the four points are 32-row arrays (the
# shallowest reduction depth in the sweep, so the worst weight reuse and the most DRAM
# traffic).
#
# KEEP IN STEP WITH fig_4_query. These values are transcribed from
# results/csv/dram_bandwidth_metrics_avg_band.csv, which fig_4_query derives by argmax;
# fig_2_query re-derives the same point from the same CSV rather than hardcoding it.
# If a re-run moves the argmax, this list must move with it; fig_2_query imports it and
# raises when a model's fig_4 point carries fewer than two SRAM sizes. Unlike
# fig_4's point, this one CANNOT be a free choice at query time: the off-base
# capacities exist only where this file puts them.
sram_sweep_point = {
    'llama_3_1_8b':   ([32, 512], 512),
    'llama_3_1_70b':  ([32, 512], 512),
    'llama_3_1_405b': ([32, 512], 512),
    'deepseek_v4':    ([128, 512], 512),
}


def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ############# Configuration Spec #############
    ##############################################
    bitwidth = 16
    base_array_size = 128
    array_shape = [32, 4096] # from 32x32 to 4096x4096, step by powers of 2

    array_range = range(int(math.log2(array_shape[0])), int(math.log2(array_shape[1])) + 1)

    # THE FULL CROSS PRODUCT OF SHAPES, not just the diagonal. `pe.instance[0]` is the
    # weight-stationary reduction depth (rows) and `pe.instance[1]` the output width
    # (columns); mapping.py has always read the two separately -- what pinned them equal
    # was the single SRAM bank tie in the constraints below, not the performance model.
    # fig_6 walks the off-diagonal shapes (32x64, 512x256, ...); fig_1 through fig_5 all
    # filter back to the diagonal, so their design points are unchanged. The 1024-4096
    # sides are fig_6's alone: every other query filters to CORE_ARRAY_SIZES
    # (results/query/utils.py).
    array_sizes = [2**i for i in array_range]
    array_shapes = [[array_m, array_n] for array_m in array_sizes for array_n in array_sizes]
    vector_shapes = [[2**i] for i in array_range]
    vector_sizes = [2**i for i in array_range]
    # batch has its own list: the 1024-4096 array sides above are for fig_6 alone
    batch_sizes = [32, 64, 128, 256, 512]

    sram_total_sizes = [i * 2**21 for i in range(5, 41)]  # 1 MiB to 10 MiB, in bits
    sram_banks = [x[0] * 2 for x in vector_shapes]

    sweep_array_shapes = [list(shape) for shape in
                          sorted({tuple(shape) for shape, _ in sram_sweep_point.values()})]
    # BOTH SIDES OF EVERY SWEPT SHAPE. isram is banked to the array's rows and wsram/osram
    # to its columns, so a rectangular sweep point needs depths for two different bank
    # counts -- a 32x512 array sweeps a 64-bank isram against a 1024-bank wsram, both held
    # to the same total size. Taking only one side would leave half the sweep with no
    # legal depth and silently drop it.
    sweep_banks = sorted({2 * side for shape in sweep_array_shapes for side in shape})

    # FIG_4 NO LONGER NEEDS A DESIGN POINT HERE, and that is deliberate.
    #
    # fig_4 reports one machine per model at that model's MAX context across the frequency
    # (technology-node) axis. Every shape, every batch, both frequencies already exist at
    # max context because fig_6's grid generates them, so fig_4's point is a CELL OF THAT
    # GRID rather than a slice this file has to carve out. Whatever shape fig_6's ranking
    # crowns is therefore already simulated, and moving fig_4's point is a one-line edit in
    # fig_4_query.py with no change here and no re-simulation.
    #
    # fig_2 is the one that still needs `sram_sweep_point` above: its axis is SRAM
    # capacity, which is off the base size and so outside fig_6's grid entirely.

    # ONLY THE REACHABLE DEPTHS. Every SRAM's bank count is pinned to 2x the array side it
    # serves by the constraints below -- so it is always one of `sram_banks`, on a
    # rectangular array as much as on a square one -- and only the arrays in
    # `sweep_array_shapes` sweep capacity; every other shape sits at its nominal size
    # (nominal_sram_bits: 10 MiB up to 512x512, doubled until one array-sized weight tile
    # fits wsram above that). So the depths that can ever survive are the sweep's, at each
    # swept bank, plus each shape's nominal-size depth on both of its sides. Taking the cross
    # product of every size with every bank instead would list ~3x as many depths, all of
    # them filtered out later, and each conditional_constraint pays for the whole list
    # when it enumerates its allowed tuples.
    sram_depths = sorted(
        {
            sram_size // (sweep_bank * bitwidth)
            for sram_size in sram_total_sizes
            for sweep_bank in sweep_banks
            if sram_size % (sweep_bank * bitwidth) == 0
        } | {
            nominal_sram_bits(*shape, bitwidth) // (2 * side * bitwidth)
            for shape in array_shapes
            for side in shape
        }
    )

    dram_bits = 8 * 2**30 # 8GB
    dram_bandwidth = 256  # GB/s

    ##############################################
    ################ Architecture ################
    ##############################################
    attributes = architecture.add_attributes(technology=7, frequency=[1000, 2000], interface='chiplet_cmos')

    # DRAM
    dram = architecture.add_module(name='dram', instance=[1], tag=['offchip', 'memory'], query={'class': 'dram', 'interface': 'cacti7', 'size': dram_bits, 'bandwidth': dram_bandwidth, 'technology': 45})

    # SRAM
    srams = architecture.add_module(name=['isram', 'wsram', 'osram'], instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': sram_banks, 'width': bitwidth, 'depth': sram_depths, 'technology': 45})

    # FIFO
    fifos = architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=vector_shapes, tag=['onchip', 'fifo', 'array'], query={'class': 'fifo_sync_tsmc', 'width': bitwidth, 'depth': vector_sizes})

    pe = architecture.add_module(name='pe', instance=array_shapes, tag=['onchip', 'array', 'compute'], query={'class': 'pe_tsmc'})
    output_adder = architecture.add_module(name='output_adder', instance=vector_shapes, tag=['onchip', 'output_adder', 'array'], query={'class': 'fp16_adder_tsmc'})
    weight_data_regs = architecture.add_module(name=['weight_reg', 'weight_path_reg'], instance=array_shapes, tag=['onchip', 'register', 'array'], query={'class': 'register_tsmc', 'width': bitwidth})
    weight_control_regs = architecture.add_module(name=['weight_en_reg', 'weight_path_en_reg'], instance=array_shapes, tag=['onchip', 'register', 'control', 'array'], query={'class': 'register_tsmc', 'width': 1})

    ##############################################
    ###############    Event    ##################
    ##############################################

    # event configuration
    layer_pf_events = ['proj_q_pf', 'proj_k_pf', 'proj_v_pf', 'qkt_pf', 'av_pf', 'a_proj_pf', 'up_proj_pf', 'gate_proj_pf', 'down_proj_pf']
    layer_dc_events = ['proj_q_dc', 'proj_k_dc', 'proj_v_dc', 'qkt_dc', 'av_dc', 'a_proj_dc', 'up_proj_dc', 'gate_proj_dc', 'down_proj_dc']
    # DeepSeek-V4's CSA and HCA layers: the same prefill and decode GEMMs with the attention
    # pair swapped for the layer type's own compressed-attention GEMMs
    # a CSA layer also runs the DSA indexer that picks which compressed entries its
    # attention pair then reads; HCA has no indexer and attends over all of its own
    layer_pf_compressed_events = {
        kind: [layer.replace('_pf', f'_{kind}_pf') if layer in ('qkt_pf', 'av_pf') else layer
               for layer in layer_pf_events] + ([f'index_{kind}_pf'] if kind == 'csa' else [])
        for kind in ('csa', 'hca')
    }
    layer_dc_compressed_events = {
        kind: [layer.replace('_dc', f'_{kind}_dc') if layer in ('qkt_dc', 'av_dc') else layer
               for layer in layer_dc_events] + ([f'index_{kind}_dc'] if kind == 'csa' else [])
        for kind in ('csa', 'hca')
    }
    compressed_attention_events = [f'{op}_{kind}_{phase}' for phase in ('pf', 'dc')
                                   for kind in ('csa', 'hca') for op in ('qkt', 'av')]
    compressed_attention_events += [f'index_csa_{phase}' for phase in ('pf', 'dc')]
    gemm_events = layer_pf_events + ['lm_head_pf'] + layer_dc_events + ['lm_head_dc'] + compressed_attention_events

    layer_pf_arr_subevents = [layer + '_arr' for layer in layer_pf_events]
    layer_dc_arr_subevents = [layer + '_arr' for layer in layer_dc_events]

    # mapping.py drives array.py's and memory.py's events directly: it scales each count
    # to the useful work and carries the true cycle_count in the edge factors
    dram_mapping_events = ['dram_input_read', 'dram_weight_read', 'dram_output_read', 'dram_output_write']
    sram_mapping_events = ['sram_input_write', 'sram_input_read', 'sram_weight_write', 'sram_weight_read',
                           'sram_output_write', 'sram_output_read']
    array_mapping_events = ['array_input', 'array_weight', 'array_compute']
    weight_events = ['wfifo', 'weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg']

    model = 'zoo/chiplet4ai/designs/llama/model.py'

    # workload events: 'llama' charges array + SRAM + DRAM per GEMM, 'llama_array'
    # is the compute-only view of the same GEMMs (model.py takes their max at the
    # root and zero-weights the array view's energy)
    event.add_event(name='llama_3_1_8b', subevent=['llama', 'llama_array'], performance=model)
    event.add_event(name='llama_3_1_70b', subevent=['llama', 'llama_array'], performance=model)
    event.add_event(name='llama_3_1_405b', subevent=['llama', 'llama_array'], performance=model)
    event.add_event(name='deepseek_v4', subevent=['llama', 'llama_array'], performance=model)

    # model events
    event.add_event(name='llama', subevent=['prefill', 'decode'], performance=model)
    event.add_event(name='llama_array', subevent=['llama_pf_array', 'llama_pf_csa_array', 'llama_pf_hca_array', 'llama_dc_array', 'llama_dc_csa_array', 'llama_dc_hca_array', 'lm_head_pf_arr', 'lm_head_dc_arr'], performance=model)
    event.add_event(name='llama_pf_array', subevent=layer_pf_arr_subevents.copy(), performance=model)
    for kind, events in layer_pf_compressed_events.items():
        event.add_event(name=f'llama_pf_{kind}_array', subevent=[layer + '_arr' for layer in events], performance=model)
    event.add_event(name='llama_dc_array', subevent=layer_dc_arr_subevents.copy(), performance=model)
    for kind, events in layer_dc_compressed_events.items():
        event.add_event(name=f'llama_dc_{kind}_array', subevent=[layer + '_arr' for layer in events], performance=model)

    # model phase events
    event.add_event(name='prefill', subevent=['layer_pf', 'layer_pf_csa', 'layer_pf_hca', 'lm_head_pf'], performance=model)
    event.add_event(name='decode', subevent=['layer_dc', 'layer_dc_moe', 'layer_dc_csa', 'layer_dc_hca', 'lm_head_dc'], performance=model)
    event.add_event(name='layer_pf', subevent=layer_pf_events.copy(), performance=model)
    for kind, events in layer_pf_compressed_events.items():
        event.add_event(name=f'layer_pf_{kind}', subevent=events.copy(), performance=model)
    event.add_event(name='layer_dc', subevent=layer_dc_events.copy(), performance=model)
    event.add_event(name='layer_dc_moe', subevent=layer_dc_events.copy(), performance=model)
    for kind, events in layer_dc_compressed_events.items():
        event.add_event(name=f'layer_dc_{kind}', subevent=events.copy(), performance=model)

    # GEMM events: cycle count is the max of array compute, SRAM ports, and the
    # DRAM channel (the '_arr'/'_sram'/'_dram' edges are parallel in model.py)
    for layer in gemm_events:
        event.add_event(name=layer, subevent=[layer + '_arr', layer + '_sram', layer + '_dram'], performance=model)
        event.add_event(name=layer + '_arr', subevent=array_mapping_events.copy(), performance=model)
        event.add_event(name=layer + '_sram', subevent=sram_mapping_events.copy(), performance=model)
        event.add_event(name=layer + '_dram', subevent=dram_mapping_events.copy(), performance=model)

    # array events
    event.add_event(name='array_input', subevent=['ififo'], performance='zoo/chiplet4ai/common/performance/array.py')
    event.add_event(name='array_weight', subevent=weight_events.copy(), performance='zoo/chiplet4ai/common/performance/array.py')
    event.add_event(name='array_compute', subevent=['pe', 'ofifo', 'output_adder'], performance='zoo/chiplet4ai/common/performance/array.py')

    # dram events
    event.add_event(name='dram_input_read', subevent=['dram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='dram_weight_read', subevent=['dram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='dram_output_read', subevent=['dram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='dram_output_write', subevent=['dram'], performance='zoo/chiplet4ai/common/performance/memory.py')

    # scratchpad events
    event.add_event(name='sram_input_read', subevent=['isram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_input_write', subevent=['isram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_weight_read', subevent=['wsram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_weight_write', subevent=['wsram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_output_read', subevent=['osram'], performance='zoo/chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_output_write', subevent=['osram'], performance='zoo/chiplet4ai/common/performance/memory.py')

    ##############################################
    ###############    Metric    #################
    ##############################################
    metric.add_metric(name='area',           unit='mm^2',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')
    metric.add_metric(name='bandwidth',      unit='GiB/s',  aggregation='specified')

    ##############################################
    ###############   Workload   #################
    ##############################################
    llama_3_8b_config = workload.add_configuration(name='llama_3_1_8b')
    llama_3_8b_batch_size = llama_3_8b_config.add_parameter(parameter_name='batch_size', parameter_value=batch_sizes, sweep=True)
    llama_3_8b_config.add_parameter(parameter_name='dim', parameter_value=4096)
    llama_3_8b_config.add_parameter(parameter_name='heads',  parameter_value=32)
    llama_3_8b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_8b_config.add_parameter(parameter_name='hidden_dim', parameter_value=14336)
    llama_3_8b_config.add_parameter(parameter_name='layers', parameter_value=32)
    llama_3_8b_seq_len = llama_3_8b_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072], sweep=True)
    llama_3_8b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=2048)
    llama_3_8b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    llama_3_70b_config = workload.add_configuration(name='llama_3_1_70b')
    llama_3_70b_batch_size = llama_3_70b_config.add_parameter(parameter_name='batch_size', parameter_value=batch_sizes, sweep=True)
    llama_3_70b_config.add_parameter(parameter_name='dim', parameter_value=8192)
    llama_3_70b_config.add_parameter(parameter_name='heads',  parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_70b_config.add_parameter(parameter_name='hidden_dim', parameter_value=28672)
    llama_3_70b_config.add_parameter(parameter_name='layers', parameter_value=80)
    llama_3_70b_seq_len = llama_3_70b_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072], sweep=True)
    llama_3_70b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=2048)
    llama_3_70b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    llama_3_405b_config = workload.add_configuration(name='llama_3_1_405b')
    llama_3_405b_batch_size = llama_3_405b_config.add_parameter(parameter_name='batch_size', parameter_value=batch_sizes, sweep=True)
    llama_3_405b_config.add_parameter(parameter_name='dim', parameter_value=16384)
    llama_3_405b_config.add_parameter(parameter_name='heads',  parameter_value=128)
    llama_3_405b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_405b_config.add_parameter(parameter_name='hidden_dim', parameter_value=53248)
    llama_3_405b_config.add_parameter(parameter_name='layers', parameter_value=126)
    llama_3_405b_seq_len = llama_3_405b_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072], sweep=True)
    llama_3_405b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=2048)
    llama_3_405b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    deepseek_v4_config = workload.add_configuration(name='deepseek_v4')
    deepseek_v4_batch_size = deepseek_v4_config.add_parameter(parameter_name='batch_size', parameter_value=batch_sizes, sweep=True)
    deepseek_v4_config.add_parameter(parameter_name='dim', parameter_value=7168)
    deepseek_v4_config.add_parameter(parameter_name='heads',  parameter_value=128)
    deepseek_v4_config.add_parameter(parameter_name='kv_heads', parameter_value=1)
    deepseek_v4_config.add_parameter(parameter_name='head_dim', parameter_value=512)
    # compressed attention, from the shipped config's compress_ratios: layers 0-1 are HCA,
    # then CSA and HCA alternate starting with CSA (model.py derives the counts)
    deepseek_v4_config.add_parameter(parameter_name='csa_compress_ratio', parameter_value=4)
    deepseek_v4_config.add_parameter(parameter_name='hca_compress_ratio', parameter_value=128)
    deepseek_v4_config.add_parameter(parameter_name='hca_lead_layers', parameter_value=2)
    deepseek_v4_config.add_parameter(parameter_name='sliding_window', parameter_value=128)
    deepseek_v4_config.add_parameter(parameter_name='index_head_dim', parameter_value=128)  # CSA indexer keys, cached per compressed entry
    deepseek_v4_config.add_parameter(parameter_name='index_n_heads', parameter_value=64)    # DSA indexer query heads
    deepseek_v4_config.add_parameter(parameter_name='index_topk', parameter_value=1024)     # compressed entries the indexer selects per query
    deepseek_v4_config.add_parameter(parameter_name='hidden_dim', parameter_value=3072)  # per-expert moe_intermediate_size; keeps up/gate/down GEMMs per-expert-shaped
    deepseek_v4_config.add_parameter(parameter_name='layers', parameter_value=61)
    deepseek_v4_seq_len = deepseek_v4_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072, 1048576], sweep=True)  # model max is 1048576; clipped to match the llama workloads' decode-step count
    deepseek_v4_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=2048)
    deepseek_v4_config.add_parameter(parameter_name='vocab_size', parameter_value=129280)
    deepseek_v4_config.add_parameter(parameter_name='n_routed_experts', parameter_value=384)
    deepseek_v4_config.add_parameter(parameter_name='n_shared_experts', parameter_value=1)
    deepseek_v4_config.add_parameter(parameter_name='experts_per_tok', parameter_value=6)
    # 1, not 2: no speculative decoding. The paper (arXiv:2606.19348 sec 4.2.1, "the
    # multi-token prediction depth is set to 1") sets MTP as a TRAINING objective and
    # describes no inference-time speculative decoding. Modeling that faithfully would
    # need an acceptance rate and a non-free draft verification pass, neither of which
    # this model has.
    deepseek_v4_config.add_parameter(parameter_name='tokens_per_step', parameter_value=1)

    ##############################################
    ###########   Constraints   ##################
    ##############################################

    # array constraint
    #
    # direct_constraint ties its parameters BY INDEX, so it can only relate lists that
    # enumerate the same thing in the same order. The registers are per-PE and share
    # `array_shapes` with the array, so they stay here. The FIFOs do NOT: they are
    # one-dimensional (`vector_shapes`), one per array row or column, and now that
    # `array_shapes` is the 8x8 cross product rather than the 8-entry diagonal, index
    # equality would silently pin the array to its first five shapes. They are pinned by
    # value instead, in the ififo/wfifo/ofifo conditional constraints below -- which is
    # what actually says which SIDE of the array each FIFO serves.
    agraph.direct_constraint([
        pe['instance'],
        weight_control_regs['weight_path_en_reg']['instance'],
        weight_control_regs['weight_en_reg']['instance'],
        weight_data_regs['weight_path_reg']['instance'],
        weight_data_regs['weight_reg']['instance'],
    ])
    
    agraph.direct_constraint([
        fifos['wfifo']['query']['depth'],
        fifos['ofifo']['query']['depth'],
        fifos['ififo']['instance'],
    ])

    agraph.direct_constraint([
        fifos['ififo']['query']['depth'],
        fifos['wfifo']['instance'],
        fifos['ofifo']['instance'],
        output_adder['instance']
    ])

    # ISRAM IS BANKED TO A DIFFERENT SIDE OF THE ARRAY THAN WSRAM AND OSRAM, so the three
    # banks only coincide when the array is square. A bank is one width-bit transfer per
    # cycle (see mapping.py's `_buffer_elements`), and the streams it has to keep up with
    # are set by the array's two dimensions independently: isram feeds the Kt = array_m
    # reduction rows, while wsram loads and osram drains the Nt = array_n columns. The
    # conditional constraints below pin each to its own FIFO, and this used to be a
    # three-way tie -- which, with ififo on the rows and wfifo on the columns, was an
    # unstated `array_m == array_n` and the sole reason the design space was diagonal.
    #
    # wsram and osram serve the SAME side, so they still sweep together.
    agraph.direct_constraint([
        srams['wsram']['query']['bank'],
        srams['osram']['query']['bank'],
    ])

    agraph.direct_constraint([
        srams['wsram']['query']['depth'],
        srams['osram']['query']['depth'],
    ])

    # ONE CAPACITY, THREE SHAPES. Decoupling the banks must not turn into decoupling the
    # capacities: a rectangular array would then quietly carry more (or less) total SRAM
    # than the square one it is being compared against, and fig_6 would be reading a
    # memory-size effect as an aspect-ratio effect. So isram is held to the same bit count
    # as wsram/osram and absorbs its different bank count in its DEPTH -- a 512x32 array
    # gets a 1024-bank, shallow isram and a 64-bank, deep wsram, both 10 MiB.
    #
    # This also keeps every downstream query's `isram_size == wsram_size == osram_size`
    # assumption true, and lets the constraints further down name isram alone and still
    # speak for all three.
    agraph.conditional_constraint(ibank = srams['isram']['query']['bank'],
                                  idepth = srams['isram']['query']['depth'],
                                  wbank = srams['wsram']['query']['bank'],
                                  wdepth = srams['wsram']['query']['depth'],
                                  condition = lambda ibank, idepth, wbank, wdepth: (
                                      ibank * idepth == wbank * wdepth
                                  ))

    # sram_size = 8 * 2**23
    # dram_bandwidth = 128 # GB/s

    # conditional constraint
    agraph.conditional_constraint(a = fifos['ififo']['instance'],
                                  b = pe['instance'],
                                  condition = lambda a, b: a[0] == b[0])
    
    agraph.conditional_constraint(a = fifos['wfifo']['instance'],
                                  b = pe['instance'],
                                  condition = lambda a, b: a[0] == b[1])
    
    agraph.conditional_constraint(a = fifos['ofifo']['instance'],
                                  b = pe['instance'],
                                  condition = lambda a, b: a[0] == b[1])
    
    agraph.conditional_constraint(a = fifos['ififo']['instance'],
                                  b = srams['isram']['query']['bank'],
                                  condition = lambda a, b: a[0] * 2 == b)
    
    agraph.conditional_constraint(a = fifos['wfifo']['instance'],
                                  b = srams['wsram']['query']['bank'],
                                  condition = lambda a, b: a[0] * 2 == b)
    
    agraph.conditional_constraint(a = fifos['ofifo']['instance'],
                                  b = srams['osram']['query']['bank'],
                                  condition = lambda a, b: a[0] * 2 == b)
    
    agraph.conditional_constraint(pe_inst = pe['instance'],
                                  bank = srams['isram']['query']['bank'],
                                  depth = srams['isram']['query']['depth'],
                                  condition = lambda pe_inst, bank, depth: (
                                      (pe_inst in sweep_array_shapes and (bank * depth * bitwidth) in sram_total_sizes)
                                      or
                                      (pe_inst not in sweep_array_shapes and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))
    
    agraph.conditional_constraint(pe_inst = pe['instance'],
                                  bank = srams['wsram']['query']['bank'],
                                  depth = srams['wsram']['query']['depth'],
                                  condition = lambda pe_inst, bank, depth: (
                                      (pe_inst in sweep_array_shapes and (bank * depth * bitwidth) in sram_total_sizes)
                                      or
                                      (pe_inst not in sweep_array_shapes and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))
    
    agraph.conditional_constraint(pe_inst = pe['instance'],
                                  bank = srams['osram']['query']['bank'],
                                  depth = srams['osram']['query']['depth'],
                                  condition = lambda pe_inst, bank, depth: (
                                      (pe_inst in sweep_array_shapes and (bank * depth * bitwidth) in sram_total_sizes)
                                      or
                                      (pe_inst not in sweep_array_shapes and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    agraph.conditional_constraint(freq = attributes['frequency'],
                                  pe_inst = pe['instance'],
                                  bank = srams['isram']['query']['bank'],
                                  depth = srams['isram']['query']['depth'],
                                  condition = lambda freq, pe_inst, bank, depth: (freq == 1000 or (freq == 2000 and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    agraph.conditional_constraint(freq = attributes['frequency'],
                                  pe_inst = pe['instance'],
                                  bank = srams['wsram']['query']['bank'],
                                  depth = srams['wsram']['query']['depth'],
                                  condition = lambda freq, pe_inst, bank, depth: (freq == 1000 or (freq == 2000 and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    agraph.conditional_constraint(freq = attributes['frequency'],
                                  pe_inst = pe['instance'],
                                  bank = srams['osram']['query']['bank'],
                                  depth = srams['osram']['query']['depth'],
                                  condition = lambda freq, pe_inst, bank, depth: (freq == 1000 or (freq == 2000 and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    # ------------------------------------------------------------------------------
    # SCOPE: generate only what a figure plots.
    #
    # Every configuration below is simulated, so an axis crossed with a point no query
    # reads is pure cost. The per-model SLICE RULE at the end of this function generates
    # exactly these four slices and nothing else (all at the nominal SRAM size except D;
    # max context is 131072 for the Llamas, 1048576 for DeepSeek):
    #
    #   slice  figures              shapes                batch      MHz         context
    #   A      fig_6, fig_7         all 64                FIG_BATCH  1000        max
    #   B      fig_4 (and fig_8,    all 25 <= 512         FIG_BATCH  2000        max
    #          fig_2's point)       (1000 MHz is in A)
    #   C      fig_1, fig_3, fig_5  5 squares <= 512      all 5      1000        4096, 131072,
    #                                                                            and max
    #   D      fig_2                sram_sweep_point      its batch  1000        max
    #                               shape, 35 off-base SRAM sizes
    #
    # Per model that is 64 + 25 + (25 x contexts - 5, C's max-context FIG_BATCH_SIZE squares
    # being in A) + 35 runs: 169 for a Llama, 194 for DeepSeek, 701 in all.
    # B keeps 2000 MHz on all 25 shapes, not only fig_4's picks, so a pick can never land
    # outside the sweep. To change a figure's scope, edit its slice in the slice rule; the
    # older rules above it are weaker than it (it implies each of them), so they only need
    # loosening if a new scope goes beyond what they allow.
    #
    # NOTE, for anyone extending fig_6's figure: `llama_array` is the COMPUTE-ONLY view
    # and is exactly frequency-invariant (frequency enters only through the DRAM lane's
    # bytes-per-cycle, mapping.py). The 2000 MHz runs therefore differ from the 1000 MHz
    # ones in `llama` cycles and in runtime, never in `llama_array` cycles. fig_6.py plots
    # the 1000 MHz slice only; the frequency axis lives in fig_6_query's CSV.
    # ------------------------------------------------------------------------------

    # WHAT BOUNDS 2000 MHz is now the per-model rule at the bottom of this section, not a
    # shape whitelist here: fig_6 needs the whole grid at 2000 MHz, so pinning the 2 GHz
    # architectures to the two swept shapes would exclude it. The one rule that still
    # holds globally -- 2000 MHz implies the base SRAM -- is already stated by the three
    # freq/SRAM-size constraints above, which keeps 2 GHz out of fig_2's capacity sweep.

    # BATCH IS FREE ON THE BASE SRAM, at both frequencies. fig_1, fig_3 and fig_5 plot one
    # series per batch at 1000 MHz, and fig_6's CSV now carries the batch axis at 2000 MHz
    # as well, so nothing on the base SRAM pins it. What still pins batch is the SRAM
    # CAPACITY sweep: that is fig_2's axis, fig_2 reads a single batch, and crossing 35
    # off-base capacities with five batches would be five times the runs for one plotted
    # curve. So the off-base capacities exist only at this model's sram_sweep_point.
    batch_parameters = {
        'llama_3_1_8b': llama_3_8b_batch_size,
        'llama_3_1_70b': llama_3_70b_batch_size,
        'llama_3_1_405b': llama_3_405b_batch_size,
        'deepseek_v4': deepseek_v4_batch_size,
    }
    for model_name, batch_parameter in batch_parameters.items():
        sweep_shape, sweep_batch = sram_sweep_point[model_name]
        agraph.conditional_constraint(batch = batch_parameter['parameter'],
                                      freq = attributes['frequency'],
                                      pe_inst = pe['instance'],
                                      bank = srams['isram']['query']['bank'],
                                      depth = srams['isram']['query']['depth'],
                                      condition = lambda batch, freq, pe_inst, bank, depth, \
                                                         shape=sweep_shape, sbatch=sweep_batch: (
                                          # the SRAM sweep: this model's fig_2 point only.
                                          # The whole SHAPE is pinned, not just its width:
                                          # the points are rectangular now, and two models
                                          # can share a width while differing in the other
                                          # side.
                                          (pe_inst == shape and batch == sbatch and freq == 1000)
                                          if bank * depth * bitwidth != nominal_sram_bits(*pe_inst, bitwidth)
                                          # base SRAM: batch free at both frequencies
                                          else True
                                      ))

    # SHORT CONTEXT ONLY AT THE BASE SRAM. The SRAM sweep is fig_2's axis and fig_2 reads
    # one sequence length per model, so the other lengths are pinned to the nominal size
    # rather than multiplied across all 36 capacities.
    #
    # NAME ONE SRAM, NOT THREE. conditional_constraint builds its allowed-tuple table by
    # walking the full Python cross product of every variable it names, so each extra
    # variable multiplies the enumeration by that variable's cardinality. isram, wsram and
    # osram are already direct_constraint-tied bank-to-bank and depth-to-depth above, so
    # naming all three costs (banks x depths)^2 = ~3.4e5 times more work to express
    # exactly the same condition -- 1.8 billion tuples across these four constraints
    # versus 17.5 thousand. One SRAM stands for all three.
    agraph.conditional_constraint(a = llama_3_8b_seq_len['parameter'],
                                  pe_inst = pe['instance'],
                                  bank = srams['isram']['query']['bank'],
                                  depth = srams['isram']['query']['depth'],
                                  condition = lambda a, pe_inst, bank, depth: (
                                      a == 131072 or (a == 4096 and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    agraph.conditional_constraint(a = llama_3_70b_seq_len['parameter'],
                                  pe_inst = pe['instance'],
                                  bank = srams['isram']['query']['bank'],
                                  depth = srams['isram']['query']['depth'],
                                  condition = lambda a, pe_inst, bank, depth: (
                                      a == 131072 or (a == 4096 and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    agraph.conditional_constraint(a = llama_3_405b_seq_len['parameter'],
                                  pe_inst = pe['instance'],
                                  bank = srams['isram']['query']['bank'],
                                  depth = srams['isram']['query']['depth'],
                                  condition = lambda a, pe_inst, bank, depth: (
                                      a == 131072 or (a == 4096 and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))

    agraph.conditional_constraint(a = deepseek_v4_seq_len['parameter'],
                                  pe_inst = pe['instance'],
                                  bank = srams['isram']['query']['bank'],
                                  depth = srams['isram']['query']['depth'],
                                  condition = lambda a, pe_inst, bank, depth: (
                                      a == 1048576 or ((a == 4096 or a == 131072) and bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth))
                                  ))
    
    # TWO RULES, ONE LOOP, BOTH PER MODEL, and both saying the same thing about context:
    # everything fig_6 added -- the off-diagonal shapes and the 2000 MHz half -- lives at
    # that model's MAXIMUM context only. 131072 for the Llamas, 1048576 for DeepSeek, the
    # same per-model maximum fig_1's and fig_3's 'mixed' slice uses, so fig_6's diagonal
    # lines up with those panels exactly.
    #
    # (1) OFF-DIAGONAL ARRAYS. Letting the shorter contexts onto the 20 rectangular shapes
    #     would multiply them by lengths no query reads.
    # (2) 2000 MHz. Its two consumers, fig_4 and fig_6, both report max context, so the
    #     short contexts stay at 1000 MHz. This is also what keeps 2 GHz out of the
    #     capacity sweep from growing: it is already base-SRAM-only by the freq/size rules.
    #
    # The slice rule below narrows both further.
    max_seq_len_parameters = [
        (llama_3_8b_seq_len, 131072),
        (llama_3_70b_seq_len, 131072),
        (llama_3_405b_seq_len, 131072),
        (deepseek_v4_seq_len, 1048576),
    ]
    for seq_len_parameter, model_max_seq_len in max_seq_len_parameters:
        agraph.conditional_constraint(a = seq_len_parameter['parameter'],
                                      pe_inst = pe['instance'],
                                      condition = lambda a, pe_inst, m=model_max_seq_len: (
                                          pe_inst[0] == pe_inst[1] or a == m
                                      ))

        agraph.conditional_constraint(freq = attributes['frequency'],
                                      a = seq_len_parameter['parameter'],
                                      condition = lambda freq, a, m=model_max_seq_len: (
                                          freq == 1000 or a == m
                                      ))

    # THE SLICE RULE: slices A-D of the SCOPE table above, per model, and nothing else. It
    # names one SRAM for the reason the short-context rule gives.
    seq_len_parameters = {
        'llama_3_1_8b': llama_3_8b_seq_len,
        'llama_3_1_70b': llama_3_70b_seq_len,
        'llama_3_1_405b': llama_3_405b_seq_len,
        'deepseek_v4': deepseek_v4_seq_len,
    }
    for (model_name, batch_parameter), (_, model_max_seq_len) in zip(
            batch_parameters.items(), max_seq_len_parameters):
        sweep_shape, sweep_batch = sram_sweep_point[model_name]
        agraph.conditional_constraint(batch = batch_parameter['parameter'],
                                      seq = seq_len_parameters[model_name]['parameter'],
                                      freq = attributes['frequency'],
                                      pe_inst = pe['instance'],
                                      bank = srams['isram']['query']['bank'],
                                      depth = srams['isram']['query']['depth'],
                                      condition = lambda batch, seq, freq, pe_inst, bank, depth, \
                                                         m=model_max_seq_len, shape=sweep_shape, \
                                                         sbatch=sweep_batch: (
                                          (bank * depth * bitwidth == nominal_sram_bits(*pe_inst, bitwidth)
                                           and (
                                               # A: fig_6/fig_7
                                               (batch == FIG_BATCH_SIZE and freq == 1000 and seq == m)
                                               # B: fig_4 at 2000 MHz
                                               or (batch == FIG_BATCH_SIZE and freq == 2000 and seq == m
                                                   and pe_inst[0] in CORE_ARRAY_SIZES
                                                   and pe_inst[1] in CORE_ARRAY_SIZES)
                                               # C: fig_1/fig_3/fig_5
                                               or (freq == 1000 and pe_inst[0] == pe_inst[1]
                                                   and pe_inst[0] in CORE_ARRAY_SIZES)))
                                          # D: fig_2's SRAM sweep
                                          or (bank * depth * bitwidth != nominal_sram_bits(*pe_inst, bitwidth)
                                              and pe_inst == shape and batch == sbatch
                                              and freq == 1000 and seq == m)
                                      ))

    agraph.generate()
    return agraph