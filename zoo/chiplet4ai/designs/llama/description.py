from archx.programming.graph.agraph import AGraph
from copy import deepcopy
import math

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
    base_array_size = 512
    array_shape = [32, 512] # from 32x32 to 512x512, step by powers of 2

    array_range = range(int(math.log2(array_shape[0])), int(math.log2(array_shape[1])) + 1)

    array_shapes = [[2**i, 2**j] for i in array_range for j in array_range]
    vector_shapes = [[2**i] for i in array_range]
    vector_sizes = [2**i for i in array_range]

    base_sram_size = 10 * 2**23 # 10MB
    sram_total_sizes = [i * 2**23 for i in range(1, 11)]  # 1 MiB to 10 MiB, in bits
    fixed_array_shape = [base_array_size, base_array_size]
    fixed_sram_bank = 2 * base_array_size
    sram_banks = [x[0] * 2 for x in vector_shapes]
    sram_depths = sorted({
        sram_size // (sram_bank * bitwidth)
        for sram_size in sram_total_sizes
        for sram_bank in sram_banks
        if sram_size % (sram_bank * bitwidth) == 0
    })

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
    model_events = ['prefill', 'decode']
    layer_events_pf = ['proj_q_pf', 'proj_k_pf', 'proj_v_pf', 'qkt_pf', 'av_pf', 'a_proj_pf', 'up_proj_pf', 'gate_proj_pf', 'down_proj_pf']
    layer_events_dc = ['proj_q_dc', 'proj_k_dc', 'proj_v_dc', 'qkt_dc', 'av_dc', 'a_proj_dc', 'up_proj_dc', 'gate_proj_dc', 'down_proj_dc']
    # Stall events are named for the PHASE they measure (fill / steady / tail), not for an
    # operand: mapping.py's stall model resolves the SRAMs' real-time demand per phase and
    # spreads the whole-span DRAM floor back over the phases, so the quantities are
    # phase-shaped and the old per-operand names actively misattributed them (under k_outer
    # the steady phase's bytes are ~96% output spill, and in the reference config the tail
    # phase is ~67% weight bytes).
    #
    # The '_dram'/'_sram' PREFIXES are NOT load-bearing and play no part in bucketing.
    # query_cycle_breakdown (results/query/utils.py) buckets on endswith '_arr' / '_sram' /
    # '_dram' over the PARENT composite events and adds each matched node's whole subtree;
    # these stall events are counted because they sit inside the '*_dram' / '*_sram'
    # wrappers' subtrees listed below, never because of their own prefix. The prefixes are
    # human-readable provenance only, and renaming them would not move a single cycle
    # between buckets.
    dram_mapping_events = [
        'dram_input_read', 'dram_weight_read', 'dram_output_read', 'dram_output_write',
        'dram_fill_stall', 'dram_steady_stall', 'dram_tail_stall'
    ]
    sram_mapping_events = [
        'sram_input_write_mapping', 'sram_weight_write_mapping', 'sram_output_read_mapping', 'sram_output_write_mapping',
        'sram_fill_stall', 'sram_steady_stall', 'sram_tail_stall'
    ]
    node_mapping_events = ['array_input_mapping', 'array_weight_mapping', 'array_compute_mapping']
    weight_events = ['wfifo', 'weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg']

    layer_arr_events_pf = [f'{event}_arr' for event in layer_events_pf]
    layer_arr_events_dc = [f'{event}_arr' for event in layer_events_dc]

    # workload events
    event.add_event(name='llama_3_1_8b', subevent=['llama', 'llama_array'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='llama_3_1_70b', subevent=['llama', 'llama_array'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='llama_3_1_405b', subevent=['llama', 'llama_array'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='deepseek_v4', subevent=['llama', 'llama_array'], performance='zoo/chiplet4ai/designs/llama/model.py')

    # 
    event.add_event(name='llama_array', subevent=['llama_pf_array', 'llama_dc_array', 'lm_head_pf_arr', 'lm_head_dc_arr'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='llama_pf_array', subevent=layer_arr_events_pf.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='llama_dc_array', subevent=layer_arr_events_dc.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='llama', subevent=model_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    # model phase events
    event.add_event(name='prefill', subevent=['layer_pf', 'lm_head_pf'], performance='zoo/chiplet4ai/designs/llama/model.py')
    # decode carries both layer variants; model.py's decode() routes per run on the
    # workload (deepseek -> layer_dc_moe, dense llama -> layer_dc, the other at count 0)
    event.add_event(name='decode', subevent=['layer_dc', 'layer_dc_moe', 'lm_head_dc'], performance='zoo/chiplet4ai/designs/llama/model.py')

    # transformer layer events
    event.add_event(name='layer_pf', subevent=layer_events_pf.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='layer_dc', subevent=layer_events_dc.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='layer_dc_moe', subevent=layer_events_dc.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    # prefill GEMM events
    event.add_event(name='proj_q_pf', subevent=['proj_q_pf_arr', 'proj_q_pf_sram', 'proj_q_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_q_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_q_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_q_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='proj_k_pf', subevent=['proj_k_pf_arr', 'proj_k_pf_sram', 'proj_k_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_k_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_k_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_k_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='proj_v_pf', subevent=['proj_v_pf_arr', 'proj_v_pf_sram', 'proj_v_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_v_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_v_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_v_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='qkt_pf', subevent=['qkt_pf_arr', 'qkt_pf_sram', 'qkt_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='qkt_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='qkt_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='qkt_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='av_pf', subevent=['av_pf_arr', 'av_pf_sram', 'av_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='av_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='av_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='av_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='a_proj_pf', subevent=['a_proj_pf_arr', 'a_proj_pf_sram', 'a_proj_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='a_proj_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='a_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='a_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='up_proj_pf', subevent=['up_proj_pf_arr', 'up_proj_pf_sram', 'up_proj_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='up_proj_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='up_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='up_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='gate_proj_pf', subevent=['gate_proj_pf_arr', 'gate_proj_pf_sram', 'gate_proj_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='gate_proj_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='gate_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='gate_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='down_proj_pf', subevent=['down_proj_pf_arr', 'down_proj_pf_sram', 'down_proj_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='down_proj_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='down_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='down_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='lm_head_pf', subevent=['lm_head_pf_arr', 'lm_head_pf_sram', 'lm_head_pf_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='lm_head_pf_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='lm_head_pf_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='lm_head_pf_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    # decode GEMM events
    event.add_event(name='proj_q_dc', subevent=['proj_q_dc_arr', 'proj_q_dc_sram', 'proj_q_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_q_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_q_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_q_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='proj_k_dc', subevent=['proj_k_dc_arr', 'proj_k_dc_sram', 'proj_k_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_k_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_k_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_k_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='proj_v_dc', subevent=['proj_v_dc_arr', 'proj_v_dc_sram', 'proj_v_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_v_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_v_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='proj_v_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='qkt_dc', subevent=['qkt_dc_arr', 'qkt_dc_sram', 'qkt_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='qkt_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='qkt_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='qkt_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='av_dc', subevent=['av_dc_arr', 'av_dc_sram', 'av_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='av_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='av_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='av_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='a_proj_dc', subevent=['a_proj_dc_arr', 'a_proj_dc_sram', 'a_proj_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='a_proj_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='a_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='a_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='up_proj_dc', subevent=['up_proj_dc_arr', 'up_proj_dc_sram', 'up_proj_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='up_proj_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='up_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='up_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='gate_proj_dc', subevent=['gate_proj_dc_arr', 'gate_proj_dc_sram', 'gate_proj_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='gate_proj_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='gate_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='gate_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='down_proj_dc', subevent=['down_proj_dc_arr', 'down_proj_dc_sram', 'down_proj_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='down_proj_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='down_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='down_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    event.add_event(name='lm_head_dc', subevent=['lm_head_dc_arr', 'lm_head_dc_sram', 'lm_head_dc_dram'], performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='lm_head_dc_arr', subevent=node_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='lm_head_dc_sram', subevent=sram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')
    event.add_event(name='lm_head_dc_dram', subevent=dram_mapping_events.copy(), performance='zoo/chiplet4ai/designs/llama/model.py')

    # node mapping events
    event.add_event(name='array_input_mapping', subevent=['array_input', 'sram_input_read'], performance='zoo/chiplet4ai/common/performance/node.py')
    event.add_event(name='array_weight_mapping', subevent=['array_weight', 'sram_weight_read'], performance='zoo/chiplet4ai/common/performance/node.py')
    event.add_event(name='array_compute_mapping', subevent=['array_compute', 'sram_output_read', 'sram_output_write'], performance='zoo/chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_input_write_mapping', subevent=['sram_input_write'], performance='zoo/chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_weight_write_mapping', subevent=['sram_weight_write'], performance='zoo/chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_output_read_mapping', subevent=['sram_output_read'], performance='zoo/chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_output_write_mapping', subevent=['sram_output_write'], performance='zoo/chiplet4ai/common/performance/node.py')

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

    # analytical stall events (per-phase; see the naming note at the event configuration)
    for stall_event in [
        'dram_fill_stall', 'dram_steady_stall', 'dram_tail_stall',
        'sram_fill_stall', 'sram_steady_stall', 'sram_tail_stall'
    ]:
        event.add_event(name=stall_event, subevent=['cycle_reference'], performance='zoo/chiplet4ai/common/performance/mapping.py')

    event.add_event(name='cycle_reference', subevent=['pe'], performance='zoo/chiplet4ai/common/performance/mapping.py')
    

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
    llama_3_8b_batch_size = llama_3_8b_config.add_parameter(parameter_name='batch_size', parameter_value=vector_sizes, sweep=True)
    llama_3_8b_config.add_parameter(parameter_name='dim', parameter_value=4096)
    llama_3_8b_config.add_parameter(parameter_name='heads',  parameter_value=32)
    llama_3_8b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_8b_config.add_parameter(parameter_name='hidden_dim', parameter_value=14336)
    llama_3_8b_config.add_parameter(parameter_name='layers', parameter_value=32)
    llama_3_8b_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072], sweep=True)
    llama_3_8b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama_3_8b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    llama_3_70b_config = workload.add_configuration(name='llama_3_1_70b')
    llama_3_70b_batch_size = llama_3_70b_config.add_parameter(parameter_name='batch_size', parameter_value=vector_sizes, sweep=True)
    llama_3_70b_config.add_parameter(parameter_name='dim', parameter_value=8192)
    llama_3_70b_config.add_parameter(parameter_name='heads',  parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_70b_config.add_parameter(parameter_name='hidden_dim', parameter_value=28672)
    llama_3_70b_config.add_parameter(parameter_name='layers', parameter_value=80)
    llama_3_70b_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072], sweep=True)
    llama_3_70b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    llama_3_405b_config = workload.add_configuration(name='llama_3_1_405b')
    llama_3_405b_batch_size = llama_3_405b_config.add_parameter(parameter_name='batch_size', parameter_value=vector_sizes, sweep=True)
    llama_3_405b_config.add_parameter(parameter_name='dim', parameter_value=16384)
    llama_3_405b_config.add_parameter(parameter_name='heads',  parameter_value=128)
    llama_3_405b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_405b_config.add_parameter(parameter_name='hidden_dim', parameter_value=53248)
    llama_3_405b_config.add_parameter(parameter_name='layers', parameter_value=126)
    llama_3_405b_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072], sweep=True)
    llama_3_405b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama_3_405b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    deepseek_v4_config = workload.add_configuration(name='deepseek_v4')
    deepseek_v4_batch_size = deepseek_v4_config.add_parameter(parameter_name='batch_size', parameter_value=vector_sizes, sweep=True)
    deepseek_v4_config.add_parameter(parameter_name='dim', parameter_value=7168)
    deepseek_v4_config.add_parameter(parameter_name='heads',  parameter_value=128)
    deepseek_v4_config.add_parameter(parameter_name='kv_heads', parameter_value=1)
    deepseek_v4_config.add_parameter(parameter_name='head_dim', parameter_value=512)  # explicit: dim // heads = 56 is wrong for MLA
    deepseek_v4_config.add_parameter(parameter_name='qk_rope_head_dim', parameter_value=64)
    deepseek_v4_config.add_parameter(parameter_name='q_lora_rank', parameter_value=1536)
    deepseek_v4_config.add_parameter(parameter_name='o_lora_rank', parameter_value=1024)
    deepseek_v4_config.add_parameter(parameter_name='o_groups', parameter_value=16)
    deepseek_v4_config.add_parameter(parameter_name='hidden_dim', parameter_value=3072)  # per-expert moe_intermediate_size; keeps up/gate/down GEMMs per-expert-shaped
    deepseek_v4_config.add_parameter(parameter_name='layers', parameter_value=61)
    deepseek_v4_config.add_parameter(parameter_name='max_seq_len', parameter_value=[4096, 131072, 1048576], sweep=True)  # model max is 1048576; clipped to match the llama workloads' decode-step count
    deepseek_v4_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    deepseek_v4_config.add_parameter(parameter_name='vocab_size', parameter_value=129280)
    # MoE routing/experts: router gate GEMM is dim x n_routed_experts;
    # activated experts per token = experts_per_tok + n_shared_experts
    deepseek_v4_config.add_parameter(parameter_name='n_routed_experts', parameter_value=384)
    deepseek_v4_config.add_parameter(parameter_name='n_shared_experts', parameter_value=1)
    deepseek_v4_config.add_parameter(parameter_name='experts_per_tok', parameter_value=6)

    ##############################################
    ###########   Constraints   ##################
    ##############################################

    # array constraint
    agraph.direct_constraint([
        pe['instance'],
        weight_control_regs['weight_path_en_reg']['instance'],
        weight_control_regs['weight_en_reg']['instance'],
        weight_data_regs['weight_path_reg']['instance'],
        weight_data_regs['weight_reg']['instance']
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

    agraph.direct_constraint([
        srams['wsram']['query']['bank'],
        srams['osram']['query']['bank']
    ])

    agraph.direct_constraint([
        srams['wsram']['query']['depth'],
        srams['osram']['query']['depth']
    ])

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
    
    agraph.conditional_constraint(a = pe['instance'],
                                  b = srams['isram']['query']['bank'],
                                  c = srams['isram']['query']['depth'],
                                  condition = lambda pe_inst, bank, depth: (
                                      (pe_inst == fixed_array_shape and bank == fixed_sram_bank and (bank * depth * bitwidth) in sram_total_sizes)
                                      or
                                      (pe_inst != fixed_array_shape and bank * depth * bitwidth == base_sram_size)
                                  ))
    
    agraph.conditional_constraint(a = pe['instance'],
                                  b = srams['wsram']['query']['bank'],
                                  c = srams['wsram']['query']['depth'],
                                  condition = lambda pe_inst, bank, depth: (
                                      (pe_inst == fixed_array_shape and bank == fixed_sram_bank and (bank * depth * bitwidth) in sram_total_sizes)
                                      or
                                      (pe_inst != fixed_array_shape and bank * depth * bitwidth == base_sram_size)
                                  ))
    
    agraph.conditional_constraint(a = pe['instance'],
                                  b = srams['osram']['query']['bank'],
                                  c = srams['osram']['query']['depth'],
                                  condition = lambda pe_inst, bank, depth: (
                                      (pe_inst == fixed_array_shape and bank == fixed_sram_bank and (bank * depth * bitwidth) in sram_total_sizes)
                                      or
                                      (pe_inst != fixed_array_shape and bank * depth * bitwidth == base_sram_size)
                                  ))
    
    agraph.generate()
    return agraph
