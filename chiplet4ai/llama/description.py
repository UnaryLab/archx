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
    attributes = architecture.add_attributes(technology=[7], frequency=1000, interface='chiplet_cmos')

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
    dram_mapping_events = ['dram_input_read', 'dram_weight_read', 'dram_output_read', 'dram_output_write']
    sram_mapping_events = ['sram_input_write_mapping', 'sram_weight_write_mapping', 'sram_output_read_mapping', 'sram_output_write_mapping']
    node_mapping_events = ['array_input_mapping', 'array_weight_mapping', 'array_compute_mapping']
    weight_events = ['wfifo', 'weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg']

    layer_arr_events_pf = [f'{event}_arr' for event in layer_events_pf]
    layer_arr_events_dc = [f'{event}_arr' for event in layer_events_dc]

    # workload events
    event.add_event(name='llama_3_8b', subevent=['llama', 'llama_array'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='llama_3_70b', subevent=['llama', 'llama_array'], performance='chiplet4ai/llama/llama_model.py')

    # 
    event.add_event(name='llama_array', subevent=['llama_pf_array', 'llama_dc_array', 'lm_head_pf_arr', 'lm_head_dc_arr'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='llama_pf_array', subevent=layer_arr_events_pf.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='llama_dc_array', subevent=layer_arr_events_dc.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='llama', subevent=model_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    # model phase events
    event.add_event(name='prefill', subevent=['layer_pf', 'lm_head_pf'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='decode', subevent=['layer_dc', 'lm_head_dc'], performance='chiplet4ai/llama/llama_model.py')

    # transformer layer events
    event.add_event(name='layer_pf', subevent=layer_events_pf.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='layer_dc', subevent=layer_events_dc.copy(), performance='chiplet4ai/llama/llama_model.py')

    # prefill GEMM events
    event.add_event(name='proj_q_pf', subevent=['proj_q_pf_arr', 'proj_q_pf_sram', 'proj_q_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_q_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_q_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_q_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='proj_k_pf', subevent=['proj_k_pf_arr', 'proj_k_pf_sram', 'proj_k_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_k_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_k_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_k_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='proj_v_pf', subevent=['proj_v_pf_arr', 'proj_v_pf_sram', 'proj_v_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_v_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_v_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_v_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='qkt_pf', subevent=['qkt_pf_arr', 'qkt_pf_sram', 'qkt_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='qkt_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='qkt_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='qkt_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='av_pf', subevent=['av_pf_arr', 'av_pf_sram', 'av_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='av_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='av_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='av_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='a_proj_pf', subevent=['a_proj_pf_arr', 'a_proj_pf_sram', 'a_proj_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='a_proj_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='a_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='a_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='up_proj_pf', subevent=['up_proj_pf_arr', 'up_proj_pf_sram', 'up_proj_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='up_proj_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='up_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='up_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='gate_proj_pf', subevent=['gate_proj_pf_arr', 'gate_proj_pf_sram', 'gate_proj_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='gate_proj_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='gate_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='gate_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='down_proj_pf', subevent=['down_proj_pf_arr', 'down_proj_pf_sram', 'down_proj_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='down_proj_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='down_proj_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='down_proj_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='lm_head_pf', subevent=['lm_head_pf_arr', 'lm_head_pf_sram', 'lm_head_pf_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='lm_head_pf_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='lm_head_pf_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='lm_head_pf_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    # decode GEMM events
    event.add_event(name='proj_q_dc', subevent=['proj_q_dc_arr', 'proj_q_dc_sram', 'proj_q_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_q_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_q_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_q_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='proj_k_dc', subevent=['proj_k_dc_arr', 'proj_k_dc_sram', 'proj_k_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_k_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_k_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_k_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='proj_v_dc', subevent=['proj_v_dc_arr', 'proj_v_dc_sram', 'proj_v_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_v_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_v_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='proj_v_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='qkt_dc', subevent=['qkt_dc_arr', 'qkt_dc_sram', 'qkt_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='qkt_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='qkt_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='qkt_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='av_dc', subevent=['av_dc_arr', 'av_dc_sram', 'av_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='av_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='av_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='av_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='a_proj_dc', subevent=['a_proj_dc_arr', 'a_proj_dc_sram', 'a_proj_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='a_proj_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='a_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='a_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='up_proj_dc', subevent=['up_proj_dc_arr', 'up_proj_dc_sram', 'up_proj_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='up_proj_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='up_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='up_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='gate_proj_dc', subevent=['gate_proj_dc_arr', 'gate_proj_dc_sram', 'gate_proj_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='gate_proj_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='gate_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='gate_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='down_proj_dc', subevent=['down_proj_dc_arr', 'down_proj_dc_sram', 'down_proj_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='down_proj_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='down_proj_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='down_proj_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    event.add_event(name='lm_head_dc', subevent=['lm_head_dc_arr', 'lm_head_dc_sram', 'lm_head_dc_dram'], performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='lm_head_dc_arr', subevent=node_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='lm_head_dc_sram', subevent=sram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')
    event.add_event(name='lm_head_dc_dram', subevent=dram_mapping_events.copy(), performance='chiplet4ai/llama/llama_model.py')

    # node mapping events
    event.add_event(name='array_input_mapping', subevent=['array_input', 'sram_input_read'], performance='chiplet4ai/common/performance/node.py')
    event.add_event(name='array_weight_mapping', subevent=['array_weight', 'sram_weight_read'], performance='chiplet4ai/common/performance/node.py')
    event.add_event(name='array_compute_mapping', subevent=['array_compute', 'sram_output_read', 'sram_output_write'], performance='chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_input_write_mapping', subevent=['sram_input_write'], performance='chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_weight_write_mapping', subevent=['sram_weight_write'], performance='chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_output_read_mapping', subevent=['sram_output_read'], performance='chiplet4ai/common/performance/node.py')
    event.add_event(name='sram_output_write_mapping', subevent=['sram_output_write'], performance='chiplet4ai/common/performance/node.py')

    # array events
    event.add_event(name='array_input', subevent=['ififo'], performance='chiplet4ai/common/performance/array.py')
    event.add_event(name='array_weight', subevent=weight_events.copy(), performance='chiplet4ai/common/performance/array.py')
    event.add_event(name='array_compute', subevent=['pe', 'ofifo', 'output_adder'], performance='chiplet4ai/common/performance/array.py')

    # dram events
    event.add_event(name='dram_input_read', subevent=['dram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='dram_weight_read', subevent=['dram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='dram_output_read', subevent=['dram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='dram_output_write', subevent=['dram'], performance='chiplet4ai/common/performance/memory.py')

    # scratchpad events
    event.add_event(name='sram_input_read', subevent=['isram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_input_write', subevent=['isram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_weight_read', subevent=['wsram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_weight_write', subevent=['wsram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_output_read', subevent=['osram'], performance='chiplet4ai/common/performance/memory.py')
    event.add_event(name='sram_output_write', subevent=['osram'], performance='chiplet4ai/common/performance/memory.py')
    

    ##############################################
    ###############    Metric    #################
    ##############################################
    metric.add_metric(name='area',           unit='mm^2',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    ##############################################
    ###############   Workload   #################
    ##############################################
    llama_3_8b_config = workload.add_configuration(name='llama_3_8b')
    llama_3_8b_batch_size = llama_3_8b_config.add_parameter(parameter_name='batch_size', parameter_value=vector_sizes, sweep=True)
    llama_3_8b_config.add_parameter(parameter_name='dim', parameter_value=4096)
    llama_3_8b_config.add_parameter(parameter_name='heads',  parameter_value=32)
    llama_3_8b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_8b_config.add_parameter(parameter_name='hidden_dim', parameter_value=14336)
    llama_3_8b_config.add_parameter(parameter_name='layers', parameter_value=32)
    llama_3_8b_config.add_parameter(parameter_name='max_seq_len', parameter_value=8192)
    llama_3_8b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama_3_8b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

    llama_3_70b_config = workload.add_configuration(name='llama_3_70b')
    llama_3_70b_batch_size = llama_3_70b_config.add_parameter(parameter_name='batch_size', parameter_value=vector_sizes, sweep=True)
    llama_3_70b_config.add_parameter(parameter_name='dim', parameter_value=8192)
    llama_3_70b_config.add_parameter(parameter_name='heads',  parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_70b_config.add_parameter(parameter_name='hidden_dim', parameter_value=28672)
    llama_3_70b_config.add_parameter(parameter_name='layers', parameter_value=80)
    llama_3_70b_config.add_parameter(parameter_name='max_seq_len', parameter_value=8192)
    llama_3_70b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)

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
