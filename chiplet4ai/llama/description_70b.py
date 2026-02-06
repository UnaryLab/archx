from archx.programming.graph.agraph import AGraph
from archx.programming.sweeping.sweeping_functions import condition_sweep, list_sweep

def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload
    ##############################################
    ################ Architecture ################
    ##############################################
    attributes = architecture.add_attributes(technology=[7, 16], frequency=100, interface='csv_cmos')
    bitwidth = 16

    # array and memory shapes
    act_sram_bank = 64
    weight_sram_bank = 64

    array_shapes = [[32, 32], [64, 64], [64, 128], [64, 256], [64, 512], [128, 128], [128, 256], [128, 512], [256, 256], [256, 512], [512, 512]]
    fifo_shapes = [[32], [64], [128], [256], [512]]
    chiplet_shape = [16, 32, 64]
    act_sram_width = [256, 512, 1024, 2048, 4096]
    weight_sram_width = [256, 512, 1024, 2048, 4096]
    mb = 1024 * 1024 * 8
    memory_size = condition_sweep(value=mb, funct=lambda x: x+mb, condition=lambda x: x < 11 * mb)
  
    act_sram_depth = []
    weight_sram_depth = []
    for m in memory_size:
        for w_act, w_weight in zip(act_sram_width, weight_sram_width):
            act_sram_depth.append(m // (w_act * act_sram_bank))
            weight_sram_depth.append(m // (w_weight * weight_sram_bank))

    

    # chiplet_shape = condition_sweep(value=16, funct=lambda x: x*2, condition=lambda x: x <= 256)

    # act_sram_width = list_sweep(values=array_shapes, funct=lambda x: x[0] * bitwidth / act_sram_bank)
    # weight_sram_width = list_sweep(values=array_shapes, funct=lambda x: x[0] * bitwidth / weight_sram_bank)

    # act_sram_depth = list_sweep(values=act_sram_width, funct=lambda x: memory_size / (x * act_sram_bank))
    # weight_sram_depth = list_sweep(values=weight_sram_width, funct=lambda x: memory_size / (x * weight_sram_bank))

    # # DRAM
    dram = architecture.add_module(name='dram', instance=[1], tag=['offchip'], query={'class': 'dram', 'interface': 'cacti7', 'size': 8589934592, 'bandwidth': 40, 'technology': 45})

    # # SRAM
    act_sram = architecture.add_module(name=['isram', 'osram'], instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': act_sram_bank, 'width': act_sram_width, 'depth': act_sram_depth, 'technology': 45})
    wsram = architecture.add_module(name='wsram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': weight_sram_bank, 'width': weight_sram_width, 'depth': weight_sram_depth, 'technology': 45})

    # # FIFO
    fifos = architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=fifo_shapes, tag=['onchip', 'fifo', 'array'], query={'class': ['fifo_sync_tsmc7', 'fifo_sync_tsmc16'], 'width': bitwidth, 'depth': chiplet_shape})

    # Output Accumulator
    output_adder = architecture.add_module(name='output_adder', instance=fifo_shapes, tag=['onchip', 'output_adder', 'array'], query={'class': ['fp16_adder_tsmc7', 'fp16_adder_tsmc16']})

    # operations
    multiplier = architecture.add_module(name='multiplier', instance=array_shapes, tag=['onchip', 'pe', 'mac', 'array'], query={'class': ['fp16_mult_tsmc7', 'fp16_mult_tsmc16']})
    adder = architecture.add_module(name='adder', instance=array_shapes, tag=['onchip', 'pe', 'mac', 'array'], query={'class': ['fp16_adder_tsmc7', 'fp16_adder_tsmc16']})

    # data registers
    control_regs = architecture.add_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=array_shapes, tag=['tag', 'pe', 'control', 'array'], query={'class': ['register_tsmc7', 'register_tsmc16'], 'width': 1})
    data_regs = architecture.add_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=array_shapes, tag=['tag', 'pe', 'data', 'array'], query={'class': ['register_tsmc7', 'register_tsmc16'], 'width': bitwidth})

    muxes = architecture.add_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=array_shapes, tag=['tag', 'pe', 'array'], query={'class': ['register_tsmc7', 'register_tsmc16'], 'width': bitwidth})

    ##############################################
    ###############    Event    ##################
    ##############################################
    array_events = ['input_reads', 'weight_reads', 'output_writes', 'output_reads', 'array_mapping', 'weight_mapping', 'input_offchip_writes', 'weight_offchip_writes', 'output_offchip_reads',
                    'output_offchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads']
    array_mapping_events = ['ififo', 'ofifo', 'output_adder', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg',
                            'acc_en_reg', 'sum_en_reg', 'act_reg', 'sum_reg', 'act_mux', 'weight_mux', 'add_mux', 'sum_mux']
    weight_mapping_events = ['wfifo', 'weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg']

    #event.add_event(name='llama_3_8b', subevent=['gemm'], performance='chiplet4ai/llama/performance/llama_model.py')
    event.add_event(name='llama_3_70b', subevent=['gemm'], performance='chiplet4ai/llama/performance/llama_model.py')
    event.add_event(name='gemm', subevent=['projection', 'attention', 'ffn', 'output'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='projection', subevent=['proj_q', 'proj_k', 'proj_v', 'proj_a'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='attention', subevent=['qkt', 'av'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='ffn', subevent=['proj_up', 'proj_down', 'proj_gate'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='output', subevent=['output_prefill', 'output_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_q', subevent=['proj_q_prefill', 'proj_q_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_k', subevent=['proj_k_prefill', 'proj_k_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_v', subevent=['proj_v_prefill', 'proj_v_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_a', subevent=['proj_a_prefill', 'proj_a_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='qkt', subevent=['qkt_prefill', 'qkt_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='av', subevent=['av_prefill', 'av_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_up', subevent=['proj_up_prefill', 'proj_up_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_down', subevent=['proj_down_prefill', 'proj_down_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_gate', subevent=['proj_gate_prefill', 'proj_gate_decode'], performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='output_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='output_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_q_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_q_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_k_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_k_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_v_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_v_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_a_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_a_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='qkt_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='qkt_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='av_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='av_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_up_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_up_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_down_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_down_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_gate_prefill', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='proj_gate_decode', subevent=array_events, performance='chiplet4ai/llama/performance/llama_architecture.py')
    event.add_event(name='input_reads', subevent=['isram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='weight_reads', subevent=['wsram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='output_writes', subevent=['osram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='output_reads', subevent=['osram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='array_mapping', subevent=array_mapping_events, performance='chiplet4ai/common/performance/chiplet_mapping.py')
    event.add_event(name='weight_mapping', subevent=weight_mapping_events, performance='chiplet4ai/common/performance/chiplet_mapping.py')
    event.add_event(name='input_offchip_writes', subevent=['isram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='weight_offchip_writes', subevent=['wsram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='output_offchip_reads', subevent=['osram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='output_offchip_writes', subevent=['osram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='dram_input_reads', subevent=['dram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='dram_weight_reads', subevent=['dram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='dram_output_writes', subevent=['dram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    event.add_event(name='dram_output_reads', subevent=['dram'], performance='chiplet4ai/common/performance/chiplet_memory.py')
    

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
    batch_size = [32, 64, 128]

    # llama_3_8b_config = workload.add_configuration(name='llama_3_8b')
    # llama_3_8b_batch_size = llama_3_8b_config.add_parameter(parameter_name='batch_size', parameter_value=batch_size, sweep=True)
    # llama_3_8b_config.add_parameter(parameter_name='dim', parameter_value=4096)
    # llama_3_8b_config.add_parameter(parameter_name='heads',  parameter_value=32)
    # llama_3_8b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    # llama_3_8b_config.add_parameter(parameter_name='hidden_dim', parameter_value=14336)
    # llama_3_8b_config.add_parameter(parameter_name='layers', parameter_value=32)
    # llama_3_8b_config.add_parameter(parameter_name='activation_bitwidth', parameter_value=bitwidth)
    # llama_3_8b_config.add_parameter(parameter_name='weight_bitwidth', parameter_value=bitwidth)
    # llama_3_8b_config.add_parameter(parameter_name='max_sequence_length', parameter_value=4096)
    # llama_3_8b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    # llama_3_8b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)
    # llama_3_8b_config.add_parameter(parameter_name='banks', parameter_value=act_sram_bank)
    # memory_size_wkld = llama_3_8b_config.add_parameter(parameter_name='memory_size_mb', parameter_value=memory_size, sweep=True)

    llama_3_70b_config = workload.add_configuration(name='llama_3_70b')
    llama_3_70b_batch_size = llama_3_70b_config.add_parameter(parameter_name='batch_size', parameter_value=batch_size, sweep=True)
    llama_3_70b_config.add_parameter(parameter_name='dim', parameter_value=8192)
    llama_3_70b_config.add_parameter(parameter_name='heads',  parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama_3_70b_config.add_parameter(parameter_name='hidden_dim', parameter_value=28672)
    llama_3_70b_config.add_parameter(parameter_name='layers', parameter_value=80)
    llama_3_70b_config.add_parameter(parameter_name='activation_bitwidth', parameter_value=bitwidth)
    llama_3_70b_config.add_parameter(parameter_name='weight_bitwidth', parameter_value=bitwidth)
    llama_3_70b_config.add_parameter(parameter_name='max_sequence_length', parameter_value=4096)
    llama_3_70b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama_3_70b_config.add_parameter(parameter_name='vocab_size', parameter_value=128256)
    llama_3_70b_config.add_parameter(parameter_name='banks', parameter_value=act_sram_bank)
    memory_size_wkld = llama_3_70b_config.add_parameter(parameter_name='memory_size_mb', parameter_value=memory_size, sweep=True)

    ##############################################
    ###########   Constraints   ##################
    ##############################################

    # direct constraint with no condition
    agraph.direct_constraint(parameters = [multiplier['instance'],
                                           adder['instance'],
                                           control_regs['act_en_reg']['instance'],
                                           control_regs['mult_en_reg']['instance'],
                                           control_regs['acc_en_reg']['instance'],
                                           control_regs['weight_path_en_reg']['instance'],
                                           control_regs['weight_en_reg']['instance'],
                                           control_regs['sum_en_reg']['instance'],
                                           data_regs['act_reg']['instance'],
                                           data_regs['weight_path_reg']['instance'],
                                           data_regs['sum_reg']['instance'],
                                           data_regs['weight_reg']['instance'],
                                           muxes['act_mux']['instance'],
                                           muxes['weight_mux']['instance'],
                                           muxes['add_mux']['instance'],
                                           muxes['sum_mux']['instance']
                                          ])

    agraph.direct_constraint(parameters = [fifos['ififo']['instance'],
                                           fifos['ofifo']['instance'],
                                           fifos['wfifo']['instance'],
                                           output_adder['instance'],
                                           act_sram['isram']['query']['width'],
                                           act_sram['osram']['query']['width'],
                                           wsram['query']['width']
                                          #  act_sram['isram']['query']['depth'],
                                          #  act_sram['osram']['query']['depth'],
                                          #  wsram['query']['depth']
                                         ])
    
    agraph.direct_constraint(parameters = [fifos['ififo']['query']['depth'],
                                           fifos['wfifo']['query']['depth'],
                                           fifos['ofifo']['query']['depth']
                                         ])
    
    agraph.direct_constraint(parameters= [act_sram['isram']['query']['depth'],
                                        act_sram['osram']['query']['depth'],
                                        wsram['query']['depth']
                                        ])

    # agraph.direct_constraint(parameters = [llama_3_70b_batch_size['batch_size'], llama_3_8b_batch_size['batch_size']])

    agraph.direct_constraint(parameters = [attributes['technology'], output_adder['query']['class'],
                                           multiplier['query']['class'], adder['query']['class'],
                                           control_regs['act_en_reg']['query']['class'],
                                           control_regs['mult_en_reg']['query']['class'],
                                           control_regs['acc_en_reg']['query']['class'],
                                           control_regs['weight_path_en_reg']['query']['class'],
                                           control_regs['weight_en_reg']['query']['class'],
                                           control_regs['sum_en_reg']['query']['class'],
                                           data_regs['act_reg']['query']['class'],
                                           data_regs['weight_path_reg']['query']['class'],
                                           data_regs['sum_reg']['query']['class'],
                                           data_regs['weight_reg']['query']['class'],
                                           muxes['act_mux']['query']['class'],
                                           muxes['weight_mux']['query']['class'],
                                           muxes['add_mux']['query']['class'],
                                           muxes['sum_mux']['query']['class'],
                                           fifos['ififo']['query']['class'],
                                           fifos['wfifo']['query']['class'],
                                           fifos['ofifo']['query']['class']
                                          ])
    
    agraph.direct_constraint(parameters=[act_sram['isram']['query']['depth'],
                                         act_sram['osram']['query']['depth'],
                                         wsram['query']['depth']
                                        ])

    # direct constraint with condition (functional, only works for two parameters)
    # agraph.direct_constraint_conditional()

    # anti constraint with no condition (works with multiple parameters)
    # agraph.anti_constraint()

    # anti constraint with condition (functional, only works for two parameters)
    # agraph.anti_constraint_conditional()

    # conditional constraint (functional, works with multiple parameters)
    # true = direct, false = anti
    agraph.conditional_constraint(a = fifos['ififo']['instance'],
                                  b = muxes['act_mux']['instance'],
                                  condition = lambda a, b: a[0] == b[1])
    
    agraph.conditional_constraint(a = fifos['wfifo']['instance'],
                                  b = fifos['wfifo']['query']['depth'],
                                  condition = lambda a, b: a[0] >= b)
    
    agraph.conditional_constraint_equation(a = act_sram['isram']['query']['width'],
                                          b = act_sram['isram']['query']['depth'],
                                          c = memory_size_wkld['memory_size_mb'],
                                          condition = lambda a, b, c: a * b * act_sram_bank == c)
  
    agraph.generate()
    return agraph