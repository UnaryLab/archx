from archx.programming.graph.agraph import AGraph

def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload
    ##############################################
    ################ Architecture ################
    ##############################################
    architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')
    bitwidth = 16

    # array and memory shapes
    act_sram_bank = weight_sram_bank = 2

    #array_shapes = [[128, 128], [256, 256], [512, 64], [512, 128], [512, 512]]
    array_shapes = [[128, 128], [256, 256], [512, 512]]
    fifo_shapes = [[128], [256]]
    chiplet_shape = 64
    act_sram_width = [1024, 2048, 4096]
    weight_sram_width = [1024, 2048, 4096]
    memory_size = 153 * 1024 * 1024

    # chiplet_shape = condition_sweep(value=16, funct=lambda x: x*2, condition=lambda x: x <= 256)

    # act_sram_width = list_sweep(values=array_shapes, funct=lambda x: x[0] * bitwidth / act_sram_bank)
    # weight_sram_width = list_sweep(values=array_shapes, funct=lambda x: x[0] * bitwidth / weight_sram_bank)

    # act_sram_depth = list_sweep(values=act_sram_width, funct=lambda x: memory_size / (x * act_sram_bank))
    # weight_sram_depth = list_sweep(values=weight_sram_width, funct=lambda x: memory_size / (x * weight_sram_bank))

    # # SRAM
    # architecture.add_module(name='isram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': act_sram_bank, 'width': act_sram_width, 'depth': act_sram_depth})
    # architecture.add_module(name='wsram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': weight_sram_bank, 'width': weight_sram_width, 'depth': weight_sram_depth})
    # architecture.add_module(name='osram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': act_sram_bank, 'width': act_sram_width, 'depth': act_sram_depth})

    # # FIFO
    # architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=fifo_shapes, tag=['fifo', 'array'], query={'class': 'fifo', 'width': bitwidth, 'depth': chiplet_shape})
    fifos = architecture.add_module(name=['ififo', 'ofifo'], instance=fifo_shapes, tag=['fifo', 'array'], query={'class': 'fifo', 'width': bitwidth, 'depth': chiplet_shape})

    # Output Accumulator
    # architecture.add_module(name='output_adder', instance=fifo_shapes, tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})

    # operations
    # architecture.add_module(name=['multiplier'], instance=array_shapes, tag=['pe', 'mac', 'array'], query={'class': 'multiplier_bfloat16'})
    # architecture.add_module(name=['adder'], instance=array_shapes, tag=['pe', 'mac', 'array'], query={'class': 'adder_bfloat16'})

    # data registers
    # architecture.add_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=array_shapes, tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
    # architecture.add_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=array_shapes, tag=['pe', 'data', 'array'], query={'class': 'register', 'width': bitwidth})

    # architecture.add_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=array_shapes, tag=['pe', 'array'], query={'class': 'and_gate', 'width': bitwidth})
    muxes = architecture.add_module(name=['act_mux', 'weight_mux'], instance=array_shapes, tag=['pe', 'array'], query={'class': 'and_gate', 'width': bitwidth})

    ##############################################
    ###############    Event    ##################
    ##############################################
    array_events = ['input_reads', 'weight_reads', 'output_writes', 'array_mapping']
    input_mapping_events = ['ififo', 'wfifo', 'ofifo', 'output_adder', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg',
                            'acc_en_reg', 'sum_en_reg', 'act_reg', 'sum_reg', 'act_mux', 'weight_mux', 'add_mux', 'sum_mux']
    weight_mapping_events = ['weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg']

    event.add_event(name='llama_2_7b', subevent=['gemm'], performance='chiplet4ai/llama/performance/llama_model.py')
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
    event.add_event(name='array_mapping', subevent=['input_mapping', 'weight_mapping'], performance='chiplet4ai/common/performance/chiplet_mapping.py')
    event.add_event(name='input_mapping', subevent=input_mapping_events, performance='chiplet4ai/common/performance/chiplet_mapping.py')
    event.add_event(name='weight_mapping', subevent=weight_mapping_events, performance='chiplet4ai/common/performance/chiplet_mapping.py')
    

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
    batch_size = [1, 2, 3]

    llama_2_7b_config = workload.add_configuration(name='llama_2_7b')
    llama_2_7b_config.add_parameter(parameter_name='batch_size', parameter_value=batch_size, sweep=True)
    llama_2_7b_config.add_parameter(parameter_name='dim', parameter_value=4096)
    # llama_2_7b_config.add_parameter(parameter_name='heads',  parameter_value=32)
    # llama_2_7b_config.add_parameter(parameter_name='hidden_dim', parameter_value=11008)
    # llama_2_7b_config.add_parameter(parameter_name='layers', parameter_value=32)
    # llama_2_7b_config.add_parameter(parameter_name='max_sequence_length', parameter_value=4096)
    # llama_2_7b_config.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    # llama_2_7b_config.add_parameter(parameter_name='vocab_size', parameter_value=32000)
    # direct constraint with no condition
    agraph.direct_constraint(parameters = [muxes['act_mux']['instance'],
                                           muxes['weight_mux']['instance']])
    
    agraph.direct_constraint(parameters = [fifos['ififo']['instance'],
                                           fifos['ofifo']['instance']])

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
                                  condition = lambda a, b: a[0] <= b[0])

    sol = agraph.generate()
    
    exit()
    return agraph