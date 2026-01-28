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

    # SRAM
    architecture.add_module(name='isram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})
    architecture.add_module(name='wsram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})
    architecture.add_module(name='osram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})

    # FIFO
    architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=[[4], [8], [16], [32]], tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': [4, 8, 16, 32]})

    # Output Accumulator
    architecture.add_module(name='output_adder', instance=[[4], [8], [16], [32]], tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})

    gemm_list = [[4, 4], [8, 8], [16, 16], [32, 32]]
    # operations
    architecture.add_module(name=['multiplier'], instance=gemm_list, tag=['pe', 'mac', 'array'], query={'class': 'multiplier_bfloat16'})
    architecture.add_module(name=['adder'], instance=gemm_list, tag=['pe', 'mac', 'array'], query={'class': 'adder_bfloat16'})

    # data registers
    architecture.add_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=gemm_list, tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
    architecture.add_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=gemm_list, tag=['pe', 'data', 'array'], query={'class': 'register', 'width': 16})

    architecture.add_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=gemm_list, tag=['pe', 'array'], query={'class': 'and_gate', 'width': 16})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='gemm', subevent=['input_reads', 'weight_reads', 'output_writes', 'input_array', 'weight_array'], performance='examples/systolic_array/description.py')
    event.add_event(name='input_reads', subevent=['isram'], performance='examples/systolic_array/description.py')
    event.add_event(name='weight_reads', subevent=['wsram'], performance='examples/systolic_array/description.py')
    event.add_event(name='output_writes', subevent=['osram'], performance='examples/systolic_array/description.py')
    event.add_event(name='input_array', subevent=['ififo', 'wfifo', 'ofifo', 'output_adder', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg', 'acc_en_reg', 'sum_en_reg', 'act_reg', 'sum_reg',
                                                'act_mux', 'weight_mux', 'add_mux', 'sum_mux'],
                    performance='examples/systolic_array/description.py')
    event.add_event(name='weight_array', subevent=['weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg'],
                    performance='examples/systolic_array/description.py')

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
    workload.add_configuration(name='gemm')
    workload.add_parameter(configuration='gemm', parameter_name='matrix_dim', parameter_value=[16, 32, 64, 128], sweep=True)

    agraph.add_constraint(multiplier=['instance'],
                        adder=['instance'],
                        act_en_reg=['instance'],
                        mult_en_reg=['instance'],
                        acc_en_reg=['instance'],
                        weight_path_en_reg=['instance'],
                        weight_en_reg=['instance'],
                        sum_en_reg=['instance'],
                        act_reg=['instance'],
                        weight_path_reg=['instance'],
                        sum_reg=['instance'],
                        weight_reg=['instance'],
                        ififo=['instance', 'depth'],
                        wfifo=['instance', 'depth'],
                        ofifo=['instance', 'depth'],
                        output_adder=['instance'],
                        isram=['width', 'depth'],
                        wsram=['width', 'depth'],
                        osram=['width', 'depth'],
                        act_mux=['instance'],
                        weight_mux=['instance'],
                        add_mux=['instance'],
                        sum_mux=['instance'],
                        gemm=['matrix_dim']
                        )

    agraph.generate()
    return agraph