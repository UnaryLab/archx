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
    array_dim = [[4, 4], [8, 8], [16, 16], [32, 32]]

    # SRAM
    srams = architecture.add_module(name=['isram', 'wsram', 'osram'], instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})

    # FIFO
    fifos = architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=[[4], [8], [16], [32]], tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': [4, 8, 16, 32]})

    # Output Accumulator
    output_adder = architecture.add_module(name='output_adder', instance=[[4], [8], [16], [32]], tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})

    # operations
    multiplier = architecture.add_module(name='multiplier', instance=array_dim, tag=['pe', 'mac', 'array'], query={'class': 'multiplier_bfloat16'})
    adder = architecture.add_module(name='adder', instance=array_dim, tag=['pe', 'mac', 'array'], query={'class': 'adder_bfloat16'})

    # data registers
    ctrl_regs = architecture.add_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=array_dim, tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
    data_regs = architecture.add_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=array_dim, tag=['pe', 'data', 'array'], query={'class': 'register', 'width': 16})

    muxes = architecture.add_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=array_dim, tag=['pe', 'array'], query={'class': 'and_gate', 'width': 16})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='gemm', subevent=['input_reads', 'weight_reads', 'output_writes', 'input_array', 'weight_array'], performance='agraph/designs/systolic_fg/performance.py')
    event.add_event(name='input_reads', subevent=['isram'], performance='agraph/designs/systolic_fg/performance.py')
    event.add_event(name='weight_reads', subevent=['wsram'], performance='agraph/designs/systolic_fg/performance.py')
    event.add_event(name='output_writes', subevent=['osram'], performance='agraph/designs/systolic_fg/performance.py')
    event.add_event(name='input_array', subevent=['ififo', 'wfifo', 'ofifo', 'output_adder', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg', 'acc_en_reg', 'sum_en_reg', 'act_reg', 'sum_reg',
                                                'act_mux', 'weight_mux', 'add_mux', 'sum_mux'], performance='agraph/designs/systolic_fg/performance.py')
    event.add_event(name='weight_array', subevent=['weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg'], performance='agraph/designs/systolic_fg/performance.py')

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
    gemm = workload.add_configuration(name='gemm')
    matrix_dim = gemm.add_parameter(parameter_name='matrix_dim', parameter_value=[16, 32, 64, 128], sweep=True)

    agraph.direct_constraint([
        multiplier['instance'],
        adder['instance'],
        ctrl_regs['act_en_reg']['instance'],
        ctrl_regs['mult_en_reg']['instance'],
        ctrl_regs['acc_en_reg']['instance'],
        ctrl_regs['weight_path_en_reg']['instance'],
        ctrl_regs['weight_en_reg']['instance'],
        ctrl_regs['sum_en_reg']['instance'],
        data_regs['act_reg']['instance'],
        data_regs['weight_path_reg']['instance'],
        data_regs['sum_reg']['instance'],
        data_regs['weight_reg']['instance'],
        fifos['ififo']['instance'],
        fifos['wfifo']['instance'],
        fifos['ofifo']['instance'],
        fifos['ififo']['query']['depth'],
        fifos['wfifo']['query']['depth'],
        fifos['ofifo']['query']['depth'],
        output_adder['instance'],
        srams['isram']['query']['width'],
        srams['wsram']['query']['width'],
        srams['osram']['query']['width'],
        srams['isram']['query']['depth'],
        srams['wsram']['query']['depth'],
        srams['osram']['query']['depth'],
        muxes['act_mux']['instance'],
        muxes['weight_mux']['instance'],
        muxes['add_mux']['instance'],
        muxes['sum_mux']['instance'],
        matrix_dim['parameter']
    ])

    return agraph.generate()