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
    srams = architecture.add_module(name=['isram', 'wsram', 'osram'], instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 4, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})

    # FIFO
    fifos = architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=[[4], [8], [16], [32]], tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': [4, 8, 16, 32]})

    # Output Accumulator
    output_adder = architecture.add_module(name='output_adder', instance=[[4], [8], [16], [32]], tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})

    pe = architecture.add_module(name='pe', instance=array_dim, tag=['pe', 'array'], query={'class': 'systolic_pe'})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='gemm', subevent=['input_reads', 'weight_reads', 'output_writes', 'input_array'], performance='zoo/agraph/designs/systolic_cg/performance.py')
    event.add_event(name='input_reads', subevent=['isram'], performance='zoo/agraph/designs/systolic_cg/performance.py')
    event.add_event(name='weight_reads', subevent=['wsram'], performance='zoo/agraph/designs/systolic_cg/performance.py')
    event.add_event(name='output_writes', subevent=['osram'], performance='zoo/agraph/designs/systolic_cg/performance.py')
    event.add_event(name='input_array', subevent=['ififo', 'wfifo', 'ofifo', 'output_adder', 'pe'], performance='zoo/agraph/designs/systolic_cg/performance.py')

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
        srams['isram']['query']['width'],
        srams['wsram']['query']['width'],
        srams['osram']['query']['width'],
        srams['isram']['query']['depth'],
        srams['wsram']['query']['depth'],
        srams['osram']['query']['depth'],
        fifos['ififo']['instance'],
        fifos['wfifo']['instance'],
        fifos['ofifo']['instance'],
        fifos['ififo']['query']['depth'],
        fifos['wfifo']['query']['depth'],
        fifos['ofifo']['query']['depth'],
        output_adder['instance'],
        pe['instance'],
        matrix_dim['parameter']
    ])

    return agraph.generate()