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

    fft_instance = [[4, 3], [8, 4], [16, 5], [32, 6]]
    
    regs = architecture.add_module(name=['regA', 'regB'], instance=fft_instance, tag=['register', 'array'], query={'class': 'register', 'width': 16})
    multiplier = architecture.add_module(name='multiplier', instance=fft_instance, tag=['multiplier', 'array'], query={'class': 'multiplier_bfloat16'})
    adders = architecture.add_module(name=['adderA', 'adderB'], instance=fft_instance, tag=['adder', 'array'], query={'class': 'adder_bfloat16'})
    not_gate = architecture.add_module(name='not_gate', instance=fft_instance, tag=['not_gate', 'array'], query={'class': 'not_gate'})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='butterfly', subevent=['regA', 'regB', 'multiplier', 'adderA', 'adderB', 'not_gate'], performance='agraph/designs/fft_fg/performance.py')

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

    butterfly = workload.add_configuration(name='butterfly')
    butterfly.add_parameter(parameter_name='mappings', parameter_value=1)

    agraph.direct_constraint([
        regs['regA']['instance'],
        regs['regB']['instance'],
        multiplier['instance'],
        adders['adderA']['instance'],
        adders['adderB']['instance'],
        not_gate['instance']
    ])

    return agraph.generate()