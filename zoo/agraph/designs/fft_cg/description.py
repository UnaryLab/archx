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
    
    pe = architecture.add_module(name='pe', instance=fft_instance, tag=['pe', 'array'], query={'class': 'fft_pe'})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='butterfly', subevent=['pe'], performance='zoo/agraph/designs/fft_cg/performance.py')

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
    return agraph.generate()