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

    architecture.add_attributes(technology=10, frequency=50, interface='csv_sc')

    architecture.add_module(name='mac', instance=[64, 64, 32], tag=['array', 'pe', 'comp', 'mac'], query={'class': 'sc_mac'})
    architecture.add_module(name='mac_splitter', instance=[64, 64, 32], tag=['array', 'comp', 'pe', 'mac_splitter'], query={'class': 'sc_splitter'})
    architecture.add_module(name='weight_splitter', instance=[64, 64, 32], tag=['array', 'comp', 'pe', 'weight_splitter'], query={'class': 'sc_splitter'})
    architecture.add_module(name='input_splitter', instance=[64, 64, 32], tag=['array', 'comp', 'pe', 'input_splitter'], query={'class': 'sc_splitter'})
    architecture.add_module(name='nrdo', instance=[32], tag=['array', 'pe', 'nrdo'], query={'class': 'sc_nrdo'})
    architecture.add_module(name='ptl', instance=[64, 64, 32], tag=['array', 'overhead'], query={'class': 'sc_ptl'})

    event.add_event(name='vgg16', subevent=['mac', 'mac_splitter', 'input_splitter', 'weight_splitter', 'nrdo', 'ptl'], performance='zoo/agraph/designs/sc_cnn/performance.py')

    metric.add_metric(name='area',           unit='jj',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    vgg16 = workload.add_configuration(name='vgg16')
    vgg16.add_parameter(parameter_name='conv3-64', parameter_value=[3, 3, 64], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-64_of', parameter_value=[[224, 224, 64], [224, 224, 64]], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-128', parameter_value=[3, 3, 128], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-128_of', parameter_value=[[112, 112, 128], [112, 112, 128]], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-256', parameter_value=[3, 3, 256], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-256_of', parameter_value=[[56, 56, 256], [56, 56, 256], [56, 56, 256]], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-512', parameter_value=[3, 3, 512], sweep=False)
    vgg16.add_parameter(parameter_name='conv3-512_of', parameter_value=[[28, 28, 512], [28, 28, 512], [28, 28, 512], [14, 14, 512], [14, 14, 512], [14, 14, 512]], sweep=False)
    vgg16.add_parameter(parameter_name='bitwidth', parameter_value=4, sweep=False)

    return agraph.generate()