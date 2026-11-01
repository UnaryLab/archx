from archx.programming.graph.agraph import AGraph
import json

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

    dims = [16, 32, 64, 128, 256]
    array_dim = [[x, y, z] for x in dims for y in dims for z in dims]
    nrdo_dim = [[z] for x, y, z in array_dim]

    mac = architecture.add_module(name='mac', instance=array_dim, tag=['array', 'pe', 'comp', 'mac'], query={'class': 'sc_mac'})
    mac_splitter = architecture.add_module(name='mac_splitter', instance=array_dim, tag=['array', 'comp', 'pe', 'mac_splitter'], query={'class': 'sc_splitter'})
    weight_splitter = architecture.add_module(name='weight_splitter', instance=array_dim, tag=['array', 'comp', 'pe', 'weight_splitter'], query={'class': 'sc_splitter'})
    input_splitter = architecture.add_module(name='input_splitter', instance=array_dim, tag=['array', 'comp', 'pe', 'input_splitter'], query={'class': 'sc_splitter'})
    nrdo = architecture.add_module(name='nrdo', instance=nrdo_dim, tag=['array', 'pe', 'nrdo'], query={'class': 'sc_nrdo'})
    ptl = architecture.add_module(name='ptl', instance=array_dim, tag=['array', 'overhead'], query={'class': 'sc_ptl'})

    event.add_event(name='cnn', subevent=['mac', 'mac_splitter', 'input_splitter', 'weight_splitter', 'nrdo', 'ptl'], performance='zoo/agraph/designs/sc_cnn_dse/performance.py')

    metric.add_metric(name='area',           unit='jj',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    # conv layers only, [R, C, M, K_h, K_w, Z, H_in, W_in, stride] per layer, from scale_sim_conv/convert.py
    cnn = workload.add_configuration(name=['vgg16', 'resnet18', 'resnet50', 'googlenet', 'alexnet'])
    for name, config in cnn.items():
        with open(f'zoo/agraph/designs/sc_cnn_dse/scale_sim_conv/{name}.json') as f:
            layers = [[layer[k] for k in ['R', 'C', 'M', 'K_h', 'K_w', 'Z', 'H_in', 'W_in', 'stride']] for layer in json.load(f)['layers']]
        config.add_parameter(parameter_name='layers', parameter_value=layers, sweep=False)
    workload.add_parameters(list(cnn.values()), parameter_name='bitwidth', parameter_value=5)

    agraph.direct_constraint([
        mac['instance'],
        mac_splitter['instance'],
        weight_splitter['instance'],
        input_splitter['instance'],
        nrdo['instance'],
        ptl['instance']
    ])

    return agraph.generate()
