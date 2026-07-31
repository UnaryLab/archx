from archx.programming.graph.agraph import AGraph
import math

def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ################ Architecture ################
    ##############################################
    architecture.add_attributes(technology=7, frequency=(0.1), interface='csv_cmos')
    architecture.add_module(name='edge2pulse', instance=[1], tag=['column'], query={'class': 'edge2pulse'})
    pulse2edge = architecture.add_module(name='pulse2edge', instance=[1], tag=['column'], query={'class': 'pulse2edge_96', 'width': [98, 154, 345]})
    architecture.add_module(name='fsm_simple', instance=[2], tag=['column'], query={'class': 'fsm_simple'})
    architecture.add_module(name='WTA', instance=[1], tag=['column', 'wta'], query={'class': 'wta'})
    architecture.add_module(name='top_reg', instance=[1], tag=['column'], query={'class': 'tnn_reg_3', 'width': 1})
    neuron = architecture.add_module(name='neuron', instance=[2], tag=['column'], query={'class': 'pac', 'width': [96, 152, 343]})
    stdp = architecture.add_module(name='stdp', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'stdp'})
    fsm_synapse = architecture.add_module(name='fsm_synapse', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'fsm_synapse'})

    event.add_event(name='tnn',
                    subevent=['edge2pulse', 'pulse2edge', 'WTA', 'top_reg', 'neuron', 'stdp', 'fsm_synapse', 'fsm_simple'],
                    performance='agraph/designs/tnn_cg/performance.py')

    metric.add_metric(name='area',           unit='mm^2',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    tnn = workload.add_configuration(name='tnn')
    tnn.add_parameter(parameter_name='mappings', parameter_value=1)

    agraph.direct_constraint([
        pulse2edge['query']['width'],
        neuron['query']['width'],
        stdp['instance'],
        fsm_synapse['instance']
    ])
    
    return agraph.generate()