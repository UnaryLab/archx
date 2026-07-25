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
    p_dim = [96, 152, 342]
    adder_inst = [sum(math.ceil(p / (2**k)) * (k + 1) for k in range(math.ceil(math.log2(p)) + 1)) for p in p_dim]

    architecture.add_attributes(technology=7, frequency=(0.1), interface='csv_cmos')
    edge2pulse = architecture.add_module(name='edge2pulse', instance=[1], tag=['column'], query={'class': 'edge2pulse'})
    pulse2edge = architecture.add_module(name='pulse2edge', instance=[[99], [155], [345]], tag=['column'], query={'class': 'pulse2edge'})
    wta = architecture.add_module(name='WTA', instance=[2], tag=['column', 'wta'], query={'class': 'less_equal'})
    stdp_case_gen = architecture.add_module(name='stdp_case_gen', instance=[[96, 2], [152, 2], [343, 2]], tag=['stdp', 'column'], query={'class': 'stdp_case_gen'})
    syn_readout = architecture.add_module(name='syn_readout', instance=[[96, 2], [152, 2], [343, 2]], tag=['weight', 'column'], query={'class': 'fsm_output'})
    syn_weight_update = architecture.add_module(name='syn_weight_update', instance=[[96, 2], [152, 2], [343, 2]], tag=['weight', 'column'], query={'class': 'fsm_weight_update'})
    spike_gen = architecture.add_module(name='spike_gen', instance=[2], tag=['weight', 'column'], query={'class': 'fsm_simple'})
    top_reg = architecture.add_module(name='top_reg', instance=[1], tag=['column'], query={'class': 'tnn_reg_3', 'width': 1})
    fsm_reg_1 = architecture.add_module(name='fsm_reg_1', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'tnn_reg_3', 'width': 1})
    fsm_reg_2 = architecture.add_module(name='fsm_reg_2', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'tnn_reg_3', 'width': 3})
    adder = architecture.add_module(name='adder', instance=[[2, 120], [2, 247], [2, 502]], tag=['column'], query={'class': 'tnn_adder', 'width': 1})
    adder_output = architecture.add_module(name='adder_output', instance=[2], tag=['column'], query={'class': 'tnn_adder', 'width': 5})
    muxout_reg = architecture.add_module(name='muxout_reg', instance=[2], tag=['column'], query={'class': 'tnn_reg_3', 'width': 5})
    wta_and = architecture.add_module(name='wta_and', instance=[2], tag=['column'], query={'class': 'tnn_and_gate', 'width': 1})
    pac_and = architecture.add_module(name='pac_and', instance=[2, 1], tag=['column'], query={'class': 'tnn_and_gate', 'width': 1})

    event.add_event(name='tnn',
                    subevent=['edge2pulse', 'pulse2edge', 'WTA', 'stdp_case_gen', 'syn_readout', 'syn_weight_update', 'spike_gen',
                            'top_reg', 'fsm_reg_1', 'fsm_reg_2', 'adder', 'adder_output', 'muxout_reg', 'wta_and', 'pac_and'],
                    performance='zoo/agraph/designs/tnn_fg/performance.py')

    metric.add_metric(name='area',           unit='mm^2',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    tnn = workload.add_configuration(name='tnn')
    tnn.add_parameter(parameter_name='mappings', parameter_value=1)

    agraph.direct_constraint([
        pulse2edge['instance'],
        stdp_case_gen['instance'],
        syn_readout['instance'],
        syn_weight_update['instance'],
        fsm_reg_1['instance'],
        fsm_reg_2['instance'],
        adder['instance']
    ])
    
    return agraph.generate()