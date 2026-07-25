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

    bitwidth = [4, 6, 8, 10, 12, 14, 16]
    array_dim = [[256], [32]]

    mult = architecture.add_module(name='mult', instance=array_dim, tag=['tap', 'mac'], query={'class': 'sc_mult_fir'})
    acc = architecture.add_module(name='acc', instance=array_dim, tag=['tap', 'mac'], query={'class': 'sc_balancer'})
    shift_reg = architecture.add_module(name='shift_reg', instance=array_dim, tag=['tap'], query={'class': 'sc_shift_reg'})
    pnm = architecture.add_module(name='pnm', instance=array_dim, tag=['tap'], query={'class': 'sc_pnm',  'width': bitwidth})
    b2rc = architecture.add_module(name='b2rc', instance=array_dim, tag=['tap'], query={'class': 'sc_b2rc', 'width': 16})
    control = architecture.add_module(name='control', instance=[1], tag=['tap'], query={'class': 'sc_fir_control'})

    event.add_event(name='fir', subevent=['mult', 'acc', 'shift_reg', 'pnm', 'b2rc', 'control'], performance='zoo/agraph/designs/sc_fir/performance.py')

    metric.add_metric(name='area',           unit='jj',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    fir = workload.add_configuration(name='fir')
    fir.add_parameter(parameter_name='mappings', parameter_value=1)

    agraph.direct_constraint([
        mult['instance'],
        acc['instance'],
        shift_reg['instance'],
        pnm['instance'],
        b2rc['instance']
    ])

    return agraph.generate()