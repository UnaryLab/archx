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
    architecture.add_attributes(technology=45, frequency=400, interface='csv_riscv')

    instr_fetch = architecture.add_module(name='instr_fetch', instance=[1], tag=['processor'], query={'class': 'IF'})
    decode = architecture.add_module(name='decode', instance=[1], tag=['processor'], query={'class': 'ID'})
    execute = architecture.add_module(name='execute', instance=[1], tag=['processor'], query={'class': 'EX'})
    compute = architecture.add_module(name='compute', instance=[1], tag=['processor'], query={'class': 'compute'})
    mem = architecture.add_module(name='mem', instance=[1], tag=['processor'], query={'class': 'MEM'})
    wb = architecture.add_module(name='wb', instance=[1], tag=['processor'], query={'class': 'WB'})
    hd = architecture.add_module(name='hd', instance=[1], tag=['processor'], query={'class': 'HD'})
    forward = architecture.add_module(name='forward', instance=[1], tag=['processor'], query={'class': 'f'})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='gemm', subevent=['pipeline', 'add', 'addi', 'andi', 'beq', 'blt', 'jal',
                                           'jalr', 'lui', 'lw', 'slli', 'srli', 'sw', 'addi_0'],
                                           performance='agraph/designs/riscv/performance.py')
    event.add_event(name='pipeline', subevent=['instr_fetch', 'decode', 'execute', 'mem', 'wb', 'hd', 'forward'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='add', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='addi_0', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='addi', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='andi', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='beq', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='blt', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='jal', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='jalr', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='lui', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='lw', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='slli', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='srli', subevent=['compute'], performance='agraph/designs/riscv/performance.py')
    event.add_event(name='sw', subevent=['compute'], performance='agraph/designs/riscv/performance.py')

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
    mapping = gemm.add_parameter(parameter_name='mapping', parameter_value=1)

    return agraph.generate()