from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
import math

design = Design(name='tnn', yaml_path='agraph_casestudy/tnn_cg/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

p_dim = [96, 152, 342]
adder_inst = [0, 0, 0]
for a_i, p in enumerate(p_dim):
    i = math.ceil(math.log2(p))
    stage = i
    stage_width = p
    while True:
        adder_inst[a_i] += stage_width * (stage - i + 1)
        if stage_width == 1:
            break
        stage_width /= 2
        stage_width = math.ceil(stage_width)
        i -= 1

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=7, frequency=(100), interface='csv_cmos')
architecture.new_module(name='edge2pulse', instance=[1], tag=['column'], query={'class': 'edge2pulse'}) # 1
architecture.new_module(name='pulse2edge', instance=[1], tag=['column'], query={'class': 'pulse2edge_96', 'width': [98, 154, 345]}) # 3 + P
architecture.new_module(name='fsm_simple', instance=[2], tag=['column'], query={'class': 'fsm_simple'}) # check
architecture.new_module(name='WTA', instance=[1], tag=['column', 'wta'], query={'class': 'wta'}) # check 
architecture.new_module(name='top_reg', instance=[1], tag=['column'], query={'class': 'tnn_reg_3', 'width': 1}) # check
architecture.new_module(name='neuron', instance=[2], tag=['column'], query={'class': 'pac', 'width': [96, 152, 343]})
architecture.new_module(name='stdp', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'stdp'})
architecture.new_module(name='fsm_synapse', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'fsm_synapse'})

event.new_event(name='tnn',
                subevent=['edge2pulse', 'pulse2edge', 'WTA', 'top_reg', 'neuron', 'stdp', 'fsm_synapse', 'fsm_simple'],
                performance='agraph_casestudy/tnn_cg/performance/performance.py')

metric.new_metric(name='area',           unit='mm^2',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specified')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

workload.new_configuration(name='tnn')
workload.new_parameter(configuration='tnn', parameter_name='mappings', parameter_value=1)
workload.new_parameter(configuration='tnn', parameter_name='column', parameter_value=[[96, 2], [152, 2], [342, 2]], sweep=True)
workload.new_parameter(configuration='tnn', parameter_name='array', parameter_value=[[96, 2], [152, 2], [342, 2]], sweep=True)

constraint_graph.add_constraint(pulse2edge=['width'], neuron=['width'], stdp=['instance'], fsm_synapse=['instance'], tnn=['column', 'array'])
constraint_graph.add_constraint()

constraint_graph.generate()