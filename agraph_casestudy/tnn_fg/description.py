from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
import math

design = Design(name='tnn', yaml_path='agraph_casestudy/tnn/description/')
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
architecture.add_attributes(technology=7, frequency=(0.1), interface='csv_cmos')
architecture.new_module(name='edge2pulse', instance=[1], tag=['column'], query={'class': 'edge2pulse'}) # 1
architecture.new_module(name='pulse2edge', instance=[[99], [155], [345]], tag=['column'], query={'class': 'pulse2edge'}) # 3 + P
architecture.new_module(name='WTA', instance=[2], tag=['column', 'wta'], query={'class': 'less_equal'}) # check 
architecture.new_module(name='stdp_case_gen', instance=[[96, 2], [152, 2], [343, 2]], tag=['stdp', 'column'], query={'class': 'stdp_case_gen'}) # check
#architecture.new_module(name='incdec', instance=[[96, 2], [152, 2], [343, 2]], tag=['stdp', 'column'], query={'class': 'incdec'}) # check
#architecture.new_module(name='stabilize_func', instance=[[96, 2], [152, 2], [343, 2]], tag=['stdp', 'column'], query={'class': 'flogic'}) # check
architecture.new_module(name='syn_readout', instance=[[96, 2], [152, 2], [343, 2]], tag=['weight', 'column'], query={'class': 'fsm_output'}) # check
architecture.new_module(name='syn_weight_update', instance=[[96, 2], [152, 2], [343, 2]], tag=['weight', 'column'], query={'class': 'fsm_weight_update'}) # check
architecture.new_module(name='spike_gen', instance=[2], tag=['weight', 'column'], query={'class': 'fsm_simple'}) # check
architecture.new_module(name='top_reg', instance=[1], tag=['column'], query={'class': 'tnn_reg_3', 'width': 1}) # check
architecture.new_module(name='fsm_reg_1', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'tnn_reg_3', 'width': 1}) # check
architecture.new_module(name='fsm_reg_2', instance=[[96, 2], [152, 2], [343, 2]], tag=['column'], query={'class': 'tnn_reg_3', 'width': 3}) # check
architecture.new_module(name='adder', instance=[[2, 120], [2, 247], [2, 502]], tag=['column'], query={'class': 'tnn_adder', 'width': 1}) #check
architecture.new_module(name='adder_output', instance=[2], tag=['column'], query={'class': 'tnn_adder', 'width': 5}) # check
architecture.new_module(name='muxout_reg', instance=[2], tag=['column'], query={'class': 'tnn_reg_3', 'width': 5}) # check
architecture.new_module(name='wta_and', instance=[2], tag=['column'], query={'class': 'tnn_and_gate', 'width': 1}) # check
architecture.new_module(name='pac_and', instance=[2, 1], tag=['column'], query={'class': 'tnn_and_gate', 'width': 1}) # check

event.new_event(name='tnn',
                subevent=['edge2pulse', 'pulse2edge', 'WTA', 'stdp_case_gen', 'incdec', 'stabilize_func', 'syn_readout', 'syn_weight_update', 'spike_gen',
                          'top_reg', 'fsm_reg_1', 'fsm_reg_2', 'adder', 'adder_output', 'muxout_reg', 'wta_and', 'pac_and'],
                performance='agraph_casestudy/tnn/performance/performance.py')

metric.new_metric(name='area',           unit='mm^2',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specified')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

workload.new_configuration(name='tnn')
workload.new_parameter(configuration='tnn', parameter_name='mappings', parameter_value=1)
workload.new_parameter(configuration='tnn', parameter_name='column', parameter_value=[[96, 2], [152, 2], [342, 2]], sweep=True)
workload.new_parameter(configuration='tnn', parameter_name='array', parameter_value=[[96, 2], [152, 2], [342, 2]], sweep=True)

constraint_graph.add_constraint(pulse2edge=['instance'], stdp_case_gen=['instance'], incdec=['instance'], stabilize_func=['instance'], syn_readout=['instance'],
                                syn_weight_update=['instance'], fsm_reg_1=['instance'], fsm_reg_2=['instance'], adder=['instance'], tnn=['column', 'array'])
constraint_graph.add_constraint()

constraint_graph.generate()