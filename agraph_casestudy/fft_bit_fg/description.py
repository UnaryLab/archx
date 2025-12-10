from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='fft', yaml_path='agraph_casestudy/fft_bit_fg/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

architecture.new_module(name=['regA', 'regB'], instance=[8, 4], tag=['register', 'array'], query={'class': 'register', 'width': [4, 8, 16, 32]})
architecture.new_module(name='multiplier', instance=[8, 4], tag=['multiplier', 'array'], query={'class': 'mult_4', 'width': [4, 8, 16, 32]})
architecture.new_module(name=['adderA', 'adderB'], instance=[8, 4], tag=['adder', 'array'], query={'class': 'adder_8', 'width': [4, 16, 32, 64]})
architecture.new_module(name='not_gate', instance=[8, 4], tag=['not_gate', 'array'], query={'class': 'not_gate'})

##############################################
###############    Event    ##################
##############################################
event.new_event(name='butterfly', subevent=['regA', 'regB', 'multiplier', 'adderA', 'adderB', 'not_gate'], performance='agraph_casestudy/fft_bit_fg/performance/performance.py')

##############################################
###############    Metric    #################
##############################################
init_metric(metric)

##############################################
###############   Workload   #################
##############################################
workload.new_configuration(name='butterfly')
workload.new_parameter(configuration='butterfly', parameter_name='input_dim', parameter_value=16)
workload.new_parameter(configuration='butterfly', parameter_name='matrix_dim', parameter_value=[16, 16], sweep=False)
workload.new_parameter(configuration='butterfly', parameter_name='bitwidth', parameter_value=[4, 8, 16, 32], sweep=True)
workload.new_parameter(configuration='butterfly', parameter_name='width', parameter_value=[4, 8, 16, 32], sweep=True)
workload.new_parameter(configuration='butterfly', parameter_name='mappings', parameter_value=1)


constraint_graph.add_constraint(regA = ['width'],
                                regB = ['width'],
                                multiplier = ['width'],
                                adderA = ['width'],
                                adderB = ['width'],
                                butterfly = ['bitwidth', 'width'])

constraint_graph.generate()