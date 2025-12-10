from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='fft', yaml_path='agraph_casestudy/fft_bit_cg/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')


architecture.new_module(name='pe', instance=[8, 4], tag=['pe', 'array'], query={'class': 'pe_16', 'width': [4, 8, 16, 32]})
architecture.new_module(name='test', instance=[[1], [2], [3], [4]], tag=['test'], query={'class': 'test'})

##############################################
###############    Event    ##################
##############################################
event.new_event(name='butterfly', subevent=['pe'], performance='agraph_casestudy/fft_bit_cg/performance/performance.py')

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


constraint_graph.add_constraint(pe = ['width'],
                                test = ['instance'],
                                butterfly = ['bitwidth', 'width'])

constraint_graph.generate()