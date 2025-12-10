from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='fft', yaml_path='agraph_casestudy/fft_course_grain/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

fft_instance = ConditionSweep(value=[4, 3], funct=lambda x: [x[0]*2, x[1]+1], condition=lambda x: x[0] <= 32)
input_dim_sweep = ConditionSweep(value=8, funct=lambda x: x*2, condition=lambda x: x <= 64)
matrix_dim_sweep = ConditionSweep(value=[8, 8], funct=lambda x: [x[0]*2, x[1]*2], condition=lambda x: x[0] <= 64 and x[1] <= 64)

architecture.new_module(name='pe', instance=fft_instance, tag=['pe', 'array'], query={'class': 'fft_pe'})
architecture.new_module(name='test', instance=fft_instance, tag=['test'], query={'class': 'test'})

##############################################
###############    Event    ##################
##############################################
event.new_event(name='butterfly', subevent=['pe'], performance='agraph_casestudy/fft_course_grain/performance/performance.py')

##############################################
###############    Metric    #################
##############################################
init_metric(metric)

##############################################
###############   Workload   #################
##############################################
workload.new_configuration(name='butterfly')
workload.new_parameter(configuration='butterfly', parameter_name='input_dim', parameter_value=input_dim_sweep)
workload.new_parameter(configuration='butterfly', parameter_name='matrix_dim', parameter_value=matrix_dim_sweep)
workload.new_parameter(configuration='butterfly', parameter_name='mappings', parameter_value=1)


constraint_graph.add_constraint(pe = ['instance'],
                                test = ['instance'],
                                butterfly = ['input_dim', 'matrix_dim'])

constraint_graph.generate()