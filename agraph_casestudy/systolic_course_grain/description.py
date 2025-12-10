from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='systolic_array', yaml_path='agraph_casestudy/systolic_course_grain/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

# SRAM
architecture.new_module(name='isram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 4, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})
architecture.new_module(name='wsram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 4, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})
architecture.new_module(name='osram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 4, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})

# FIFO
architecture.new_module(name=['ififo', 'wfifo', 'ofifo'], instance=[[4], [8], [16], [32]], tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': [4, 8, 16, 32]})

# Output Accumulator
architecture.new_module(name='output_adder', instance=[[4], [8], [16], [32]], tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})

# PE
gemm_sweep = ConditionSweep(value=[4, 4], funct=lambda x: [x[0]*2, x[1]*2], condition=lambda x: x[0] <= 32 and x[1] <= 32)

# operations
architecture.new_module(name='pe', instance=gemm_sweep, tag=['pe', 'array'], query={'class': 'systolic_pe'})

##############################################
###############    Event    ##################
##############################################
event.new_event(name='gemm', subevent=['input_reads', 'weight_reads', 'output_writes', 'input_array'], performance='agraph_casestudy/systolic_course_grain/performance/performance.py')
event.new_event(name='input_reads', subevent=['isram'], performance='agraph_casestudy/systolic_course_grain/performance/performance.py')
event.new_event(name='weight_reads', subevent=['wsram'], performance='agraph_casestudy/systolic_course_grain/performance/performance.py')
event.new_event(name='output_writes', subevent=['osram'], performance='agraph_casestudy/systolic_course_grain/performance/performance.py')
event.new_event(name='input_array', subevent=['ififo', 'wfifo', 'ofifo', 'output_adder', 'pe'],
                performance='agraph_casestudy/systolic_course_grain/performance/performance.py')

##############################################
###############    Metric    #################
##############################################
init_metric(metric)

##############################################
###############   Workload   #################
##############################################
workload.new_configuration(name='gemm')
workload.new_parameter(configuration='gemm', parameter_name='m', parameter_value=[16, 32, 64, 128], sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='k', parameter_value=[16, 32, 64, 128], sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='n', parameter_value=[16, 32, 64, 128], sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='array_dim', parameter_value=[4, 8, 16, 32], sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='bitwidth', parameter_value=16, sweep=False)


constraint_graph.add_constraint(pe=['instance'],
                                ififo=['instance', 'depth'],
                                wfifo=['instance', 'depth'],
                                ofifo=['instance', 'depth'],
                                output_adder=['instance'],
                                isram=['width', 'depth'],
                                wsram=['width', 'depth'],
                                osram=['width', 'depth'],
                                gemm=['m', 'n', 'k', 'array_dim'])

constraint_graph.generate()