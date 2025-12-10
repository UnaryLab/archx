from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='systolic_array', yaml_path='agraph_casestudy/systolic_pe/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

# PE
bitwidth = [4, 8, 16, 32]

# operations
architecture.new_module(name=['multiplier'], instance=[1], tag=['pe', 'mac', 'array'], query={'class': 'multiplier', 'width': bitwidth})
architecture.new_module(name=['adder'], instance=[1], tag=['pe', 'mac', 'array'], query={'class': 'adder', 'width': bitwidth})

# data registers
architecture.new_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=[1], tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
architecture.new_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=[1], tag=['pe', 'data', 'array'], query={'class': 'register', 'width': bitwidth})

architecture.new_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=[1], tag=['pe', 'array'], query={'class': 'and_gate', 'width': bitwidth})

##############################################
###############    Event    ##################
##############################################
event.new_event(name='gemm', subevent=['multiplier', 'adder', 'act_en_reg', 'mult_en_reg', 'acc_en_reg', 'sum_en_reg', 'act_reg', 'sum_reg',
                                       'act_mux', 'weight_mux', 'add_mux', 'sum_mux'],
                             performance='agraph_casestudy/systolic_pe/performance/performance.py')

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


constraint_graph.add_constraint(multiplier=['width'],
                                adder=['width'],
                                act_reg=['width'],
                                weight_path_reg=['width'],
                                sum_reg=['width'],
                                weight_reg=['width'],
                                act_mux=['width'],
                                weight_mux=['width'],
                                add_mux=['width'],
                                sum_mux=['width'],
                                gemm=['m', 'n', 'k', 'array_dim'])

constraint_graph.generate()