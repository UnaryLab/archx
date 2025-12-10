from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, NumSweep, CombinationalSweep

# module <-> module dependency
# module <-> workload dependency
# workload <-> workload dependency

# no dependency (independent)
# direct dependency (sweep together)
# anti dependency (do not sweep together (needs default case))
# conditional dependency (direct / anti / no based on dependency)

design = Design(name='systolic_array', yaml_path='frontend_test/test_outputs/')
dependency_graph = design.get_dependency_graph()
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

condition_sweep_test = ConditionSweep(value=[2], funct=lambda x: x * 2, condition=lambda x: x <= 32)
# print(condition_sweep_test._apply())
num_sweep_test = NumSweep(value=[4], num=4, funct=lambda x: x * 2)
# print(num_sweep_test._apply())
combinational_sweep_test = CombinationalSweep(sweeps=[ConditionSweep(value=2, funct=lambda x: x * 2, condition=lambda x: x <= 16),
                                                      NumSweep(value=4, num=4, funct=lambda x: x * 2)])
# print(combinational_sweep_test._apply())

##############################################
################ Architecture ################
##############################################

array_sweep = ConditionSweep(value=[8, 8], funct=lambda x: x * 2, condition=lambda x: x <= 64)
row_sweep = ConditionSweep(value=[8], funct=lambda x: x * 2, condition=lambda x: x <= 64)

architecture.add_attributes(technology=32, frequency=400, interface='csv_cmos')
architecture.new_module(name='accumulator',              instance=[[8, 8], [16, 16], [32, 32]], tag=['pe'],  query={'class': 'adderbf16'})
architecture.new_module(name='multiplier',               instance=[[8, 8], [16, 16]], tag=['pe'],  query={'class': 'multiplierbf16'})
architecture.new_module(name='input_register',           instance=[[8, 8], [16, 16], [32, 32]], tag=['pe'],  query={'class': 'register', 'width': 16, 'depth': [4, 8]})
architecture.new_module(name='weight_register',          instance=[[8, 8], [16, 16]], tag=['pe'],  query={'class': 'register', 'width': 16, 'depth': [4, 8]})
architecture.new_module(name='partial_sum_register',    instance=[[8, 8], [16, 16], [32, 32], [64, 64]], tag=['pe'],  query={'class': 'register'})
#architecture.new_module(name='row_accumulator_register', instance=[[8], [16]],        tag=['acc'], query={'class': 'register', 'width': [16, 32], 'pipeline': [1, 2], 'depth': [4, 5, 6], 'height': [8, 9, 10]})

##############################################
#################    Event    ################
##############################################

event.new_event(name='gemm',       subevent=['tiled_gemm'])
event.new_event(name='tiled_gemm', subevent=['input_register', 'weight_register', 'partial_sum_register', 'accumulator_register', 'mac'])
event.new_event(name='mac',        subevent=['accumulator, multiplier'])

##############################################
################    Metric    ################
##############################################

metric.new_metric(name='area',           unit='mm^2',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specificied')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

##############################################
##############    Workload    ################
##############################################

gemm_shape_sweep = NumSweep(value=64, num=4, funct=lambda x: x * 2)

workload.new_configuration(name='gemm_config_1')

workload.new_parameter(configuration='gemm_config_1', parameter_name='m', parameter_value=[32, 64, 128], sweep=True)
workload.new_parameter(configuration='gemm_config_1', parameter_name='k', parameter_value=[32, 64, 128], sweep=True)
workload.new_parameter(configuration='gemm_config_1', parameter_name='n', parameter_value=[32, 64, 128], sweep=True)

workload.new_configuration(name='gemm_config_2')

workload.new_parameter(configuration='gemm_config_2', parameter_name='m', parameter_value=[256, 512, 1024], sweep=True)
workload.new_parameter(configuration='gemm_config_2', parameter_name='k', parameter_value=[256, 512, 1024], sweep=True)
workload.new_parameter(configuration='gemm_config_2', parameter_name='n', parameter_value=[256, 512, 1024], sweep=True)


workload.new_configuration(name='gemm_config_3')

workload.new_parameter(configuration='gemm_config_3', parameter_name='m', parameter_value=1024)
workload.new_parameter(configuration='gemm_config_3', parameter_name='k', parameter_value=1024)
workload.new_parameter(configuration='gemm_config_3', parameter_name='n', parameter_value=1024)

workload.new_configuration(name='gemm_config_4')
workload.new_parameter(configuration='gemm_config_4', parameter_name='dim', parameter_value=[64, 64, 64], sweep=False)
workload.new_parameter(configuration='gemm_config_4', parameter_name='batch_size', parameter_value=[4, 8, 16], sweep=True)

##############################################
################ Dependency ##################
##############################################

# two types of conditions:
# 1. single input function that indexes sweep
# 2. comparison between two parameters

# module self-dependency
dependency_graph.add_dependency(a_name='input_register',
                                a_parameters=['instance', 'depth'],
                                condition=lambda i: i < 2)

# configuration self-dependency
dependency_graph.add_dependency(a_name='gemm_config_1',
                                a_parameters=['m', 'k', 'n'])

# module <-> module dependency
dependency_graph.add_dependency(a_name='input_register',
                                a_parameters=['instance'],
                                b_name='multiplier',
                                b_parameters=['instance'],
                                condition= lambda i: i < 2)
dependency_graph.add_dependency(a_name='input_register',
                                a_parameters=['depth'],
                                b_name='weight_register',
                                b_parameters=['depth'])
dependency_graph.add_dependency(a_name='input_register',
                                a_parameters=['instance'],
                                b_name='weight_register',
                                b_parameters=['instance'],
                                condition=lambda i: i < 2)
# dependency_graph.add_dependency(a_name='accumulator',
#                                 a_parameters=['instance'],
#                                 b_name='multiplier',
#                                 b_parameters=['instance'],
#                                 dependency='anti',
#                                 condition=lambda a, b: a == b)
dependency_graph.add_dependency(a_name='accumulator',
                                a_parameters=['instance'],
                                b_name='partial_sum_register',
                                b_parameters=['instance'],
                                condition=lambda i: i < 2)

# configuration <-> configuration dependency
dependency_graph.add_dependency(a_name='gemm_config_1',
                                a_parameters=['m', 'k', 'n'],
                                b_name='gemm_config_2',
                                b_parameters=['m', 'k', 'n'])

dependency_graph.add_dependency(a_name='multiplier',
                                a_parameters=['instance'],
                                b_name='gemm_config_1',
                                b_parameters=['m', 'k', 'n'],
                                dependency='anti',
                                condition=lambda a, b: (a == [8, 8]))
dependency_graph.add_dependency(a_name='accumulator',
                                a_parameters=['instance'],
                                b_name='gemm_config_1',
                                b_parameters=['m', 'k', 'n'])

##############################################
############## Save to YAML  #################
##############################################

#dependency_graph.generate()
dependency_graph.generate_test()
# design.to_yaml()