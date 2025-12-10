from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, NumSweep, CombinationalSweep

design = Design(name='systolic_array', yaml_path='frontend_test/test_outputs/list/')
architecture = design.new_architecture_description(name='mac_systolic_array')
event = design.new_event_description(name='mac_systolic_array', performance_path='systolic/performance/')
metric = design.new_metric_description(name='metric')
workload = design.new_workload_description(name='GEMM')
sa_configuration_1 = workload.new_configuration(name='gemm_config_1')
sa_configuration_2 = workload.new_configuration(name='gemm_config_2')

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
architecture.new_module(name='accumulator',              instance=array_sweep, tag=['pe'],  query={'class': 'adderbf16'})
architecture.new_module(name='multiplier',               instance=array_sweep, tag=['pe'],  query={'class': 'multiplierbf16'})
architecture.new_module(name='input_register',           instance=array_sweep, tag=['pe'],  query={'class': 'register', 'width': [8]})
architecture.new_module(name='weight_register',          instance=array_sweep, tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='partial_sum_register',     instance=array_sweep, tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='accumulator_register',     instance=array_sweep, tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='row_accumulator',          instance=row_sweep,   tag=['acc'], query={'class': 'adderbf16'})
architecture.new_module(name='row_accumulator_register', instance=row_sweep,   tag=['acc'], query={'class': 'register', 'width': 16})
##############################################
#################    Event    ################
##############################################

event.new_event(name='gemm',       subevent=['tiled_gemm'])
event.new_event(name='tiled_gemm', subevent=['input_register', 'weight_register', 'partial_sum_register', 'accumulator_register', 'mac'])
event.new_event(name='mac',        subevent=['accumulator, multiplier'])

##############################################
################    Metric    ################
##############################################8

metric.new_metric(name='area',           unit='mm^2',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specificied')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

##############################################
##############    Workload    ################
##############################################

gemm_shape_sweep = NumSweep(value=64, num=4, funct=lambda x: x * 2)

sa_configuration_1.new_parameter(name='m', value=[64, 128], sweep=True)
sa_configuration_1.new_parameter(name='k', value=[64, 128], sweep=True)
sa_configuration_1.new_parameter(name='n', value=[64, 128], sweep=True)

sa_configuration_2.new_parameter(name='m', value=1024)
sa_configuration_2.new_parameter(name='k', value=1024)
sa_configuration_2.new_parameter(name='n', value=1024)

##############################################
############## Save to YAML  #################
##############################################

design.to_yaml()