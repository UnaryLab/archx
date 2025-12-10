from frontend.design import Design

design = Design(name='systolic_array', yaml_path='frontend_test/test_outputs/manual/')
architecture = design.new_architecture_description(name='mac_systolic_array')
event = design.new_event_description(name='mac_systolic_array', performance_path='systolic/performance/')
metric = design.new_metric_description(name='metric')
workload = design.new_workload_description(name='GEMM')
sa_configuration = workload.new_configuration(name='llama_2_7b')

##############################################
################ Architecture ################
##############################################

architecture.add_attributes(technology=32, frequency=400, interface='csv_cmos')
architecture.new_module(name='accumulator',              instance=[8, 8], tag=['pe'],  query={'class': 'adderbf16'})
architecture.new_module(name='multiplier',               instance=[8, 8], tag=['pe'],  query={'class': 'multiplierbf16'})
architecture.new_module(name='input_register',           instance=[8, 8], tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='weight_register',          instance=[8, 8], tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='partial_sum_register',     instance=[8, 8], tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='accumulator_register',     instance=[8, 8], tag=['pe'],  query={'class': 'register', 'width': 16})
architecture.new_module(name='row_accumulator',          instance=[8],    tag=['acc'], query={'class': 'adderbf16'})
architecture.new_module(name='row_accumulator_register', instance=[8],    tag=['acc'], query={'class': 'register', 'width': 16})

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

sa_configuration.new_parameters(m=64,
                                k=64,
                                n=64)

##############################################
############## Save to YAML  #################
##############################################

design.to_yaml()