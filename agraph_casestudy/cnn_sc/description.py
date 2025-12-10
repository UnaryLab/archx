from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='cnn_sc', yaml_path='agraph_casestudy/cnn_sc/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################

architecture.add_attributes(technology=10, frequency=50, interface='csv_sc')

# PE

# architecture.new_module(name='mult', instance=[64, 64, 32], tag=['pe', 'mac'], query={'class': 'sc_mult_cnn'})
# architecture.new_module(name='acc', instance=[64, 64, 32], tag=['pe', 'mac'], query={'class': 'sc_accumulator'})
architecture.new_module(name='mac', instance=[64, 64, 32], tag=['array', 'pe', 'comp', 'mac'], query={'class': 'sc_mac'})
architecture.new_module(name='mac_splitter', instance=[64, 64, 32], tag=['array', 'comp', 'pe', 'mac_splitter'], query={'class': 'sc_splitter'})
architecture.new_module(name='weight_splitter', instance=[64, 64, 32], tag=['array', 'comp', 'pe', 'weight_splitter'], query={'class': 'sc_splitter'})
architecture.new_module(name='input_splitter', instance=[64, 64, 32], tag=['array', 'comp', 'pe', 'input_splitter'], query={'class': 'sc_splitter'})
architecture.new_module(name='nrdo', instance=[1, 1, 32], tag=['array', 'pe', 'nrdo'], query={'class': 'sc_nrdo'})
architecture.new_module(name='ptl', instance=[64, 64, 32], tag=['array', 'overhead'], query={'class': 'sc_ptl'})

event.new_event(name='vgg16', subevent=['mac', 'mac_splitter', 'input_splitter', 'weight_splitter', 'nrdo', 'ptl'], performance='agraph_casestudy/cnn_sc/performance/performance.py')
# event.new_event(name='vgg16', subevent=['convolution'], performance='agraph_casestudy/cnn_sc/performance/performance.py')
# event.new_event(name='convolution', subevent=['mult', 'mac_splitter', 'weight_splitter'], performance='agraph_casestudy/cnn_sc/performance/performance.py')
# event.new_event(name='fully_connected', subevent=['mult', 'mac_splitter', 'weight_splitter'], performance='agraph_casestudy/cnn_sc/performance/performance.py')

metric.new_metric(name='area',           unit='jj',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specified')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

workload.new_configuration(name='vgg16')
workload.new_parameter(configuration='vgg16', parameter_name='conv3-64', parameter_value=[3, 3, 64], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-64_of', parameter_value=[[224, 224, 64], [224, 224, 64]], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-128', parameter_value=[3, 3, 128], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-128_of', parameter_value=[[112, 112, 128], [112, 112, 128]], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-256', parameter_value=[3, 3, 256], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-256_of', parameter_value=[[56, 56, 256], [56, 56, 256], [56, 56, 256]], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-512', parameter_value=[3, 3, 512], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='conv3-512_of', parameter_value=[[28, 28, 512], [28, 28, 512], [28, 28, 512], [14, 14, 512], [14, 14, 512], [14, 14, 512]], sweep=False)
workload.new_parameter(configuration='vgg16', parameter_name='bitwidth', parameter_value=4, sweep=False)
#workload.new_parameter(configuration='vgg16', parameter_name='maxpool', parameter_value=[2, 2, 2], sweep=False)
# workload.new_parameter(configuration='vgg16', parameter_name='fc1', parameter_value=[25088, 4096], sweep=False)
# workload.new_parameter(configuration='vgg16', parameter_name='fc2', parameter_value=[4096, 4096], sweep=False)
# workload.new_parameter(configuration='vgg16', parameter_name='fc3', parameter_value=[4096, 1000], sweep=False)

constraint_graph.generate()