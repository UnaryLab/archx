from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='fir_sc_32', yaml_path='agraph_casestudy/fir_sc_32/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

##############################################
################ Architecture ################
##############################################

bitwidth = [4, 6, 8, 10, 12, 14, 16]
#instance = [[256], [32]]
test_instance = 32

architecture.add_attributes(technology=10, frequency=50, interface='csv_sc')
architecture.new_module(name='mult', instance=[test_instance], tag=['tap', 'mac'], query={'class': 'sc_mult_fir'})
architecture.new_module(name='acc', instance=[test_instance], tag=['tap', 'mac'], query={'class': 'sc_balancer'})
architecture.new_module(name='shift_reg', instance=[test_instance], tag=['tap'], query={'class': 'sc_shift_reg'})
architecture.new_module(name='pnm', instance=[test_instance], tag=['tap'], query={'class': 'sc_pnm',  'width': bitwidth})
architecture.new_module(name='b2rc', instance=[test_instance], tag=['tap'], query={'class': 'sc_b2rc', 'width': 16})
architecture.new_module(name='control', instance=[1], tag=['tap'], query={'class': 'sc_fir_control'})
architecture.new_module(name='test', instance=[[1], [2], [3], [4], [5], [6], [7]], tag=['test'], query={'class': 'test'})

event.new_event(name='fir', subevent=['mult', 'acc', 'shift_reg', 'pnm', 'b2rc', 'control'], performance='agraph_casestudy/fir_sc_32/performance/performance.py')

metric.new_metric(name='area',           unit='jj',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specified')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

workload.new_configuration(name='fir')
workload.new_parameter(configuration='fir', parameter_name='mappings', parameter_value=1)
workload.new_parameter(configuration='fir', parameter_name='width', parameter_value=test_instance)
workload.new_parameter(configuration='fir', parameter_name='a_w', parameter_value=test_instance)
workload.new_parameter(configuration='fir', parameter_name='bitwidth', parameter_value=bitwidth, sweep=True)
workload.new_parameter(configuration='fir', parameter_name='bitwidth2', parameter_value=bitwidth, sweep=True)

# constraint_graph.add_constraint(pnm=['width'], b2rc=['width'], fir=['bitwidth', 'bitwidth2'])
constraint_graph.add_constraint(pnm=['width'], test=['instance'], fir=['bitwidth', 'bitwidth2'])

# constraint_graph.add_constraint(mult=['instance'], acc=['instance'], shift_reg=['instance'], pnm=['instance'], b2rc=['instance'], fir=['width', 'a_w'])
# constraint_graph.add_constraint(pnm=['width'], mult=['instance'], condition=lambda a, b: True)
# constraint_graph.add_constraint(pnm=['width'], b2rc=['width'])
constraint_graph.generate()