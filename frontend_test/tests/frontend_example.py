from frontend.design import Design
from frontend.constraint import ConstraintGraph

design = Design(name='mac_array')
constraint_graph = design.get_dependency_graph()
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')
architecture.new_module(name='sram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 4, 'width': [16, 32, 64], 'depth': [16, 8, 4]})

architecture.new_module(name='multiplier', instance=[[2, 2], [4, 4], [8, 8]], tag=['pe'], query={'class': 'multiplier'})

architecture.new_module(name='accumulator', instance=[2, 2], tag=['pe'], query={'class': 'accumulator'},
                        funct=lambda x: [x[0] * 2, x[1] * 2],
                        condition=lambda x: x[0] <= 8 and x[1] <= 8)

register_instance=[2, 2]
register_function = lambda x: [x[0] * 2, x[1] * 2]
register_iterations = 3
architecture.new_module(name='input_register', instance=register_iterations, tag=['pe'], query={'class': 'register', 'width': 16},
                        funct=register_function,
                        register_iterations=register_iterations)
architecture.new_module(name='weight_register', instance=register_iterations, tag=['pe'], query={'class': 'register', 'width': 16},
                        funct=register_function,
                        register_iterations=register_iterations)
architecture.new_module(name='output_register', instance=register_iterations, tag=['pe'], query={'class': 'register', 'width': 16},
                        funct=register_function,
                        register_iterations=register_iterations)

event.new_event(name='matrix_32x32', subevent=['multiply_accumulate'])
event.new_event(name='matrix_64x64', subevent=['multiply_accumulate'])
event.new_event(name='multiply_accumulate', subevent=['sram_read', 'sram_write', 'multiplier', 'accumulator',
                                                      'input_register', 'weight_register', 'output_register'])

metric.new_metric(name='area',           unit='mm^2',   aggregation='module')
metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specificied')
metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')

workload.new_configuration(name='matrix_32x32')
workload.new_configuration(name='matrix_64x64')

batch_size = 1
batch_size_function = lambda x: x * 2
batch_size_iterations = 5
workload.new_parameter(configuration='matrix_32x32', parameter_name='matrix_size', parameter_value=[32, 32], sweep=False)
workload.new_parameter(configuration='matrix_32x32', parameter_name='tile_size', parameter_value=[2, 4, 8], sweep=True)
workload.new_parameter(configuration='matrix_32x32', parameter_name='batch_size', parameter_value=batch_size,
                       funct=batch_size_function,
                       iterations=batch_size_iterations)
workload.new_parameter(configuration='matrix_64x64', parameter_name='matrix_size', parameter_value=[64, 64], sweep=False)
workload.new_parameter(configuration='matrix_64x64', parameter_name='tile_size', parameter_value=[4, 8, 16], sweep=True)
workload.new_parameter(configuration='matrix_64x64', parameter_name='batch_size', parameter_value=batch_size,
                       funct=batch_size_function,
                       iterations=batch_size_iterations)

constraint_graph.add_constraint(multiplier=['instance'],
                                accumulator=['instance'],
                                input_register= ['instance'],
                                weight_register=['instance'],
                                output_register=['instance'],
                                matrix_32x32=['tile_size'],
                                matrix_64x64=['tile_size'],
                                constraint_type='injection')

constraint_graph.add_constraint(a_parameters={'muliplier': 'instance',
                                              'accumulator': 'instance', 
                                              'input_register': 'instance',
                                              'weight_register': 'instance',
                                              'output_register': 'instance'},
                                b_parameters={'sram': 'width'},
                                constraint_type='injection',
                                condition=lambda a, b: a[0] * 8 == b)

constraint_graph.add_constraint(sram=['width', 'depth'],
                                constraint_type='injection',
                                condition=lambda a, b: 256 / a == b)

constraint_graph.add_constraint(a_parameters={'matrix_32x32': 'tile_size',
                                              'matrix_64x64': 'tile_size'},
                                b_parameters={'matrix_32x32': 'batch_size',
                                              'matrix_64x64': 'batch_size'},
                                constraint_type='exclusion',
                                condition=lambda a, b: a * b <= 8)

design.generate()