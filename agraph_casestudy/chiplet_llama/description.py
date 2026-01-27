from frontend.design import Design
from frontend.sweeping_functions import ConditionSweep, IterationSweep
from agraph_casestudy.common.metric import init_metric
design = Design(name='chiplet_llama', yaml_path='agraph_casestudy/chiplet_llama/description/')
constraint_graph = design.constraint_graph
architecture = design.architecture
event = design.event
metric = design.metric
workload = design.workload

architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

# ai chiplet
row_shapes = [[1, 1, 32], [1, 1, 64], [1, 1, 128], [1, 1, 256], [1, 1, 512], [2, 2, 64], [4, 4, 64], [1, 8, 64], [8, 8, 64]]
array_shapes = [[1, 1, 32, 32], [1, 1, 64, 64], [1, 1, 128, 128], [1, 1, 256, 26], [1, 1, 512, 512], [2, 2, 64, 64], [4, 4, 64, 64], [1, 8, 64, 64], [8, 8, 64, 64]]

architecture.new_module(name=['ififo', 'wfifo', 'ofifo'], instance=row_shapes, tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': 64})
architecture.new_module(name=['multiplier'], instance=array_shapes, tag=['pe', 'mac', 'array'], query={'class': 'multiplier_bfloat16'})
architecture.new_module(name=['adder'], instance=array_shapes, tag=['pe', 'mac', 'array'], query={'class': 'adder_bfloat16'})
architecture.new_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=array_shapes, tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
architecture.new_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=array_shapes, tag=['pe', 'data', 'array'], query={'class': 'register', 'width': 16})
architecture.new_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=array_shapes, tag=['pe', 'array'], query={'class': 'and_gate', 'width': 16})

# memory chiplet
architecture.new_module(name='sram', instance=[[1], [1], [1], [1], [1], [8], [14], [25], [29]], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': [32, 64, 128, 256, 512, 64, 64, 64, 64], 'width': 16, 'depth': 8192})

# accumulator chiplet
architecture.new_module(name='accumulator', instance=row_shapes, tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})
architecture.new_module(name='accumulator_reg', instance=row_shapes, tag=['output_register', 'array'], query={'class': 'register', 'width': 16})

# vector unit chiplet
architecture.new_module(name='vector_multiplier', instance=row_shapes, tag=['vector', 'array'], query={'class': 'multiplier_bfloat16'})
architecture.new_module(name='vector_adder', instance=row_shapes, tag=['vector', 'array'], query={'class': 'adder_bfloat16'})
architecture.new_module(name='vector_reg', instance=row_shapes, tag=['vector', 'array'], query={'class': 'register', 'width': 16})

event.new_event(name='gemm', subevent=['ai_chiplet', 'memory_chiplet', 'accumulator_chiplet', 'vector_unit_chiplet'], performance='agraph_casestudy/chiplet_performance/models/llama/performance.py')
event.new_event(name='ai_chiplet_compute', subevent=['ififo', 'wfifo', 'ofifo', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg', 'acc_en_reg'
                                                    'sum_en_reg', 'act_reg', 'sum_reg', 'act_mux', 'add_mux', 'sum_mux'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.new_event(name='ai_chiplet_weight', subevent=['weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg', 'weight_mux'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.new_event(name='memory_weight_reads', subevent=['sram'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.new_event(name='memory_compute_reads', subevent=['sram'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.new_event(name='memory_compute_writes', subevent=['sram'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.new_event(name='accumulator_chiplet', subevent=['accumulator', 'accumulator_reg'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.new_event(name='vector_unit_chiplet', subevent=['vector_multiplier', 'vector_adder', 'vector_reg'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')

init_metric(metric)
workload_sweep = [32, 64, 128, 256, 512, 64, 64, 64, 64]
workload.new_configuration(name='gemm')
workload.new_parameter(configuration='gemm', parameter_name='m', parameter_value=workload_sweep, sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='k', parameter_value=workload_sweep, sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='n', parameter_value=workload_sweep, sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='array_dim', parameter_value=workload_sweep, sweep=True)
workload.new_parameter(configuration='gemm', parameter_name='bitwidth', parameter_value=16, sweep=False)

constraint_graph.add_constraint(ififo=['instance'],
                                wfifo=['instance'],
                                ofifo=['instance'],
                                multiplier=['instance'],
                                adder=['instance'],
                                act_en_reg=['instance'],
                                mult_en_reg=['instance'],
                                acc_en_reg=['instance'],
                                weight_path_en_reg=['instance'],
                                weight_en_reg=['instance'],
                                sum_en_reg=['instance'],
                                act_reg=['instance'],
                                weight_path_reg=['instance'],
                                sum_reg=['instance'],
                                weight_reg=['instance'],
                                act_mux=['instance'],
                                weight_mux=['instance'],
                                add_mux=['instance'],
                                sum_mux=['instance'],
                                sram=['instance', 'bank'],
                                accumulator=['instance'],
                                accumulator_reg=['instance'],
                                vector_multiplier=['instance'],
                                vector_adder=['instance'],
                                vector_reg=['instance'],
                                gemm=['m', 'k', 'n', 'array_dim'])

constraint_graph.generate()