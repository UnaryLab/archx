from archx.programming.graph.agraph import AGraph
from agraph_casestudy.common.metric import init_metric
agraph = AGraph(path='agraph_casestudy/chiplet_model/description/')
architecture = agraph.architecture
event = agraph.event
metric = agraph.metric
workload = agraph.workload

architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

# ai chiplet
# row_shapes = [[1, 1, 32], [1, 1, 64], [1, 1, 128], [1, 1, 256], [1, 1, 512], [2, 2, 64], [4, 4, 64], [1, 8, 64], [8, 8, 64]]
# array_shapes = [[1, 1, 32, 32], [1, 1, 64, 64], [1, 1, 128, 128], [1, 1, 256, 256], [1, 1, 512, 512], [2, 2, 64, 64], [4, 4, 64, 64], [1, 8, 64, 64], [8, 8, 64, 64]]
# sram_inst = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [8, 6], [14, 7], [25, 8], [29, 9]]
# sram_bank = [[32, 1], [64, 1], [128, 1], [256, 1], [512, 1], [64, 2], [64, 3], [64, 4], [64, 5]]
# workload_sweep = [[32, 1], [64, 1], [128, 1], [256, 1], [512, 1], [64, 2], [64, 3], [64, 4], [64, 5]]

row_shapes = [[1, 1, 32], [1, 1, 64], [1, 1, 128], [1, 1, 256], [1, 1, 512], [2, 2, 64], [4, 4, 64], [1, 8, 64], [8, 8, 64]]
array_shapes = [[1, 1, 32, 32], [1, 1, 64, 64], [1, 1, 128, 128], [1, 1, 256, 256], [1, 1, 512, 512], [2, 2, 64, 64], [4, 4, 64, 64], [1, 8, 64, 64], [8, 8, 64, 64]]
sram_inst = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [8, 6], [14, 7], [25, 8], [29, 9]]
sram_bank = [[32, 1], [64, 1], [128, 1], [256, 1], [512, 1], [64, 2], [64, 3], [64, 4], [64, 5]]
workload_sweep = [32, 64, 128, 256, 512, 65, 66, 67, 68]

architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=row_shapes, tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': 64})
architecture.add_module(name=['multiplier'], instance=array_shapes, tag=['pe', 'mac', 'array'], query={'class': 'multiplier_bfloat16'})
architecture.add_module(name=['adder'], instance=array_shapes, tag=['pe', 'mac', 'array'], query={'class': 'adder_bfloat16'})
architecture.add_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=array_shapes, tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
architecture.add_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=array_shapes, tag=['pe', 'data', 'array'], query={'class': 'register', 'width': 16})
architecture.add_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=array_shapes, tag=['pe', 'array'], query={'class': 'and_gate', 'width': 16})

# memory chiplet
architecture.add_module(name='sram', instance=sram_inst, tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': sram_bank, 'width': 16, 'depth': 8192})

# accumulator chiplet
architecture.add_module(name='accumulator', instance=row_shapes, tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})
architecture.add_module(name='accumulator_reg', instance=row_shapes, tag=['output_register', 'array'], query={'class': 'register', 'width': 16})

# vector unit chiplet
architecture.add_module(name='vector_multiplier', instance=row_shapes, tag=['vector', 'array'], query={'class': 'multiplier_bfloat16'})
architecture.add_module(name='vector_adder', instance=row_shapes, tag=['vector', 'array'], query={'class': 'adder_bfloat16'})
architecture.add_module(name='vector_reg', instance=row_shapes, tag=['vector', 'array'], query={'class': 'register', 'width': 16})

event.add_event(name='gemm', subevent=['ai_chiplet', 'memory_chiplet', 'accumulator_chiplet', 'vector_unit_chiplet'], performance='agraph_casestudy/chiplet_performance/models/llama/performance.py')
event.add_event(name='ai_chiplet_compute', subevent=['ififo', 'wfifo', 'ofifo', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg', 'acc_en_reg'
                                                    'sum_en_reg', 'act_reg', 'sum_reg', 'act_mux', 'add_mux', 'sum_mux'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.add_event(name='ai_chiplet_weight', subevent=['weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg', 'weight_mux'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.add_event(name='memory_weight_reads', subevent=['sram'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.add_event(name='memory_compute_reads', subevent=['sram'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.add_event(name='memory_compute_writes', subevent=['sram'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.add_event(name='accumulator_chiplet', subevent=['accumulator', 'accumulator_reg'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')
event.add_event(name='vector_unit_chiplet', subevent=['vector_multiplier', 'vector_adder', 'vector_reg'],
                performance='agraph_casestudy/chiplet_performance/chiplets/performance.py')

init_metric(metric)

workload.add_configuration(name='gemm')
workload.add_parameter(configuration='gemm', parameter_name='array_dim', parameter_value=workload_sweep, sweep=True)
workload.add_parameter(configuration='gemm', parameter_name='bitwidth', parameter_value=16, sweep=False)

workload.add_configuration(name='alexnet')
workload.add_parameter(configuration='alexnet', parameter_name='image_size', parameter_value=[3, 224, 224], sweep=False)
workload.add_parameter(configuration='alexnet', parameter_name='conv_0', parameter_value=[11, 11, 4], sweep=False)
workload.add_parameter(configuration='alexnet', parameter_name='act_0', parameter_value='relu', sweep=False)


agraph.add_constraint(ififo=['instance'],
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
                      gemm=['array_dim'])

agraph.generate()