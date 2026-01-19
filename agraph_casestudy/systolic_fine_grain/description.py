from archx.programming.graph.agraph import AGraph
from agraph_casestudy.common.metric import init_metric
agraph = AGraph(path='agraph_casestudy/systolic_fine_grain/description/')
architecture = agraph.architecture
event = agraph.event
metric = agraph.metric
workload = agraph.workload

##############################################
################ Architecture ################
##############################################
architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

# SRAM
architecture.add_module(name='isram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})
architecture.add_module(name='wsram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})
architecture.add_module(name='osram', instance=[1], tag=['memory'], query={'class': 'sram', 'interface': 'cacti7', 'bank': 2, 'width': [32, 64, 128, 256], 'depth': [32, 64, 128, 256]})

# FIFO
architecture.add_module(name=['ififo', 'wfifo', 'ofifo'], instance=[[4], [8], [16], [32]], tag=['fifo', 'array'], query={'class': 'fifo', 'width': 16, 'depth': [4, 8, 16, 32]})

# Output Accumulator
architecture.add_module(name='output_adder', instance=[[4], [8], [16], [32]], tag=['output_adder', 'array'], query={'class': 'adder_bfloat16'})

gemm_list = [[4, 4], [8, 8], [16, 16], [32, 32]]
# operations
architecture.add_module(name=['multiplier'], instance=gemm_list, tag=['pe', 'mac', 'array'], query={'class': 'multiplier_bfloat16'})
architecture.add_module(name=['adder'], instance=gemm_list, tag=['pe', 'mac', 'array'], query={'class': 'adder_bfloat16'})

# data registers
architecture.add_module(name=['act_en_reg', 'mult_en_reg', 'acc_en_reg', 'weight_path_en_reg', 'weight_en_reg', 'sum_en_reg'], instance=gemm_list, tag=['pe', 'control', 'array'], query={'class': 'register', 'width': 1})
architecture.add_module(name=['act_reg', 'weight_path_reg', 'sum_reg', 'weight_reg'], instance=gemm_list, tag=['pe', 'data', 'array'], query={'class': 'register', 'width': 16})

architecture.add_module(name=['act_mux', 'weight_mux', 'add_mux', 'sum_mux'], instance=gemm_list, tag=['pe', 'array'], query={'class': 'and_gate', 'width': 16})

##############################################
###############    Event    ##################
##############################################
event.add_event(name='gemm', subevent=['input_reads', 'weight_reads', 'output_writes', 'input_array', 'weight_array'], performance='agraph_casestudy/systolic_fine_grain/performance/performance.py')
event.add_event(name='input_reads', subevent=['isram'], performance='agraph_casestudy/systolic_fine_grain/performance/performance.py')
event.add_event(name='weight_reads', subevent=['wsram'], performance='agraph_casestudy/systolic_fine_grain/performance/performance.py')
event.add_event(name='output_writes', subevent=['osram'], performance='agraph_casestudy/systolic_fine_grain/performance/performance.py')
event.add_event(name='input_array', subevent=['ififo', 'wfifo', 'ofifo', 'output_adder', 'multiplier', 'adder', 'act_en_reg', 'mult_en_reg', 'acc_en_reg', 'sum_en_reg', 'act_reg', 'sum_reg',
                                              'act_mux', 'weight_mux', 'add_mux', 'sum_mux'],
                performance='agraph_casestudy/systolic_fine_grain/performance/performance.py')
event.add_event(name='weight_array', subevent=['weight_path_en_reg', 'weight_en_reg', 'weight_path_reg', 'weight_reg'],
                performance='agraph_casestudy/systolic_fine_grain/performance/performance.py')

##############################################
###############    Metric    #################
##############################################
init_metric(metric)

##############################################
###############   Workload   #################
##############################################
workload.add_configuration(name='gemm')
workload.add_parameter(configuration='gemm', parameter_name='m', parameter_value=[16, 32, 64, 128], sweep=True)
workload.add_parameter(configuration='gemm', parameter_name='k', parameter_value=[16, 32, 64, 128], sweep=True)
workload.add_parameter(configuration='gemm', parameter_name='n', parameter_value=[16, 32, 64, 128], sweep=True)
workload.add_parameter(configuration='gemm', parameter_name='array_dim', parameter_value=[4, 8, 16, 32], sweep=True)
workload.add_parameter(configuration='gemm', parameter_name='bitwidth', parameter_value=16, sweep=False)


agraph.add_constraint(multiplier=['instance'],
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
                      ififo=['instance', 'depth'],
                      wfifo=['instance', 'depth'],
                      ofifo=['instance', 'depth'],
                      output_adder=['instance'],
                      isram=['width', 'depth'],
                      wsram=['width', 'depth'],
                      osram=['width', 'depth'],
                      act_mux=['instance'],
                      weight_mux=['instance'],
                      add_mux=['instance'],
                      sum_mux=['instance'],
                      gemm=['m', 'n', 'k', 'array_dim'])

agraph.generate()