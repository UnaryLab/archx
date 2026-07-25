from archx.programming.graph.agraph import AGraph

def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ################ Architecture ################
    ##############################################

    architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos')

    bitwidth = 16
    qbitwidth = 4
    array_width = 8
    broadcast_array = 8
    height_pipeline_tree = 1

    sram_width = 128
    sram_depth = 1024

    height_bitwidth = [32, 64, 128]
    array_height = [[8], [16], [32]]
    array = [[8, 8], [8, 16], [8, 32]]
    array_double = [[8, 8, 2], [8, 16, 2], [8, 32, 2]]
    height_tree = [[7], [15], [31]]

    vector_height = [[1], [2], [4]]
    
    # memory
    dram = architecture.add_module(name='dram', instance=[1], tag=['memory'], query={'interface': 'cacti7', 'class': 'dram', 'size': 8589934592, 'bandwidth': 128})
    isram = architecture.add_module(name='isram', instance=[1], tag=['memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': sram_width, 'depth': sram_depth})
    wsram = architecture.add_module(name='wsram', instance=[1], tag=['memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': sram_width, 'depth': sram_depth})
    osram = architecture.add_module(name='osram', instance=[1], tag=['memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': sram_width, 'depth': sram_depth})

    # per node
    imux = architecture.add_module(name='imux', instance=[1], tag=['array'], query={'class': 'mux', 'width': [array_width * bitwidth], 'inputs': 2})
    wmux = architecture.add_module(name='wmux', instance=[1], tag=['array'], query={'class': 'mux', 'width': height_bitwidth, 'inputs': 2})
    counter = architecture.add_module(name='counter', instance=[1], tag=['array'], query={'class': 'counter', 'width': 3})
    counter_reg = architecture.add_module(name='counter_register', instance=[1], tag=['array'], query={'class': 'register', 'width': 3})
    window_select = architecture.add_module(name='window_select', instance=[1], tag=['array'], query={'class': 'vlp_window'})

    # array width
    multiplier = architecture.add_module(name='multiplier', instance=[array_width], tag=['array'], query={'class': 'multiplier_bfloat16'})
    ififo = architecture.add_module(name='ififo', instance=[array_width], tag=['array'], query={'class': 'fifo', 'width': bitwidth, 'depth': array_width})

    # array / array double
    temporal_reg = architecture.add_module(name='temporal_register', instance=array, tag=['array'], query={'class': 'register', 'width': 1})
    and_gate = architecture.add_module(name='and_gate', instance=array, tag=['array'], query={'class': 'and_gate', 'width': 16})
    or_gate = architecture.add_module(name='or_gate', instance=array_double, tag=['array'], query={'class': 'or_gate', 'width': 16})

    # array height
    sign_fifo = architecture.add_module(name='sign_fifo', instance=array_height, tag=['array'], query={'class': 'fifo', 'width': 1, 'depth': 2})
    pe_fifo = architecture.add_module(name='pe_fifo', instance=array_height, tag=['array'], query={'class': 'fifo', 'width': 16, 'depth': array_width})
    sign_xor = architecture.add_module(name='sign_xor', instance=array_height, tag=['array'], query={'class': 'xor_bitwise', 'width': 1})
    adder = architecture.add_module(name='adder', instance=array_height, tag=['array'], query={'class': 'adder_bfloat16'})
    wfifo = architecture.add_module(name='wfifo', instance=array_height, tag=['array'], query={'class': 'fifo', 'width': qbitwidth, 'depth': 2})
    ofifo = architecture.add_module(name='ofifo', instance=array_height, tag=['array'], query={'class': 'fifo', 'width': bitwidth, 'depth': 2})
    exp_norm = architecture.add_module(name='exp_norm', instance=array_height, tag=['array'], query={'class': 'exp_norm'})
    exp_select = architecture.add_module(name='exp_select', instance=array_height, tag=['array'], query={'class': 'exp_select'})
    round = architecture.add_module(name='round', instance=array_height, tag=['array'], query={'class': 'vlp_round'})
    sign_mant_reg = architecture.add_module(name='sign_mantissa_register', instance=array_height, tag=['array'], query={'class': 'register', 'width': 4})
    exp_reg = architecture.add_module(name='exponent_register', instance=array_height, tag=['array'], query={'class': 'register', 'width': 8})
    exp_clamp = architecture.add_module(name='exp_clamp', instance=array_height, tag=['array'], query={'class': 'vlp_clamp'})
    comparator = architecture.add_module(name='comparator', instance=array_height, tag=['array'], query={'class': 'comparator', 'width': 4})
    mag_reg = architecture.add_module(name='magnitude_register', instance=array_height, tag=['array'], query={'class': 'register', 'width': 3})
    sign_reg = architecture.add_module(name='sign_register', instance=array_height, tag=['array'], query={'class': 'register', 'width': 1})

    # tree
    multiplier_reg = architecture.add_module(name='multiplier_register', instance=[broadcast_array], tag=['array'], query={'class': 'register', 'width': 16})
    min_max_tree = architecture.add_module(name='min_max_tree', instance=height_tree, tag=['array'], query={'class': 'vlp_max_min'})
    min_max_tree_reg = architecture.add_module(name='min_max_tree_register', instance=[height_pipeline_tree], tag=['array'], query={'class': 'register', 'width': 8})
    
    # vector
    multiplier_vector = architecture.add_module(name='multiplier_vector', instance=vector_height, tag=['array'], query={'class': 'multiplier_bfloat16'})
    register_vector = architecture.add_module(name='register_vector', instance=vector_height, tag=['array'], query={'class': 'register', 'width': 16})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='llama_2_7b', subevent=['gemm', 'nonlinear'], performance='zoo/agraph/designs/mugi/performance/llama.py')
    event.add_event(name='gemm', subevent=['projection', 'attention', 'ffn', 'output'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='nonlinear', subevent=['softmax', 'silu'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='projection', subevent=['proj_q', 'proj_k', 'proj_v', 'proj_a'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='attention', subevent=['qkt', 'av'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='ffn', subevent=['proj_up', 'proj_down', 'proj_gate'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='output', subevent=['output_prefill', 'output_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')

    event.add_event(name='proj_q', subevent=['proj_q_prefill', 'proj_q_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_k', subevent=['proj_k_prefill', 'proj_k_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_v', subevent=['proj_v_prefill', 'proj_v_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_a', subevent=['proj_a_prefill', 'proj_a_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='qkt', subevent=['qkt_prefill', 'qkt_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='av', subevent=['av_prefill', 'av_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_up', subevent=['proj_up_prefill', 'proj_up_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_down', subevent=['proj_down_prefill', 'proj_down_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_gate', subevent=['proj_gate_prefill', 'proj_gate_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='softmax', subevent=['softmax_prefill', 'softmax_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='silu', subevent=['silu_prefill', 'silu_decode'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')

    event.add_event(name='proj_q_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_q_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_k_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_k_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_v_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_v_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_a_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_a_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='qkt_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='qkt_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='av_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='av_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_up_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_up_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_down_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_down_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_gate_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='proj_gate_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='output_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='output_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                           'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                           'instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                                           'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='softmax_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                                'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                                'instruction', 'counter_reuse', 'vector', 'input_nonlinear', 'weight_nonlinear',
                                                'weight_reuse_nonlinear', 'array_nonlinear', 'array_fifo_nonlinear', 'summation', 'gemm_nonlinear'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='softmax_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                                'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                                'instruction', 'counter_reuse', 'vector', 'input_nonlinear', 'weight_nonlinear',
                                                'weight_reuse_nonlinear', 'array_nonlinear', 'array_fifo_nonlinear', 'summation', 'gemm_nonlinear'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='silu_prefill', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                                'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                                'instruction', 'counter_reuse', 'vector', 'input_nonlinear', 'weight_nonlinear',
                                                'weight_reuse_nonlinear', 'array_nonlinear', 'array_fifo_nonlinear', 'summation', 'gemm_nonlinear'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')
    event.add_event(name='silu_decode', subevent=['isram_offchip_writes', 'wsram_offchip_writes', 'osram_offchip_writes', 'osram_offchip_reads', 'isram_onchip_reads', 'wsram_onchip_reads',
                                                'osram_onchip_reads', 'osram_onchip_writes', 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads',
                                                'instruction', 'counter_reuse', 'vector', 'input_nonlinear', 'weight_nonlinear',
                                                'weight_reuse_nonlinear', 'array_nonlinear', 'array_fifo_nonlinear', 'summation', 'gemm_nonlinear'], performance='zoo/agraph/designs/mugi/performance/model_arch.py')

    
    event.add_event(name='instruction', subevent=['imux', 'wmux'], performance='zoo/agraph/designs/mugi/performance/common.py')
    event.add_event(name='counter_reuse', subevent=['counter', 'counter_register'], performance='zoo/agraph/designs/mugi/performance/common.py')
    event.add_event(name='vector', subevent=['multiplier_vector', 'register_vector'], performance='zoo/agraph/designs/mugi/performance/common.py')

    event.add_event(name='input_gemm', subevent=['ififo'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    event.add_event(name='input_reuse_gemm', subevent=['multiplier', 'multiplier_register'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    event.add_event(name='weight_gemm', subevent=['wfifo', 'magnitude_register', 'sign_register', 'sign_fifo'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    event.add_event(name='weight_reuse_gemm', subevent=['comparator'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    event.add_event(name='array_gemm', subevent=['temporal_register', 'and_gate', 'or_gate', 'sign_xor', 'adder', 'ofifo'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    event.add_event(name='array_fifo_gemm', subevent=['or_gate', 'pe_fifo'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    event.add_event(name='nonlinear_gemm', subevent=['round', 'sign_mantissa_register', 'exponent_register', 'exp_clamp', 'window_select', 'exp_norm', 'exp_select', 'min_max_tree', 'min_max_tree_register'], performance='zoo/agraph/designs/mugi/performance/gemm.py')
    
    event.add_event(name='input_nonlinear', subevent=['ififo'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')
    event.add_event(name='weight_nonlinear', subevent=['round', 'sign_mantissa_register', 'exponent_register', 'exp_clamp', 'window_select', 'wfifo',
               'min_max_tree', 'min_max_tree_register', 'magnitude_register', 'sign_register', 'exp_norm', 'exp_select', 'ofifo'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')
    event.add_event(name='weight_reuse_nonlinear', subevent=['comparator'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')
    event.add_event(name='array_nonlinear', subevent=['temporal_register', 'and_gate', 'or_gate'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')
    event.add_event(name='array_fifo_nonlinear', subevent=['or_gate', 'pe_fifo'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')
    event.add_event(name='summation', subevent=['adder'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')
    event.add_event(name='gemm_nonlinear', subevent=['multiplier', 'multiplier_register', 'sign_xor', 'sign_fifo'], performance='zoo/agraph/designs/mugi/performance/nonlinear.py')

    event.add_event(name='isram_offchip_writes', subevent=['isram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='wsram_offchip_writes', subevent=['wsram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='osram_offchip_writes', subevent=['osram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='osram_offchip_reads', subevent=['osram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='isram_onchip_reads', subevent=['isram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='wsram_onchip_reads', subevent=['wsram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='osram_onchip_reads', subevent=['osram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='osram_onchip_writes', subevent=['osram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='dram_input_reads', subevent=['dram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='dram_weight_reads', subevent=['dram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='dram_output_writes', subevent=['dram'], performance='zoo/agraph/designs/mugi/performance/memory.py')
    event.add_event(name='dram_output_reads', subevent=['dram'], performance='zoo/agraph/designs/mugi/performance/memory.py')

    # event.add_event(name='irouter_mapping', subevent=['irouter'], performance='zoo/agraph/designs/mugi/performance/router.py')
    # event.add_event(name='wrouter_mapping', subevent=['wrouter'], performance='zoo/agraph/designs/mugi/performance/router.py')
    # event.add_event(name='orouter_mapping', subevent=['orouter'], performance='zoo/agraph/designs/mugi/performance/router.py')

    ##############################################
    ###############    Metric    #################
    ##############################################
    metric.add_metric(name='area',           unit='mm^2',   aggregation='module')
    metric.add_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.add_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime',        unit='ms',     aggregation='specified')

    ##############################################
    ###############   Workload   #################
    ##############################################
    llama = workload.add_configuration(name='llama_2_7b')
    llama.add_parameter(parameter_name='dim', parameter_value=4096)
    llama.add_parameter(parameter_name='layers', parameter_value=32)
    llama.add_parameter(parameter_name='heads', parameter_value=32)
    llama.add_parameter(parameter_name='kv_heads', parameter_value=32)
    llama.add_parameter(parameter_name='hidden_dim', parameter_value=11008)
    llama.add_parameter(parameter_name='prefill_seq_len', parameter_value=64)
    llama.add_parameter(parameter_name='vocab_size', parameter_value=32000)
    llama.add_parameter(parameter_name='activation_bitwidth', parameter_value=16)
    llama.add_parameter(parameter_name='weight_bitwidth', parameter_value=4)
    llama.add_parameter(parameter_name='noc_stationary', parameter_value='os')
    llama.add_parameter(parameter_name='node_stationary', parameter_value='os')
    llama.add_parameter(parameter_name='max_seq_len', parameter_value=[1024, 2048, 4096], sweep=True)
    llama.add_parameter(parameter_name='batch_size', parameter_value=8)
    llama.add_parameter(parameter_name='kv_heads', parameter_value=8)
    llama.add_parameter(parameter_name='nonlinear_avg_early_termination_cycles', parameter_value=8)
    llama.add_parameter(parameter_name='default_avg_early_termination_cycles', parameter_value=8)
    llama.add_parameter(parameter_name='lut_height', parameter_value=8)
    llama.add_parameter(parameter_name='lut_width', parameter_value=12)
    llama.add_parameter(parameter_name='window_width', parameter_value=8)
    llama.add_parameter(parameter_name='cycles', parameter_value=8)
    llama.add_parameter(parameter_name='architecture', parameter_value='mugi')
    llama.add_parameter(parameter_name='noc_tile_m', parameter_value=256)
    llama.add_parameter(parameter_name='noc_tile_k', parameter_value=256)
    llama.add_parameter(parameter_name='noc_tile_n', parameter_value=256)
    llama.add_parameter(parameter_name='proj_avg_early_termination_cycles', parameter_value=8)
    llama.add_parameter(parameter_name='ffn_avg_early_termination_cycles', parameter_value=8)
    llama.add_parameter(parameter_name='k_avg_early_termination_cycles', parameter_value=8)
    llama.add_parameter(parameter_name='v_avg_early_termination_cycles', parameter_value=8)

    agraph.direct_constraint([
        temporal_reg['instance'],
        and_gate['instance'],
        or_gate['instance'],
        sign_fifo['instance'],
        sign_xor['instance'],
        pe_fifo['instance'],
        adder['instance'],
        wfifo['instance'],
        ofifo['instance'],
        exp_norm['instance'],
        exp_select['instance'],
        round['instance'],
        sign_mant_reg['instance'],
        exp_reg['instance'],
        exp_clamp['instance'],
        comparator['instance'],
        mag_reg['instance'],
        sign_reg['instance'],
        wmux['query']['width'],
        multiplier_vector['instance'],
        register_vector['instance'],
        min_max_tree['instance']
    ])

    return agraph.generate()