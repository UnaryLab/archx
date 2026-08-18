from archx.programming.graph.agraph import AGraph
from math import log2

def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ###############   Workload   #################
    ##############################################
    llama = workload.add_configuration(name=['llama_2_7b', 'llama_2_13b', 'llama_2_70b'])
    llama_configs = list(llama.values())

    # per-model fixed dimensions
    llama['llama_2_7b'].add_parameter(parameter_name='dim',        parameter_value=4096)
    llama['llama_2_7b'].add_parameter(parameter_name='layers',     parameter_value=32)
    llama['llama_2_7b'].add_parameter(parameter_name='heads',      parameter_value=32)
    llama['llama_2_7b'].add_parameter(parameter_name='hidden_dim', parameter_value=11008)
    llama['llama_2_7b'].add_parameter(parameter_name='kv_heads',   parameter_value=32)

    llama['llama_2_13b'].add_parameter(parameter_name='dim',        parameter_value=5120)
    llama['llama_2_13b'].add_parameter(parameter_name='layers',     parameter_value=40)
    llama['llama_2_13b'].add_parameter(parameter_name='heads',      parameter_value=40)
    llama['llama_2_13b'].add_parameter(parameter_name='hidden_dim', parameter_value=13824)
    llama['llama_2_13b'].add_parameter(parameter_name='kv_heads',   parameter_value=40)

    llama['llama_2_70b'].add_parameter(parameter_name='dim',        parameter_value=8192)
    llama['llama_2_70b'].add_parameter(parameter_name='layers',     parameter_value=80)
    llama['llama_2_70b'].add_parameter(parameter_name='heads',      parameter_value=64)
    llama['llama_2_70b'].add_parameter(parameter_name='hidden_dim', parameter_value=28672)
    kv_heads = llama['llama_2_70b'].add_parameter(parameter_name='kv_heads',   parameter_value=[8, 16, 32, 64], sweep=True)

    # shared swept parameters
    max_seq_len = workload.add_parameters(llama_configs, parameter_name='max_seq_len', parameter_value=[128, 256, 512, 1024, 2048, 4096], sweep=True)
    batch_size  = workload.add_parameters(llama_configs, parameter_name='batch_size',  parameter_value=[1, 2, 4, 8, 16, 32],            sweep=True)
    subarch     = workload.add_parameters(llama_configs, parameter_name='subarch',     parameter_value=['lut', 'vlp'],                  sweep=True)

    # shared constant parameters (workload configuration + full_termination architecture set)
    workload.add_parameters(llama_configs, parameter_name='prefill_seq_len',                      parameter_value=64)
    workload.add_parameters(llama_configs, parameter_name='vocab_size',                           parameter_value=32000)
    workload.add_parameters(llama_configs, parameter_name='activation_bitwidth',                  parameter_value=16)
    workload.add_parameters(llama_configs, parameter_name='weight_bitwidth',                      parameter_value=4)
    workload.add_parameters(llama_configs, parameter_name='noc_stationary',                       parameter_value='os')
    workload.add_parameters(llama_configs, parameter_name='node_stationary',                      parameter_value='os')
    workload.add_parameters(llama_configs, parameter_name='nonlinear_avg_early_termination_cycles', parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='default_avg_early_termination_cycles', parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='lut_height',                           parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='lut_width',                            parameter_value=12)
    workload.add_parameters(llama_configs, parameter_name='window_width',                         parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='cycles',                               parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_m',                           parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_k',                           parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_n',                           parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='architecture',                        parameter_value='mugi')
    workload.add_parameters(llama_configs, parameter_name='proj_avg_early_termination_cycles',    parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='ffn_avg_early_termination_cycles',     parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='k_avg_early_termination_cycles',       parameter_value=8)
    workload.add_parameters(llama_configs, parameter_name='v_avg_early_termination_cycles',       parameter_value=8)

    ##############################################
    ###############    Event    ##################
    ##############################################
    # Performance model paths
    common_hw = 'zoo/llm/designs/mugi/performance/common.performance.py'
    gemm      = 'zoo/llm/designs/mugi/performance/gemm.performance.py'
    nonlinear = 'zoo/llm/designs/mugi/performance/nonlinear.performance.py'
    llama     = 'zoo/llm/common/performance/model/llama.performance.py'
    model     = 'zoo/llm/common/performance/model/model_architecture.performance.py'
    memory    = 'zoo/llm/common/performance/memory/memory.performance.py'
    router    = 'zoo/llm/common/performance/router/router.performance.py'

    # Per-subarchitecture module groups (event sweep candidates ordered lut, vlp)
    nonlinear_preprocess  = ['round', 'sign_mantissa_register', 'exponent_register', 'exp_clamp']
    nonlinear_postprocess = ['exp_norm', 'exp_select', 'max_tree', 'max_tree_register']
    nonlinear_gemm_lut = nonlinear_preprocess + ['lut_register', 'lut_decoder'] + nonlinear_postprocess
    nonlinear_gemm_vlp = nonlinear_preprocess + ['window_select']               + nonlinear_postprocess
    weight_nl_lut = nonlinear_preprocess + ['lut_register', 'lut_decoder'] + ['wfifo', 'max_tree', 'max_tree_register',
                    'magnitude_register', 'sign_register', 'exp_norm', 'exp_select', 'ofifo']
    weight_nl_vlp = nonlinear_preprocess + ['window_select']               + ['wfifo', 'max_tree', 'max_tree_register',
                    'magnitude_register', 'sign_register', 'exp_norm', 'exp_select', 'ofifo']

    hw_gemm   = ['instruction', 'input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm',
                 'weight_reuse_gemm', 'array_gemm', 'array_fifo_gemm', 'nonlinear_gemm', 'vector']
    hw_nl     = ['instruction', 'input_nonlinear', 'counter_reuse', 'weight_nonlinear',
                 'weight_reuse_nonlinear', 'array_nonlinear', 'array_fifo_nonlinear', 'summation',
                 'gemm_nonlinear', 'vector']
    mem       = ['isram_offchip_writes', 'isram_onchip_reads',
                 'wsram_offchip_writes', 'wsram_onchip_reads',
                 'osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_writes', 'osram_onchip_reads',
                 'dram_input_reads', 'dram_weight_reads', 'dram_output_reads', 'dram_output_writes']
    router_ev = ['irouter_mapping', 'wrouter_mapping', 'orouter_mapping']

    # single-node event graph maps leaves to hardware + memory; multi-node adds the routers
    gemm_leaf      = hw_gemm + mem
    nonlinear_leaf = hw_nl + mem

    # software tree (4 model roots collapse into one llama_2 root)
    event.add_event(name='llama_2',    subevent=['gemm', 'nonlinear'],                        performance=llama)
    event.add_event(name='gemm',       subevent=['projection', 'attention', 'ffn', 'output'], performance=model)
    event.add_event(name='nonlinear',  subevent=['softmax', 'silu'],                          performance=model)
    event.add_event(name='projection', subevent=['proj_q', 'proj_k', 'proj_v', 'proj_a'],     performance=model)
    event.add_event(name='attention',  subevent=['qkt', 'av'],                                performance=model)
    event.add_event(name='ffn',        subevent=['proj_up', 'proj_down', 'proj_gate'],        performance=model)

    # each middle layer decomposes into prefill/decode
    gemm_layers = ['proj_q', 'proj_k', 'proj_v', 'proj_a', 'qkt', 'av', 'proj_up', 'proj_down', 'proj_gate', 'output']
    nonlinear_layers = ['softmax', 'silu']
    layer_dict = {k: [k + '_prefill', k + '_decode'] for k in gemm_layers + nonlinear_layers}
    event.add_event(event_dict=layer_dict, performance=model)

    # each leaf sweeps between the single-node and multi-node event graphs
    gemm_leaves      = [layer + suffix for layer in gemm_layers      for suffix in ['_prefill', '_decode']]
    nonlinear_leaves = [layer + suffix for layer in nonlinear_layers for suffix in ['_prefill', '_decode']]
    gemm_events      = event.add_event(name=gemm_leaves,      subevent=[gemm_leaf, gemm_leaf + router_ev],           performance=model)
    nonlinear_events = event.add_event(name=nonlinear_leaves, subevent=[nonlinear_leaf, nonlinear_leaf + router_ev], performance=model)

    # hardware events (nonlinear_gemm / weight_nonlinear sweep per subarchitecture: lut, vlp)
    event.add_event(name='instruction',    subevent=['imux', 'wmux'],                          performance=common_hw)
    event.add_event(name='counter_reuse',  subevent=['counter', 'counter_register'],           performance=common_hw)
    event.add_event(name='vector',         subevent=['multiplier_vector', 'register_vector'],  performance=common_hw)

    event.add_event(name='input_gemm',        subevent=['ififo'],                              performance=gemm)
    event.add_event(name='input_reuse_gemm',  subevent=['multiplier', 'multiplier_register'],  performance=gemm)
    event.add_event(name='weight_gemm',       subevent=['wfifo', 'magnitude_register', 'sign_register', 'sign_fifo'], performance=gemm)
    event.add_event(name='weight_reuse_gemm', subevent=['comparator'],                         performance=gemm)
    event.add_event(name='array_gemm',        subevent=['temporal_register', 'and_gate', 'or_gate', 'sign_xor', 'adder', 'ofifo'], performance=gemm)
    event.add_event(name='array_fifo_gemm',   subevent=['or_gate', 'pe_fifo'],                 performance=gemm)
    nonlinear_gemm = event.add_event(name='nonlinear_gemm', subevent=[nonlinear_gemm_lut, nonlinear_gemm_vlp], performance=gemm)

    event.add_event(name='input_nonlinear',        subevent=['ififo'],                         performance=nonlinear)
    weight_nonlinear = event.add_event(name='weight_nonlinear', subevent=[weight_nl_lut, weight_nl_vlp],      performance=nonlinear)
    event.add_event(name='weight_reuse_nonlinear', subevent=['comparator'],                    performance=nonlinear)
    event.add_event(name='array_nonlinear',        subevent=['temporal_register', 'and_gate', 'or_gate'], performance=nonlinear)
    event.add_event(name='array_fifo_nonlinear',   subevent=['or_gate', 'pe_fifo'],            performance=nonlinear)
    event.add_event(name='summation',              subevent=['adder'],                         performance=nonlinear)
    event.add_event(name='gemm_nonlinear',         subevent=['multiplier', 'multiplier_register', 'sign_xor', 'sign_fifo'], performance=nonlinear)

    # memory events (grouped by backing module)
    event.add_event(name=['isram_offchip_writes', 'isram_onchip_reads'], subevent=['isram'], performance=memory)
    event.add_event(name=['wsram_offchip_writes', 'wsram_onchip_reads'], subevent=['wsram'], performance=memory)
    event.add_event(name=['osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_writes', 'osram_onchip_reads'], subevent=['osram'], performance=memory)
    event.add_event(name=['dram_input_reads', 'dram_weight_reads', 'dram_output_reads', 'dram_output_writes'],         subevent=['dram'],  performance=memory)

    # router events
    event.add_event(name='irouter_mapping', subevent=['irouter'], performance=router)
    event.add_event(name='wrouter_mapping', subevent=['wrouter'], performance=router)
    event.add_event(name='orouter_mapping', subevent=['orouter'], performance=router)

    # ============================================================
    # Architecture description
    # ============================================================
    architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos_asplos_2026_ae')

    arch_inst = {}
    # === Static parameters ======================================
    # Array parameters
    width = 8

    arch_inst['single'] = [1]
    arch_inst['width']  = [width]

    # Memory parameters
    sram_size = 2**17                                 # per-bank-set capacity (width * depth)
    min_sram_width, max_sram_width = 128, 256

    # === Sweep parameters =======================================
    # NoC parameters
    node_inst = [[1, 1], [4, 4], [8, 8]]              # 1x1 (single-node), then two multi-node meshes

    # Array parameters (derived from the height sweep)
    min_height, max_height = 32, 256

    arch_inst['height']               = [[2**h] for h in range(int(log2(min_height)), int(log2(max_height)) + 1)]
    arch_inst['array']                = [[h[0], width]              for h in arch_inst['height']]
    arch_inst['array_double']         = [[h[0], width, 2]           for h in arch_inst['height']]
    arch_inst['broadcast_array']      = [[(h[0] // 128) * 2, width] for h in arch_inst['height']]
    arch_inst['broadcast_height']     = [[(h[0] // 128) * 2 + 1]    for h in arch_inst['height']]
    arch_inst['height_pipeline_tree'] = [[(h[0] // 128) * 2 + 1]    for h in arch_inst['height']]
    arch_inst['height_tree']          = [[h[0] - 1]                 for h in arch_inst['height']]
    arch_inst['vector']               = [[h[0] // 8]                for h in arch_inst['height']]
    # subarchitecture modules, built ungated (the event graph selects lut vs vlp)
    arch_inst['lut_register']         = [[h[0] // 8, 2, 8, 12]      for h in arch_inst['height']]
    arch_inst['lut_decoder']          = [[h[0] // 8]                for h in arch_inst['height']]
    arch_inst['window_select']        = [[1]                        for h in arch_inst['height']]

    # Memory parameters (derived from the height sweep)
    wsram_width = [min(max_sram_width, max(min_sram_width, h[0])) for h in arch_inst['height']]
    osram_width = [max(min_sram_width, h[0] * 2)                  for h in arch_inst['height']]
    hbw         = [h[0] * 4                                       for h in arch_inst['height']]
    wsram_depth = [sram_size // w for w in wsram_width]
    osram_depth = [sram_size // w for w in osram_width]

    # expand every parameter to node instances
    for param, values in arch_inst.items():
        if isinstance(values[0], list):
            arch_inst[param] = [node + value for node in node_inst for value in values]
        else:
            arch_inst[param] = [node + values for node in node_inst]

    # tag definitions (from configuration.yaml)
    node_memory_tag = ['onchip', 'node_memory', 'memory']
    fifo_tag        = ['onchip', 'array', 'fifo']
    control_tag     = ['onchip', 'array', 'control']
    nonlinear_tag   = ['onchip', 'array', 'nonlinear']
    vector_tag      = ['onchip', 'array', 'vector']
    router_tag      = ['onchip', 'router']
    accumulator_tag = ['onchip', 'array', 'accumulator']
    value_reuse_tag = ['onchip', 'array', 'value_reuse']
    tc_tag          = ['onchip', 'array', 'tc']
    pe_tag          = ['onchip', 'array', 'pe']

    # === Memory modules =========================================
    # DRAM
    architecture.add_module(name='dram', instance=[1], tag=['memory', 'dram'], query={'interface': 'cacti7', 'class': 'dram', 'size': 8589934592, 'bandwidth': 128})

    # SRAMs (isram fixed; wsram/osram widths/depths swept)
    isram = architecture.add_module(name='isram', instance=arch_inst['single'], tag=node_memory_tag, query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': 128, 'depth': 1024})
    wsram = architecture.add_module(name='wsram', instance=arch_inst['single'], tag=node_memory_tag, query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': wsram_width, 'depth': wsram_depth})
    osram = architecture.add_module(name='osram', instance=arch_inst['single'], tag=node_memory_tag, query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': osram_width, 'depth': osram_depth})

    # === Fifo modules ===========================================
    ififo = architecture.add_module(name='ififo', instance=arch_inst['width'],  tag=fifo_tag, query={'class': 'fifo', 'width': 16, 'depth': 8})
    wfifo = architecture.add_module(name='wfifo', instance=arch_inst['height'], tag=fifo_tag, query={'class': 'fifo', 'width': 4,  'depth': 2})
    ofifo = architecture.add_module(name='ofifo', instance=arch_inst['height'], tag=fifo_tag, query={'class': 'fifo', 'width': 16, 'depth': 2})

    # ---- tc ----
    comparator         = architecture.add_module(name='comparator',         instance=arch_inst['height'], tag=tc_tag, query={'class': 'comparator', 'width': 4})
    magnitude_register = architecture.add_module(name='magnitude_register', instance=arch_inst['height'], tag=tc_tag, query={'class': 'register', 'width': 3})
    sign_register      = architecture.add_module(name='sign_register',      instance=arch_inst['height'], tag=tc_tag, query={'class': 'register', 'width': 1})

    # ---- pe ----
    temporal_register = architecture.add_module(name='temporal_register', instance=arch_inst['array'],        tag=pe_tag, query={'class': 'register', 'width': 1})
    and_gate          = architecture.add_module(name='and_gate',          instance=arch_inst['array'],        tag=pe_tag, query={'class': 'and_bitwise', 'width': 16})
    or_gate           = architecture.add_module(name='or_gate',           instance=arch_inst['array_double'], tag=pe_tag, query={'class': 'or_bitwise', 'width': 16})
    sign_xor          = architecture.add_module(name='sign_xor',          instance=arch_inst['height'],       tag=pe_tag, query={'class': 'xor_bitwise', 'width': 1})

    # ---- array (value reuse) ----
    multiplier          = architecture.add_module(name='multiplier',          instance=arch_inst['width'],           tag=value_reuse_tag, query={'class': 'multiplierbf16'})
    multiplier_register = architecture.add_module(name='multiplier_register', instance=arch_inst['broadcast_array'], tag=value_reuse_tag, query={'class': 'register', 'width': 16})
    sign_fifo           = architecture.add_module(name='sign_fifo',           instance=arch_inst['height'],          tag=fifo_tag,        query={'class': 'fifo', 'width': 1, 'depth': 2})
    pe_fifo             = architecture.add_module(name='pe_fifo',             instance=arch_inst['height'],          tag=fifo_tag,        query={'class': 'fifo', 'width': 16, 'depth': 8})
    adder               = architecture.add_module(name='adder',               instance=arch_inst['height'],          tag=accumulator_tag, query={'class': 'adderbf16'})

    # ---- control ----
    counter          = architecture.add_module(name='counter',          instance=arch_inst['single'],                 tag=value_reuse_tag, query={'class': 'counter', 'width': 3})
    counter_register = architecture.add_module(name='counter_register', instance=arch_inst['broadcast_height'],     tag=value_reuse_tag, query={'class': 'register', 'width': 3})
    max_tree         = architecture.add_module(name='max_tree',         instance=arch_inst['height_tree'],          tag=nonlinear_tag,   query={'class': 'vlp_max_min'})
    max_tree_register = architecture.add_module(name='max_tree_register', instance=arch_inst['height_pipeline_tree'], tag=nonlinear_tag, query={'class': 'register', 'width': 8})
    imux             = architecture.add_module(name='imux',             instance=arch_inst['single'],                 tag=control_tag,     query={'class': 'multiplexer', 'width': 128, 'num_inputs': 2})
    wmux             = architecture.add_module(name='wmux',             instance=arch_inst['single'],                 tag=control_tag,     query={'class': 'multiplexer', 'width': hbw, 'num_inputs': 2})

    # ---- preprocess ----
    round_                 = architecture.add_module(name='round',                  instance=arch_inst['height'], tag=nonlinear_tag, query={'class': 'vlp_round'})
    sign_mantissa_register = architecture.add_module(name='sign_mantissa_register', instance=arch_inst['height'], tag=nonlinear_tag, query={'class': 'register', 'width': 4})
    exponent_register      = architecture.add_module(name='exponent_register',      instance=arch_inst['height'], tag=nonlinear_tag, query={'class': 'register', 'width': 8})
    exp_clamp              = architecture.add_module(name='exp_clamp',              instance=arch_inst['height'], tag=nonlinear_tag, query={'class': 'vlp_clamp'})

    # ---- postprocess ----
    exp_norm   = architecture.add_module(name='exp_norm',   instance=arch_inst['height'], tag=nonlinear_tag, query={'class': 'exp_norm'})
    exp_select = architecture.add_module(name='exp_select', instance=arch_inst['height'], tag=nonlinear_tag, query={'class': 'exp_select'})

    # ---- scaling vector ----
    multiplier_vector = architecture.add_module(name='multiplier_vector', instance=arch_inst['vector'], tag=vector_tag, query={'class': 'multiplierbf16'})
    register_vector   = architecture.add_module(name='register_vector',   instance=arch_inst['vector'], tag=vector_tag, query={'class': 'register', 'width': 16})

    # ---- routers (always present; instance = node + [1], single-node collapses to 1) ----
    irouter = architecture.add_module(name='irouter', instance=arch_inst['single'], tag=router_tag, query={'class': 'noc_router', 'bandwidth': 64})
    wrouter = architecture.add_module(name='wrouter', instance=arch_inst['single'], tag=router_tag, query={'class': 'noc_router', 'bandwidth': 64})
    orouter = architecture.add_module(name='orouter', instance=arch_inst['single'], tag=router_tag, query={'class': 'noc_router', 'bandwidth': 64})

    # ---- subarchitecture modules (costed only when their event graph references them) ----
    lut_register  = architecture.add_module(name='lut_register',  instance=arch_inst['lut_register'],  tag=nonlinear_tag, query={'class': 'register', 'width': 16})
    lut_decoder   = architecture.add_module(name='lut_decoder',   instance=arch_inst['lut_decoder'],   tag=nonlinear_tag, query={'class': 'decoder'})
    window_select = architecture.add_module(name='window_select', instance=arch_inst['window_select'], tag=nonlinear_tag, query={'class': 'vlp_window'})

    ##############################################
    ###############    Metric    #################
    ##############################################
    metric.add_metric(name='area', unit='mm^2', aggregation='module')
    metric.add_metric(name='leakage_power', unit='mW', aggregation='module')
    metric.add_metric(name='dynamic_energy', unit='nJ', aggregation='summation')
    metric.add_metric(name='cycle_count', unit='cycles', aggregation='specified')
    metric.add_metric(name='runtime', unit='ms', aggregation='specified')

    ##############################################
    ###############  Constraints  ################
    ##############################################
    # Master sweep: every height-derived instance (node x height) locked to one index.
    agraph.direct_constraint([
        wfifo['instance'],
        ofifo['instance'],
        comparator['instance'],
        magnitude_register['instance'],
        sign_register['instance'],
        temporal_register['instance'],
        and_gate['instance'],
        or_gate['instance'],
        sign_xor['instance'],
        multiplier_register['instance'],
        sign_fifo['instance'],
        pe_fifo['instance'],
        adder['instance'],
        counter_register['instance'],
        max_tree['instance'],
        max_tree_register['instance'],
        round_['instance'],
        sign_mantissa_register['instance'],
        exponent_register['instance'],
        exp_clamp['instance'],
        exp_norm['instance'],
        exp_select['instance'],
        multiplier_vector['instance'],
        register_vector['instance'],
        lut_register['instance'],
        lut_decoder['instance'],
        window_select['instance'],
    ])
    # Node-only instances tied within their own group (one instance per node mesh).
    agraph.direct_constraint([
        isram['instance'], wsram['instance'], osram['instance'],
        counter['instance'], imux['instance'], wmux['instance'],
        irouter['instance'], wrouter['instance'], orouter['instance'],
    ])
    agraph.direct_constraint([ififo['instance'], multiplier['instance']])
    # Join the node-only groups to the master sweep by matching node mesh coordinates.
    agraph.conditional_constraint(single=isram['instance'], height=adder['instance'],
                                  condition=lambda single, height: single[0:2] == height[0:2])
    agraph.conditional_constraint(width=ififo['instance'], height=adder['instance'],
                                  condition=lambda width, height: width[0:2] == height[0:2])
    # Tie the height-indexed SRAM/mux query values to the master sweep.
    agraph.direct_constraint_partition(wsram['query']['width'], adder['instance'])
    agraph.direct_constraint_partition(wsram['query']['depth'], adder['instance'])
    agraph.direct_constraint_partition(osram['query']['width'], adder['instance'])
    agraph.direct_constraint_partition(osram['query']['depth'], adder['instance'])
    agraph.direct_constraint_partition(wmux['query']['width'],  adder['instance'])

    # subarchitecture: hardware events and the workload flag sweep together (lut, vlp)
    subarch_params = [p['parameter'] for p in subarch.values()]
    agraph.direct_constraint([
        nonlinear_gemm['subevent'],
        weight_nonlinear['subevent']
    ] + subarch_params)

    # network: every software leaf picks the same event graph (single-node or multi-node)
    leaf_events = list(gemm_events.values()) + list(nonlinear_events.values())
    agraph.direct_constraint([e['subevent'] for e in leaf_events])

    # routers are mapped if and only if the design is multi-node
    agraph.conditional_constraint(
        m = adder['instance'],
        ev = gemm_events['proj_q_prefill']['subevent'],
        condition = lambda m, ev: ('irouter_mapping' in ev) == (m[0] != 1)
    )

    # batch size sweeps on every single-node design; multi-node fixes batch size to 8
    agraph.conditional_constraint(
        m = adder['instance'],
        b = list(batch_size.values())[0]['parameter'],
        condition = lambda m, b: (m[0] == 1) or (b == 8)
    )

    # seq len sweeps on every single-node design; multi-node fixes seq len to 4096
    agraph.conditional_constraint(
        m = adder['instance'],
        q = list(max_seq_len.values())[0]['parameter'],
        condition = lambda m, q: (m[0] == 1) or (q == 4096)
    )

    return agraph.generate()
