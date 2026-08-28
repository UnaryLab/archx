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

    # shared swept parameters (tied across all model configs in the Constraints section)
    max_seq_len = workload.add_parameters(llama_configs, parameter_name='max_seq_len', parameter_value=[128, 256, 512, 1024, 2048, 4096], sweep=True)
    batch_size  = workload.add_parameters(llama_configs, parameter_name='batch_size',  parameter_value=[1, 2, 4, 8, 16, 32],            sweep=True)
    subarch     = workload.add_parameters(llama_configs, parameter_name='subarch',     parameter_value=['mac', 'figna', 'pwl', 'taylor'], sweep=True)

    # shared constant parameters
    # model parameters
    workload.add_parameters(llama_configs, parameter_name='prefill_seq_len',             parameter_value=64)
    workload.add_parameters(llama_configs, parameter_name='vocab_size',                  parameter_value=32000)

    # Noc configuration
    workload.add_parameters(llama_configs, parameter_name='noc_stationary',              parameter_value='os')
    workload.add_parameters(llama_configs, parameter_name='node_stationary',             parameter_value='ws')

    # Operation latency
    workload.add_parameters(llama_configs, parameter_name='exp_mult_cycles',             parameter_value=30)
    workload.add_parameters(llama_configs, parameter_name='division_mult_cycles',        parameter_value=14)
    workload.add_parameters(llama_configs, parameter_name='accumulation_cycles',         parameter_value=1)

    # approximate operation latency
    workload.add_parameters(llama_configs, parameter_name='pwl_cycles',                  parameter_value=4)
    workload.add_parameters(llama_configs, parameter_name='approximate_division_cycles', parameter_value=1)

    # NoC tiling
    workload.add_parameters(llama_configs, parameter_name='noc_tile_m',                  parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_k',                  parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_n',                  parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='architecture',                parameter_value='systolic')

    # bitwidths (required by the shared performance models)
    workload.add_parameters(llama_configs, parameter_name='activation_bitwidth',         parameter_value=16)
    workload.add_parameters(llama_configs, parameter_name='weight_bitwidth',             parameter_value=4)

    ##############################################
    ###############    Event    ##################
    ##############################################
    # Performance model paths
    gemm      = 'zoo/llm/designs/systolic/performance/gemm.performance.py'
    nonlinear = 'zoo/llm/designs/systolic/performance/nonlinear.performance.py'
    llama     = 'zoo/llm/common/performance/model/llama.performance.py'
    model     = 'zoo/llm/common/performance/model/model_architecture.performance.py'
    memory    = 'zoo/llm/common/performance/memory/memory.performance.py'
    router    = 'zoo/llm/common/performance/router/router.performance.py'

    # Per-subarchitecture module groups (event sweep candidates ordered mac, figna, pwl, taylor)
    weight_mac    = ['wfifo', 'weight_register', 'int_to_fp']
    weight_figna  = ['wfifo', 'weight_register']
    array_mac     = ['icnt', 'icmp', 'iadd', 'wcnt', 'wcmp', 'wadd', 'multiplier', 'pe_register', 'accumulator', 'adder', 'ofifo']
    array_figna   = ['ch_aloc', 'ch_dealoc', 'int_to_fp_figna', 'prealigner'] + array_mac
    vector_mac    = ['multiplier_vector', 'accumulator_vector', 'mac_register_vector', 'register_vector']
    vector_pwl    = vector_mac + ['pwl_comparator', 'pwl_encoder', 'pwl_register', 'pipeline_register', 'adder_vector']
    vector_taylor = vector_mac + ['taylor_register', 'adder_vector']
    gemm_nl_mac   = ['ififo', 'input_register'] + weight_mac + array_mac
    gemm_nl_figna = ['ififo', 'input_register'] + weight_figna + array_figna

    gemm_mapping = ['input_gemm', 'weight_gemm', 'array_gemm', 'vector_gemm']
    mem_mapping  = ['isram_offchip_writes', 'isram_onchip_reads',
                 'wsram_offchip_writes', 'wsram_onchip_reads',
                 'osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_reads', 'osram_onchip_writes',
                 'dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads']
    router_ev = ['irouter_mapping', 'wrouter_mapping', 'orouter_mapping']

    # single-node event graph maps leaves to hardware + memory; multi-node adds the routers
    gemm_leaf    = gemm_mapping + mem_mapping
    softmax_leaf = ['gemm_nonlinear', 'softmax_nonlinear'] + mem_mapping
    silu_leaf    = ['gemm_nonlinear', 'silu_nonlinear'] + mem_mapping

    gemm_layers = ['proj_q', 'proj_k', 'proj_v', 'proj_a', 'qkt', 'av', 'proj_up', 'proj_down', 'proj_gate', 'output']
    nonlinear_layers = ['softmax', 'silu']
    layer_dict = {k: [k + '_prefill', k + '_decode'] for k in gemm_layers + nonlinear_layers}

    gemm_leaves = [layer + suffix for layer in gemm_layers for suffix in ['_prefill', '_decode']]

    
    event.add_event(name='llama_2',    subevent=['gemm', 'nonlinear'],                        performance=llama)
    event.add_event(name='gemm',       subevent=['projection', 'attention', 'ffn', 'output'], performance=model)
    event.add_event(name='nonlinear',  subevent=['softmax', 'silu'],                          performance=model)
    event.add_event(name='projection', subevent=['proj_q', 'proj_k', 'proj_v', 'proj_a'],     performance=model)
    event.add_event(name='attention',  subevent=['qkt', 'av'],                                performance=model)
    event.add_event(name='ffn',        subevent=['proj_up', 'proj_down', 'proj_gate'],        performance=model)

    
    
    event.add_event(event_dict=layer_dict, performance=model)
    # each leaf sweeps between the single-node and multi-node event graphs
    gemm_events    = event.add_event(name=gemm_leaves,                           subevent=[gemm_leaf, gemm_leaf + router_ev],       performance=model)
    softmax_events = event.add_event(name=['softmax_prefill', 'softmax_decode'], subevent=[softmax_leaf, softmax_leaf + router_ev], performance=model)
    silu_events    = event.add_event(name=['silu_prefill', 'silu_decode'],       subevent=[silu_leaf, silu_leaf + router_ev],       performance=model)

    # hardware events sweep per subarchitecture (mac, figna, pwl, taylor)
    event.add_event(name='input_gemm', subevent=['ififo', 'input_register'], performance=gemm)
    weight_gemm       = event.add_event(name='weight_gemm',       subevent=[weight_mac, weight_figna, weight_mac, weight_mac],     performance=gemm)
    array_gemm        = event.add_event(name='array_gemm',        subevent=[array_mac, array_figna, array_mac, array_mac],         performance=gemm)
    vector_gemm       = event.add_event(name='vector_gemm',       subevent=[vector_mac, vector_mac, vector_pwl, vector_taylor],    performance=gemm)
    gemm_nonlinear    = event.add_event(name='gemm_nonlinear',    subevent=[gemm_nl_mac, gemm_nl_figna, gemm_nl_mac, gemm_nl_mac], performance=nonlinear)
    softmax_nonlinear = event.add_event(name='softmax_nonlinear', subevent=[vector_mac, vector_mac, vector_pwl, vector_taylor],    performance=nonlinear)
    silu_nonlinear    = event.add_event(name='silu_nonlinear',    subevent=[vector_mac, vector_mac, vector_pwl, vector_taylor],    performance=nonlinear)

    event.add_event(name=['isram_offchip_writes', 'isram_onchip_reads'], subevent=['isram'], performance=memory)
    event.add_event(name=['wsram_offchip_writes', 'wsram_onchip_reads'], subevent=['wsram'], performance=memory)
    event.add_event(name=['osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_reads', 'osram_onchip_writes'], subevent=['osram'], performance=memory)
    event.add_event(name=['dram_input_reads', 'dram_weight_reads', 'dram_output_writes', 'dram_output_reads'],         subevent=['dram'],  performance=memory)

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
    arch_inst['single'] = [1]
    noc_sram_size = 2**19 # 512KB
    scaleup_sram_size = 2**22 # 4MB

    sram_bank = 4
    su_sram_width = 1024
    su_sram_depth = 1024
    min_su_array_dim = 32

    # === Sweep parameters =======================================
    # NoC parametersnode_inst
    node_inst = [[1, 1], [4, 4], [8, 8]]

    # Array parameters
    min_dim, max_dim = 8, 64

    arch_inst['dim']        = [[2**h] for h in range(int(log2(min_dim)), int(log2(max_dim)) + 1)]
    arch_inst['array']      = [[h[0], h[0]] for h in arch_inst['dim']]
    arch_inst['dim_double'] = [[h[0], 2]    for h in arch_inst['dim']]
    control_width = [int(log2(h[0])) - 1 for h in arch_inst['dim']]

    # non-standard instance shapes (every other module reuses 'single', 'dim', or 'array')
    arch_inst['pwl_comp'] = [[h[0], 21] for h in arch_inst['dim']]
    arch_inst['pwl_reg']  = [[h[0], 22] for h in arch_inst['dim']]
    arch_inst['taylor']   = [[9]        for h in arch_inst['dim']]

    # Memory and fifo parameters (width up to the scale-up sram width)
    min_sram_width, max_sram_width = 128, su_sram_width
    min_sram_depth, max_sram_depth = 512, 1024
    sram_width = [2**w for w in range(int(log2(min_sram_width)), int(log2(max_sram_width)) + 1)]
    sram_depth = [2**d for d in range(int(log2(min_sram_depth)), int(log2(max_sram_depth)) + 1)]
    fifo_depth = [h[0] for h in arch_inst['dim']]

    # expand every parameter to node instances
    for param, values in arch_inst.items():
        if isinstance(values[0], list):
            arch_inst[param] = [node + value for node in node_inst for value in values]
        else:
            arch_inst[param] = [node + values for node in node_inst]

    # === Memory modules =========================================
    # DRAM
    architecture.add_module(name='dram', instance=[1], tag=['memory', 'dram'], query={'interface': 'cacti7', 'class': 'dram', 'size': 8589934592, 'bandwidth': 128})

    # SRAMs
    isram = architecture.add_module(name='isram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': sram_bank, 'width': sram_width, 'depth': sram_depth})
    wsram = architecture.add_module(name='wsram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': sram_bank, 'width': sram_width, 'depth': sram_depth})
    osram = architecture.add_module(name='osram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': sram_bank, 'width': sram_width, 'depth': sram_depth})

    # === Fifo modules ===========================================
    ififo = architecture.add_module(name='ififo', instance=arch_inst['dim'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': fifo_depth})
    wfifo = architecture.add_module(name='wfifo', instance=arch_inst['dim'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 4,  'depth': fifo_depth})
    ofifo = architecture.add_module(name='ofifo', instance=arch_inst['dim'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': fifo_depth})

    # ---- pe cluster (instance = node + [H, H]) ----
    input_register  = architecture.add_module(name='input_register',  instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})
    weight_register = architecture.add_module(name='weight_register', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})
    multiplier      = architecture.add_module(name='multiplier',      instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'multiplierbf16'})
    accumulator     = architecture.add_module(name='accumulator',     instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'adderbf16'})
    pe_register     = architecture.add_module(name='pe_register',     instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})

    # ---- control registers (width = control_width) ----
    icnt = architecture.add_module(name='icnt', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': control_width})
    icmp = architecture.add_module(name='icmp', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': control_width})
    iadd = architecture.add_module(name='iadd', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': control_width})
    wcnt = architecture.add_module(name='wcnt', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': control_width})
    wcmp = architecture.add_module(name='wcmp', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': control_width})
    wadd = architecture.add_module(name='wadd', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': control_width})

    # ---- accumulator adder + vector lane ----
    adder               = architecture.add_module(name='adder',               instance=arch_inst['dim'],        tag=['onchip', 'array', 'accumulator'], query={'class': 'adderbf16'})
    multiplier_vector   = architecture.add_module(name='multiplier_vector',   instance=arch_inst['dim'],        tag=['onchip', 'array', 'vector'],      query={'class': 'multiplierbf16'})
    accumulator_vector  = architecture.add_module(name='accumulator_vector',  instance=arch_inst['dim'],        tag=['onchip', 'array', 'vector'],      query={'class': 'adderbf16'})
    mac_register_vector = architecture.add_module(name='mac_register_vector', instance=arch_inst['dim'],        tag=['onchip', 'array', 'vector'],      query={'class': 'register', 'width': 16})
    register_vector     = architecture.add_module(name='register_vector',     instance=arch_inst['dim_double'], tag=['onchip', 'array', 'vector'],      query={'class': 'register', 'width': 16})

    # ---- dtype-conversion + nonlinear modules ----
    int_to_fp         = architecture.add_module(name='int_to_fp',         instance=arch_inst['dim'],       tag=['onchip', 'array', 'pe'],        query={'class': 'int_to_bf16'})
    ch_aloc           = architecture.add_module(name='ch_aloc',           instance=arch_inst['array'],     tag=['onchip', 'array', 'pe'],        query={'class': 'ch_aloc'})
    ch_dealoc         = architecture.add_module(name='ch_dealoc',         instance=arch_inst['array'],     tag=['onchip', 'array', 'pe'],        query={'class': 'ch_dealoc'})
    int_to_fp_figna   = architecture.add_module(name='int_to_fp_figna',   instance=arch_inst['array'],     tag=['onchip', 'array', 'pe'],        query={'class': 'int_to_fp'})
    prealigner        = architecture.add_module(name='prealigner',        instance=arch_inst['array'],     tag=['onchip', 'array', 'pe'],        query={'class': 'prealigner'})
    pwl_comparator    = architecture.add_module(name='pwl_comparator',    instance=arch_inst['pwl_comp'],  tag=['onchip', 'array', 'nonlinear'], query={'class': 'pwl_bf_max'})
    pwl_encoder       = architecture.add_module(name='pwl_encoder',       instance=arch_inst['dim'],       tag=['onchip', 'array', 'nonlinear'], query={'class': 'pwl_22_seg_encoder'})
    pwl_register      = architecture.add_module(name='pwl_register',      instance=arch_inst['pwl_reg'],   tag=['onchip', 'array', 'nonlinear'], query={'class': 'register', 'width': 32})
    pipeline_register = architecture.add_module(name='pipeline_register', instance=arch_inst['pwl_reg'],   tag=['onchip', 'array', 'nonlinear'], query={'class': 'register', 'width': 16})
    adder_vector      = architecture.add_module(name='adder_vector',      instance=arch_inst['dim'],       tag=['onchip', 'array', 'nonlinear'], query={'class': 'adderbf16'})
    taylor_register   = architecture.add_module(name='taylor_register',   instance=arch_inst['taylor'],    tag=['onchip', 'array', 'nonlinear'], query={'class': 'register', 'width': 16})

    # ---- routers (always present; instance = node + [1], so single-node collapses to 1) ----
    irouter = architecture.add_module(name='irouter', instance=arch_inst['single'], tag=['onchip', 'router'], query={'class': 'noc_router', 'bandwidth': 64})
    wrouter = architecture.add_module(name='wrouter', instance=arch_inst['single'], tag=['onchip', 'router'], query={'class': 'noc_router', 'bandwidth': 64})
    orouter = architecture.add_module(name='orouter', instance=arch_inst['single'], tag=['onchip', 'router'], query={'class': 'noc_router', 'bandwidth': 64})

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
    # Master sweep: every dim-derived instance (node x dim) locked to one index.
    agraph.direct_constraint([
        ififo['instance'],
        wfifo['instance'],
        ofifo['instance'],
        input_register['instance'],
        weight_register['instance'],
        multiplier['instance'],
        accumulator['instance'],
        pe_register['instance'],
        icnt['instance'],
        icmp['instance'],
        iadd['instance'],
        wcnt['instance'],
        wcmp['instance'],
        wadd['instance'],
        adder['instance'],
        multiplier_vector['instance'],
        accumulator_vector['instance'],
        mac_register_vector['instance'],
        register_vector['instance'],
        int_to_fp['instance'],
        ch_aloc['instance'],
        ch_dealoc['instance'],
        int_to_fp_figna['instance'],
        prealigner['instance'],
        pwl_comparator['instance'],
        pwl_encoder['instance'],
        pwl_register['instance'],
        pipeline_register['instance'],
        adder_vector['instance'],
        taylor_register['instance'],
    ])

    # fifo depths follow the array dimension
    agraph.direct_constraint([
        ififo['query']['depth'],
        wfifo['query']['depth'],
        ofifo['query']['depth']
    ])
    agraph.direct_constraint_partition(ififo['query']['depth'], multiplier['instance'])

    # control register widths follow the array dimension
    agraph.direct_constraint([
        icnt['query']['width'],
        icmp['query']['width'],
        iadd['query']['width'],
        wcnt['query']['width'],
        wcmp['query']['width'],
        wadd['query']['width']
    ])
    agraph.direct_constraint_partition(icnt['query']['width'], multiplier['instance'])

    agraph.direct_constraint([
        isram['query']['width'],
        wsram['query']['width'],
        osram['query']['width']
    ])

    agraph.direct_constraint([
        isram['query']['depth'],
        wsram['query']['depth'],
        osram['query']['depth']
    ])

    # node-level modules share one instance per node mesh
    agraph.direct_constraint([
        isram['instance'],
        wsram['instance'],
        osram['instance'],
        irouter['instance'],
        wrouter['instance'],
        orouter['instance']
    ])
    agraph.conditional_constraint(
        m = multiplier['instance'],
        r = irouter['instance'],
        condition = lambda m, r: m[0] == r[0]
    )

    # subarchitecture: hardware events and the workload flag sweep together (mac, figna, pwl, taylor)
    subarch_params = [p['parameter'] for p in subarch.values()]
    agraph.direct_constraint([
        weight_gemm['subevent'],
        array_gemm['subevent'],
        vector_gemm['subevent'],
        gemm_nonlinear['subevent'],
        softmax_nonlinear['subevent'],
        silu_nonlinear['subevent']
    ] + subarch_params)

    # network: every software leaf picks the same event graph (single-node or multi-node)
    leaf_events = list(gemm_events.values()) + list(softmax_events.values()) + list(silu_events.values())
    agraph.direct_constraint([e['subevent'] for e in leaf_events])

    # routers are mapped if and only if the design is multi-node
    agraph.conditional_constraint(
        m = multiplier['instance'],
        ev = gemm_events['proj_q_prefill']['subevent'],
        condition = lambda m, ev: ('irouter_mapping' in ev) == (m[0] != 1)
    )

    # multi-node designs are limited to the mac and figna subarchitectures
    agraph.conditional_constraint(
        m = multiplier['instance'],
        s = subarch_params[0],
        condition = lambda m, s: (m[0] == 1) or (s in ['mac', 'figna'])
    )

    # batch size sweeps on every single-node design; multi-node fixes batch size to 8
    agraph.conditional_constraint(
        m = multiplier['instance'],
        b = list(batch_size.values())[0]['parameter'],
        condition = lambda m, b: (m[0] == 1) or (b == 8)
    )

    # seq len sweeps on every single-node design; multi-node fixes seq len to 4096
    agraph.conditional_constraint(
        m = multiplier['instance'],
        q = list(max_seq_len.values())[0]['parameter'],
        condition = lambda m, q: (m[0] == 1) or (q == 4096)
    )

    # multi-node meshes pair only with the small (NoC-scale) array dimensions
    agraph.conditional_constraint(
        m = multiplier['instance'],
        condition = lambda m: (m[0] == 1) or (m[2] < min_su_array_dim)
    )

    # srams are sized by array dimension: NoC-scale arrays pair dim*16-wide 512KB srams,
    # scale-up arrays pair 1024-wide 4MB srams
    agraph.conditional_constraint(
        m = multiplier['instance'],
        w = isram['query']['width'],
        d = isram['query']['depth'],
        condition = lambda m, w, d: ((m[2] * 16 == w) and (w * d * sram_bank == noc_sram_size)) if m[2] < min_su_array_dim else ((w == su_sram_width) and (w * d * sram_bank == scaleup_sram_size))
    )

    return agraph.generate()
