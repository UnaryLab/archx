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
    subarch     = workload.add_parameters(llama_configs, parameter_name='subarch',     parameter_value=['mac', 'figna'],                sweep=True)

    # shared constant parameters
    # model parameters
    workload.add_parameters(llama_configs, parameter_name='prefill_seq_len',      parameter_value=64)
    workload.add_parameters(llama_configs, parameter_name='vocab_size',           parameter_value=32000)

    # bitwidths
    workload.add_parameters(llama_configs, parameter_name='activation_bitwidth',  parameter_value=16)
    workload.add_parameters(llama_configs, parameter_name='weight_bitwidth',      parameter_value=4)

    # Noc configuration
    workload.add_parameters(llama_configs, parameter_name='noc_stationary',       parameter_value='os')
    workload.add_parameters(llama_configs, parameter_name='node_stationary',      parameter_value='ws')

    # Operation latency
    workload.add_parameters(llama_configs, parameter_name='exp_mult_cycles',      parameter_value=30)
    workload.add_parameters(llama_configs, parameter_name='division_mult_cycles', parameter_value=14)
    workload.add_parameters(llama_configs, parameter_name='accumulation_cycles',  parameter_value=1)

    # architecture selection
    workload.add_parameters(llama_configs, parameter_name='architecture',         parameter_value='simd')

    # NoC tiling
    workload.add_parameters(llama_configs, parameter_name='noc_tile_m',           parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_k',           parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_n',           parameter_value=256)

    ##############################################
    ###############    Event    ##################
    ##############################################
    # Performance model paths
    gemm      = 'zoo/mugi/designs/simd/performance/gemm.performance.py'
    nonlinear = 'zoo/mugi/designs/simd/performance/nonlinear.performance.py'
    llama     = 'zoo/mugi/common/performance/model/llama.performance.py'
    model     = 'zoo/mugi/common/performance/model/model_architecture.performance.py'
    memory    = 'zoo/mugi/common/performance/memory/memory.performance.py'
    router    = 'zoo/mugi/common/performance/router/router.performance.py'

    # Per-subarchitecture module groups (event sweep candidates ordered mac, figna)
    weight_mac    = ['wfifo', 'int_to_fp', 'weight_register']
    weight_figna  = ['wfifo', 'weight_register']
    array_mac     = ['multiplier', 'pe_register', 'accumulator', 'accumulator_register', 'adder', 'ofifo']
    array_figna   = ['ch_aloc', 'ch_dealoc', 'int_to_fp_figna', 'prealigner'] + array_mac
    gemm_nl_mac   = ['ififo', 'input_register'] + weight_mac + array_mac
    gemm_nl_figna = ['ififo', 'input_register'] + weight_figna + array_figna

    gemm_mapping = ['input_gemm', 'weight_gemm', 'array_gemm', 'vector_gemm']
    mem_mapping  = ['isram_offchip_writes', 'isram_onchip_reads',
                 'wsram_offchip_writes', 'wsram_onchip_reads',
                 'osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_writes', 'osram_onchip_reads',
                 'dram_input_reads', 'dram_weight_reads', 'dram_output_reads', 'dram_output_writes']
    router_ev = ['irouter_mapping', 'wrouter_mapping', 'orouter_mapping']

    # single-node event graph maps leaves to hardware + memory; multi-node adds the routers
    gemm_leaf      = gemm_mapping + mem_mapping
    nonlinear_leaf = ['gemm_nonlinear', 'vector_nonlinear'] + mem_mapping

    gemm_layers = ['proj_q', 'proj_k', 'proj_v', 'proj_a', 'qkt', 'av', 'proj_up', 'proj_down', 'proj_gate', 'output']
    nonlinear_layers = ['softmax', 'silu']
    layer_dict = {k: [k + '_prefill', k + '_decode'] for k in gemm_layers + nonlinear_layers}

    gemm_leaves      = [layer + suffix for layer in gemm_layers      for suffix in ['_prefill', '_decode']]
    nonlinear_leaves = [layer + suffix for layer in nonlinear_layers for suffix in ['_prefill', '_decode']]

    event.add_event(name='llama_2',    subevent=['gemm', 'nonlinear'],                        performance=llama)
    event.add_event(name='gemm',       subevent=['projection', 'attention', 'ffn', 'output'], performance=model)
    event.add_event(name='nonlinear',  subevent=['softmax', 'silu'],                          performance=model)
    event.add_event(name='projection', subevent=['proj_q', 'proj_k', 'proj_v', 'proj_a'],     performance=model)
    event.add_event(name='attention',  subevent=['qkt', 'av'],                                performance=model)
    event.add_event(name='ffn',        subevent=['proj_up', 'proj_down', 'proj_gate'],        performance=model)

    event.add_event(event_dict=layer_dict, performance=model)
    # each leaf sweeps between the single-node and multi-node event graphs
    gemm_events      = event.add_event(name=gemm_leaves,      subevent=[gemm_leaf, gemm_leaf + router_ev],           performance=model)
    nonlinear_events = event.add_event(name=nonlinear_leaves, subevent=[nonlinear_leaf, nonlinear_leaf + router_ev], performance=model)

    # hardware events sweep per subarchitecture (mac, figna)
    event.add_event(name='input_gemm', subevent=['ififo', 'input_register'], performance=gemm)
    weight_gemm      = event.add_event(name='weight_gemm',    subevent=[weight_mac, weight_figna],     performance=gemm)
    array_gemm       = event.add_event(name='array_gemm',     subevent=[array_mac, array_figna],       performance=gemm)
    event.add_event(name='vector_gemm', subevent=['multiplier_vector', 'accumulator_vector', 'mac_register_vector', 'register_vector'], performance=gemm)
    gemm_nonlinear   = event.add_event(name='gemm_nonlinear', subevent=[gemm_nl_mac, gemm_nl_figna],   performance=nonlinear)
    event.add_event(name='vector_nonlinear', subevent=['multiplier_vector', 'accumulator_vector', 'mac_register_vector', 'register_vector'], performance=nonlinear)

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
    arch_inst['single'] = [1]
    sram_bank = 4

    # === Sweep parameters =======================================
    # NoC parameters
    node_inst = [[1, 1], [4, 4], [8, 8]]

    # Array parameters
    min_dim, max_dim = 8, 64

    arch_inst['dim']        = [[2**h] for h in range(int(log2(min_dim)), int(log2(max_dim)) + 1)]
    arch_inst['array']      = [[h[0], h[0]]     for h in arch_inst['dim']]
    arch_inst['array_tree'] = [[h[0], h[0] - 1] for h in arch_inst['dim']]

    # Memory and fifo parameters (per dim 8/16/32/64; no clean formula)
    sram_width = [128, 256, 1024, 1024]
    sram_depth = [1024, 512, 1024, 1024]
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
    pe_register     = architecture.add_module(name='pe_register',     instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})

    # ---- adder tree + per-column accumulator ----
    accumulator          = architecture.add_module(name='accumulator',          instance=arch_inst['array_tree'], tag=['onchip', 'array', 'pe'],          query={'class': 'adderbf16'})
    accumulator_register = architecture.add_module(name='accumulator_register', instance=arch_inst['dim'],        tag=['onchip', 'array', 'pe'],          query={'class': 'register', 'width': 16})
    adder                = architecture.add_module(name='adder',                instance=arch_inst['dim'],        tag=['onchip', 'array', 'accumulator'], query={'class': 'adderbf16'})

    # ---- vector lane ----
    multiplier_vector   = architecture.add_module(name='multiplier_vector',   instance=arch_inst['dim'], tag=['onchip', 'array', 'vector'], query={'class': 'multiplierbf16'})
    accumulator_vector  = architecture.add_module(name='accumulator_vector',  instance=arch_inst['dim'], tag=['onchip', 'array', 'vector'], query={'class': 'adderbf16'})
    mac_register_vector = architecture.add_module(name='mac_register_vector', instance=arch_inst['dim'], tag=['onchip', 'array', 'vector'], query={'class': 'register', 'width': 16})
    register_vector     = architecture.add_module(name='register_vector',     instance=arch_inst['dim'], tag=['onchip', 'array', 'vector'], query={'class': 'register', 'width': 16})

    # ---- dtype-conversion modules ----
    int_to_fp       = architecture.add_module(name='int_to_fp',       instance=arch_inst['dim'],   tag=['onchip', 'array', 'pe'], query={'class': 'int_to_bf16'})
    ch_aloc         = architecture.add_module(name='ch_aloc',         instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'ch_aloc'})
    ch_dealoc       = architecture.add_module(name='ch_dealoc',       instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'ch_dealoc'})
    int_to_fp_figna = architecture.add_module(name='int_to_fp_figna', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'int_to_fp'})
    prealigner      = architecture.add_module(name='prealigner',      instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'prealigner'})

    # ---- routers (single-node event graphs never reference them) ----
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
        pe_register['instance'],
        accumulator['instance'],
        accumulator_register['instance'],
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
    ])

    # fifo depths follow the array dimension
    agraph.direct_constraint([
        ififo['query']['depth'],
        wfifo['query']['depth'],
        ofifo['query']['depth']
    ])
    agraph.direct_constraint_partition(ififo['query']['depth'], multiplier['instance'])

    # sram widths and depths follow the array dimension
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
    agraph.direct_constraint_partition(isram['query']['width'], multiplier['instance'])
    agraph.direct_constraint_partition(isram['query']['depth'], multiplier['instance'])

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

    # subarchitecture: hardware events and the workload flag sweep together (mac, figna)
    subarch_params = [p['parameter'] for p in subarch.values()]
    agraph.direct_constraint([
        weight_gemm['subevent'],
        array_gemm['subevent'],
        gemm_nonlinear['subevent']
    ] + subarch_params)

    # network: every software leaf picks the same event graph (single-node or multi-node)
    leaf_events = list(gemm_events.values()) + list(nonlinear_events.values())
    agraph.direct_constraint([e['subevent'] for e in leaf_events])

    # routers are mapped if and only if the design is multi-node
    agraph.conditional_constraint(
        m = multiplier['instance'],
        ev = gemm_events['proj_q_prefill']['subevent'],
        condition = lambda m, ev: ('irouter_mapping' in ev) == (m[0] != 1)
    )

    # batch size sweeps on every single-node design; multi-node fixes batch size to 8
    agraph.conditional_constraint(
        m = multiplier['instance'],
        b = list(batch_size.values())[0]['parameter'],
        condition = lambda m, b: (m[0] == 1) or (b == 8)
    )

    # seq len sweeps for both single-node designs; multi-node fixes seq len to 4096
    agraph.conditional_constraint(
        m = multiplier['instance'],
        q = list(max_seq_len.values())[0]['parameter'],
        condition = lambda m, q: (m[0] == 1) or (q == 4096)
    )

    return agraph.generate()
