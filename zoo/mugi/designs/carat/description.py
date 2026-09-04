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

    # workload parameters
    min_seq_len, max_seq_len = 128, 4096
    min_batch_size, max_batch_size = 1, 32
    seq_len = [2**l for l in range(int(log2(min_seq_len)), int(log2(max_seq_len)) + 1)]
    batch_size = [2**b for b in range(int(log2(min_batch_size)), int(log2(max_batch_size)) + 1)]

    # llama 2 7b
    llama['llama_2_7b'].add_parameter(parameter_name='dim',        parameter_value=4096)
    llama['llama_2_7b'].add_parameter(parameter_name='layers',     parameter_value=32)
    llama['llama_2_7b'].add_parameter(parameter_name='heads',      parameter_value=32)
    llama['llama_2_7b'].add_parameter(parameter_name='hidden_dim', parameter_value=11008)
    llama['llama_2_7b'].add_parameter(parameter_name='kv_heads',   parameter_value=32)

    # llama 2 13b
    llama['llama_2_13b'].add_parameter(parameter_name='dim',        parameter_value=5120)
    llama['llama_2_13b'].add_parameter(parameter_name='layers',     parameter_value=40)
    llama['llama_2_13b'].add_parameter(parameter_name='heads',      parameter_value=40)
    llama['llama_2_13b'].add_parameter(parameter_name='hidden_dim', parameter_value=13824)
    llama['llama_2_13b'].add_parameter(parameter_name='kv_heads',   parameter_value=40)

    # llama 2 70b (gqa kv head sweep folded into the 70B model)
    llama['llama_2_70b'].add_parameter(parameter_name='dim',        parameter_value=8192)
    llama['llama_2_70b'].add_parameter(parameter_name='layers',     parameter_value=80)
    llama['llama_2_70b'].add_parameter(parameter_name='heads',      parameter_value=64)
    llama['llama_2_70b'].add_parameter(parameter_name='hidden_dim', parameter_value=28672)
    kv_heads = llama['llama_2_70b'].add_parameter(parameter_name='kv_heads',   parameter_value=[8, 16, 32, 64], sweep=True)

    # parameters added to all llama configurations
    max_seq_len = workload.add_parameters(configs=llama_configs, parameter_name='max_seq_len',          parameter_value=seq_len,    sweep=True)
    batch_size  = workload.add_parameters(configs=llama_configs, parameter_name='batch_size',           parameter_value=batch_size, sweep=True)
    workload.add_parameters(configs=llama_configs, parameter_name='prefill_seq_len',      parameter_value=64)
    workload.add_parameters(configs=llama_configs, parameter_name='vocab_size',           parameter_value=32000)
    workload.add_parameters(configs=llama_configs, parameter_name='activation_bitwidth',  parameter_value=16)
    workload.add_parameters(configs=llama_configs, parameter_name='weight_bitwidth',      parameter_value=4)
    workload.add_parameters(configs=llama_configs, parameter_name='noc_stationary',       parameter_value='os')
    workload.add_parameters(configs=llama_configs, parameter_name='node_stationary',      parameter_value='os')
    workload.add_parameters(configs=llama_configs, parameter_name='exp_mult_cycles',      parameter_value=30)
    workload.add_parameters(configs=llama_configs, parameter_name='division_mult_cycles', parameter_value=14)
    workload.add_parameters(configs=llama_configs, parameter_name='accumulation_cycles',  parameter_value=1)
    workload.add_parameters(configs=llama_configs, parameter_name='cycles',               parameter_value=8)
    workload.add_parameters(configs=llama_configs, parameter_name='noc_tile_m',           parameter_value=256)
    workload.add_parameters(configs=llama_configs, parameter_name='noc_tile_k',           parameter_value=256)
    workload.add_parameters(configs=llama_configs, parameter_name='noc_tile_n',           parameter_value=256)
    workload.add_parameters(configs=llama_configs, parameter_name='architecture',         parameter_value='carat')

    ##############################################
    ###############    Event    ##################
    ##############################################
    # Performance model paths
    gemm = 'zoo/mugi/designs/carat/performance/gemm.performance.py'
    nonlinear = 'zoo/mugi/designs/carat/performance/nonlinear.performance.py'
    llama = 'zoo/mugi/common/performance/model/llama.performance.py'
    model = 'zoo/mugi/common/performance/model/model_architecture.performance.py'
    memory = 'zoo/mugi/common/performance/memory/memory.performance.py'
    router = 'zoo/mugi/common/performance/router/router.performance.py'

    proj_layers = ['proj_q', 'proj_k', 'proj_v', 'proj_a']
    attn_layers = ['qkt', 'av']
    ffn_layers = ['proj_up', 'proj_down', 'proj_gate']
    nonlinear_layers = ['softmax', 'silu']

    # hardware / memory / router event groups
    gemm_hw_events      = ['input_gemm', 'input_reuse_gemm', 'counter_reuse', 'weight_gemm', 'weight_reuse_gemm', 'array_gemm', 'vector_gemm']
    nonlinear_hw_events = ['gemm_nonlinear', 'vector_nonlinear']
    mem_mapping = ['isram_offchip_writes', 'isram_onchip_reads',
                   'wsram_offchip_writes', 'wsram_onchip_reads',
                   'osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_writes', 'osram_onchip_reads',
                   'dram_input_reads', 'dram_weight_reads', 'dram_output_reads', 'dram_output_writes']
    router_ev = ['irouter_mapping', 'wrouter_mapping', 'orouter_mapping']

    # single-node event graph maps leaves to hardware + memory; multi-node adds the routers
    gemm_leaf      = gemm_hw_events + mem_mapping
    nonlinear_leaf = nonlinear_hw_events + mem_mapping

    # model events
    event.add_event(name='llama_2', subevent=['gemm', 'nonlinear'], performance=llama)
    event.add_event(name='gemm', subevent=['projection', 'attention', 'ffn', 'output'], performance=model)
    event.add_event(name='nonlinear', subevent=nonlinear_layers, performance=model)
    event.add_event(name='projection', subevent=proj_layers, performance=model)
    event.add_event(name='attention', subevent=attn_layers, performance=model)
    event.add_event(name='ffn', subevent=ffn_layers, performance=model)
    event.add_event(name='output', subevent=['output_prefill', 'output_decode'], performance=model)

    # model decompisition events
    gemm_layers = proj_layers + attn_layers + ffn_layers
    layer_dict = {k: [k + '_prefill', k + '_decode'] for k in gemm_layers + nonlinear_layers}

    event.add_event(event_dict=layer_dict, performance=model)

    # model -> hardware mapping events (each leaf sweeps between the single-node and multi-node event graphs)
    gemm_layer_subevents = [
        layer + suffix
        for layer in gemm_layers
        for suffix in ['_prefill', '_decode']
    ] + ['output_prefill', 'output_decode']
    nonlinear_layer_subevents = [
        layer + suffix
        for layer in nonlinear_layers
        for suffix in ['_prefill', '_decode']
    ]

    gemm_events      = event.add_event(name=gemm_layer_subevents,      subevent=[gemm_leaf, gemm_leaf + router_ev],           performance=model)
    nonlinear_events = event.add_event(name=nonlinear_layer_subevents, subevent=[nonlinear_leaf, nonlinear_leaf + router_ev], performance=model)

    # hardware events (gemm)
    event.add_event(name='input_gemm',    subevent=['ififo'],           performance=gemm)
    event.add_event(name='weight_gemm',   subevent=['wfifo'],           performance=gemm)
    event.add_event(name='counter_reuse', subevent=['counter', 'creg'], performance=gemm)

    event.add_event(name='weight_reuse_gemm', subevent=['comparator', 'ireg', 'z_flag', 'nan_flag', 'sign_fifo'],    performance=gemm)
    event.add_event(name='input_reuse_gemm',  subevent=['accumulator', 'accumulator_register', 'exp_scale', 'wreg'], performance=gemm)

    event.add_event(name='vector_gemm', subevent=['multiplier_vector', 'accumulator_vector', 'register_vector', 'mac_register_vector'],                       performance=gemm)
    event.add_event(name='array_gemm',  subevent=['temporal_register', 'and_gate', 'or_gate', 'areg', 'pe_fifo', 'sign_xor', 'shifterexp', 'adder', 'ofifo'], performance=gemm)

    # hardware events (nonlinear)
    event.add_event(name='gemm_nonlinear',
                    subevent=['ififo', 'accumulator', 'accumulator_register', 'exp_scale', 'wreg', 'counter', 'wfifo', 'comparator', 'ireg', 'z_flag', 'nan_flag',
                              'sign_fifo', 'temporal_register', 'and_gate', 'or_gate', 'areg', 'creg', 'pe_fifo', 'sign_xor', 'shifterexp', 'adder', 'ofifo'],
                    performance=nonlinear)
    event.add_event(name='vector_nonlinear', subevent=['multiplier_vector', 'accumulator_vector', 'register_vector', 'mac_register_vector'], performance=nonlinear)

    # memory events
    event.add_event(name=['isram_offchip_writes', 'isram_onchip_reads'],                                               subevent=['isram'], performance=memory)
    event.add_event(name=['wsram_offchip_writes', 'wsram_onchip_reads'],                                               subevent=['wsram'], performance=memory)
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
    arch_inst['width'] = [width]

    # Memory parameters
    sram_bank = 4
    sram_size = 2**19 # 512KB (width * depth * bank)
    min_sram_width, max_sram_width = 128, 256
    min_sram_depth, max_sram_depth = 512, 1024

    isram_width = min_sram_width
    isram_depth = max_sram_depth

    # === Sweep parameters =======================================
    # NoC parameters
    node_inst = [[1, 1], [4, 4], [8, 8]]              # 1x1 (single-node), then two multi-node meshes

    # Array parameters
    min_height, max_height = 32, 256

    arch_inst['height']    = [[2**h] for h in range(int(log2(min_height)), int(log2(max_height)) + 1)]
    arch_inst['height_2']  = [[h[0], 2]          for h in arch_inst['height']]
    arch_inst['array']     = [[h[0], width]      for h in arch_inst['height']]
    arch_inst['array_2']   = [[h[0], width, 2]   for h in arch_inst['height']]
    arch_inst['creg']      = [[h[0] // 8, width] for h in arch_inst['height']]
    # broadcast register count: 1 for heights up to 128, 3 for 256
    arch_inst['broadcast'] = [[h[0] // 128 + (h[0] != 128), width] for h in arch_inst['height']]
    fifo_depth             = [h[0] // 32 for h in arch_inst['height']]

    # Memory parameters
    wsram_width = [2**w for w in range(int(log2(min_sram_width)), int(log2(max_sram_width)) + 1)]
    osram_width = [2**w for w in range(int(log2(min_sram_width)), int(log2(max_sram_width)) + 1)]
    wsram_depth = [2**d for d in range(int(log2(min_sram_depth)), int(log2(max_sram_depth)) + 1)]
    osram_depth = [2**d for d in range(int(log2(min_sram_depth)), int(log2(max_sram_depth)) + 1)]

    # expand every parameter to node instances
    for param, values in arch_inst.items():
        if isinstance(values[0], list):
            arch_inst[param] = [node + value for node in node_inst for value in values]
        else:
            arch_inst[param] = [node + values for node in node_inst]

    # === Memory modules =======================================
    # DRAM
    architecture.add_module(name='dram', instance=[1], tag=['memory', 'dram'], query={'interface': 'cacti7', 'class': 'dram', 'size': 8589934592, 'bandwidth': 128})

    # SRAMs
    isram = architecture.add_module(name='isram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': sram_bank, 'width': isram_width, 'depth': isram_depth})
    wsram = architecture.add_module(name='wsram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': sram_bank, 'width': wsram_width, 'depth': wsram_depth})
    osram = architecture.add_module(name='osram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': sram_bank, 'width': osram_width, 'depth': osram_depth})

    # === Fifo modules =======================================
    ififo = architecture.add_module(name='ififo', instance=arch_inst['width'],  tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': fifo_depth})
    wfifo = architecture.add_module(name='wfifo', instance=arch_inst['height'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 8,  'depth': fifo_depth})
    ofifo = architecture.add_module(name='ofifo', instance=arch_inst['height'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': fifo_depth})

    # ---- tc (temporal coding) ----
    comparator = architecture.add_module(name='comparator', instance=arch_inst['height'], tag=['onchip', 'array', 'tc'], query={'class': 'comparator',  'width': 3})
    ireg       = architecture.add_module(name='ireg',       instance=arch_inst['height'], tag=['onchip', 'array', 'tc'], query={'class': 'register',    'width': 12})
    z_flag     = architecture.add_module(name='z_flag',     instance=arch_inst['height'], tag=['onchip', 'array', 'tc'], query={'class': 'or_bitwise',  'width': 7})
    nan_flag   = architecture.add_module(name='nan_flag',   instance=arch_inst['height'], tag=['onchip', 'array', 'tc'], query={'class': 'and_bitwise', 'width': 7})

    # ---- pe cluster ----
    temporal_register = architecture.add_module(name='temporal_register', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 1})
    and_gate          = architecture.add_module(name='and_gate', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'and_bitwise', 'width': 16})
    areg              = architecture.add_module(name='areg', instance=arch_inst['array'], tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 1.5})
    or_gate           = architecture.add_module(name='or_gate', instance=arch_inst['array_2'], tag=['onchip', 'array', 'pe'], query={'class': 'or_bitwise', 'width': 16})
    sign_xor          = architecture.add_module(name='sign_xor', instance=arch_inst['height'], tag=['onchip', 'array', 'pe'], query={'class': 'xnor_bitwise', 'width': 1})

    # ---- value-reuse / accumulator array ----
    counter = architecture.add_module(name='counter', instance=arch_inst['single'], tag=['onchip', 'array', 'value_reuse'], query={'class': 'counter', 'width': 3})
    creg = architecture.add_module(name='creg', instance=arch_inst['creg'], tag=['onchip', 'array', 'value_reuse'], query={'class': 'register', 'width': 3})
    accumulator = architecture.add_module(name='accumulator', instance=arch_inst['width'], tag=['onchip', 'array', 'value_reuse'], query={'class': 'adderbf16'})
    accumulator_register = architecture.add_module(name='accumulator_register', instance=arch_inst['broadcast'], tag=['onchip', 'array', 'value_reuse'], query={'class': 'register', 'width': 16})
    wreg = architecture.add_module(name='wreg', instance=arch_inst['width'], tag=['onchip', 'array', 'fifo'], query={'class': 'register', 'width': 16})
    sign_fifo = architecture.add_module(name='sign_fifo', instance=arch_inst['height_2'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 8, 'depth': 1})
    pe_fifo = architecture.add_module(name='pe_fifo', instance=arch_inst['height_2'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': 8})
    adder = architecture.add_module(name='adder', instance=arch_inst['height'], tag=['onchip', 'array', 'accumulator'], query={'class': 'adderbf16'})
    exp_scale = architecture.add_module(name='exp_scale', instance=arch_inst['width'], tag=['onchip', 'array', 'accumulator'], query={'class': 'adder', 'width': 8})
    shifterexp = architecture.add_module(name='shifterexp', instance=arch_inst['height'], tag=['onchip', 'array', 'accumulator'], query={'class': 'adder', 'width': 8})

    # ---- simd vector ----
    multiplier_vector = architecture.add_module(name='multiplier_vector', instance=arch_inst['height'], tag=['onchip', 'array', 'vector'], query={'class': 'multiplierbf16'})
    accumulator_vector = architecture.add_module(name='accumulator_vector', instance=arch_inst['height'], tag=['onchip', 'array', 'vector'], query={'class': 'adderbf16'})
    register_vector = architecture.add_module(name='register_vector', instance=arch_inst['height'], tag=['onchip', 'array', 'vector'], query={'class': 'register', 'width': 16})
    mac_register_vector = architecture.add_module(name='mac_register_vector', instance=arch_inst['height'], tag=['onchip', 'array', 'vector'], query={'class': 'register', 'width': 16})

    # ---- routers ----
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
    # Master sweep: every height-derived instance (node x height) locked to one index.
    agraph.direct_constraint([
        wfifo['instance'],
        ofifo['instance'],
        comparator['instance'],
        ireg['instance'],
        z_flag['instance'],
        nan_flag['instance'],
        temporal_register['instance'],
        and_gate['instance'],
        areg['instance'],
        or_gate['instance'],
        sign_xor['instance'],
        creg['instance'],
        accumulator_register['instance'],
        sign_fifo['instance'],
        pe_fifo['instance'],
        adder['instance'],
        shifterexp['instance'],
        multiplier_vector['instance'],
        accumulator_vector['instance'],
        register_vector['instance'],
        mac_register_vector['instance']
    ])

    # fifo depths follow the array height
    agraph.direct_constraint([
        ififo['query']['depth'],
        wfifo['query']['depth'],
        ofifo['query']['depth']
    ])
    agraph.direct_constraint_partition(ififo['query']['depth'], adder['instance'])

    # width-level modules share one instance per node mesh
    agraph.direct_constraint([
        ififo['instance'],
        accumulator['instance'],
        wreg['instance'],
        exp_scale['instance']
    ])
    agraph.conditional_constraint(
        w = ififo['instance'],
        height = adder['instance'],
        condition = lambda w, height: w[0:2] == height[0:2]
    )

    # node-level modules share one instance per node mesh
    agraph.direct_constraint([
        isram['instance'],
        wsram['instance'],
        osram['instance'],
        counter['instance'],
        irouter['instance'],
        wrouter['instance'],
        orouter['instance']
    ])
    agraph.conditional_constraint(
        router = irouter['instance'],
        height = adder['instance'],
        condition = lambda router, height: router[0:2] == height[0:2]
    )

    # wsram and osram sweep together
    agraph.direct_constraint([wsram['query']['width'], osram['query']['width']])
    agraph.direct_constraint([wsram['query']['depth'], osram['query']['depth']])

    # sram width matches the array height (down to the minimum width) and width * depth * bank matches the sram size
    agraph.conditional_constraint(
        height = adder['instance'],
        w = wsram['query']['width'],
        d = wsram['query']['depth'],
        condition = lambda height, w, d: (w == max(min_sram_width, height[2])) and (w * d * sram_bank == sram_size)
    )

    # network: every software leaf picks the same event graph (single-node or multi-node)
    leaf_events = list(gemm_events.values()) + list(nonlinear_events.values())
    agraph.direct_constraint([e['subevent'] for e in leaf_events])

    # routers are mapped if and only if the design is multi-node
    agraph.conditional_constraint(
        height = adder['instance'],
        ev = gemm_events['proj_q_prefill']['subevent'],
        condition = lambda height, ev: ('irouter_mapping' in ev) == (height[0] != 1)
    )

    # batch size sweeps only on the single-node design; multi-node fixes batch size to 8
    agraph.conditional_constraint(
        height = adder['instance'],
        b = list(batch_size.values())[0]['parameter'],
        condition = lambda height, b: (height[0] == 1) or (b == 8)
    )

    # seq len sweeps only on the single-node design; multi-node fixes seq len to 4096
    agraph.conditional_constraint(
        height = adder['instance'],
        q = list(max_seq_len.values())[0]['parameter'],
        condition = lambda height, q: (height[0] == 1) or (q == 4096)
    )

    return agraph.generate()
