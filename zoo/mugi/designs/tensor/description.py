from archx.programming.graph.agraph import AGraph


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

    # shared constant parameters
    workload.add_parameters(llama_configs, parameter_name='prefill_seq_len',     parameter_value=64)
    workload.add_parameters(llama_configs, parameter_name='vocab_size',          parameter_value=32000)
    workload.add_parameters(llama_configs, parameter_name='activation_bitwidth', parameter_value=16)
    workload.add_parameters(llama_configs, parameter_name='weight_bitwidth',     parameter_value=4)
    workload.add_parameters(llama_configs, parameter_name='noc_stationary',      parameter_value='os')
    workload.add_parameters(llama_configs, parameter_name='node_stationary',     parameter_value='ws')
    workload.add_parameters(llama_configs, parameter_name='exp_mult_cycles',     parameter_value=30)
    workload.add_parameters(llama_configs, parameter_name='division_mult_cycles', parameter_value=14)
    workload.add_parameters(llama_configs, parameter_name='accumulation_cycles', parameter_value=1)
    workload.add_parameters(llama_configs, parameter_name='architecture',        parameter_value='tensor')
    workload.add_parameters(llama_configs, parameter_name='noc_tile_m',          parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_k',          parameter_value=256)
    workload.add_parameters(llama_configs, parameter_name='noc_tile_n',          parameter_value=256)

    ##############################################
    ###############    Event    ##################
    ##############################################
    # Performance model paths
    gemm      = 'zoo/mugi/designs/tensor/performance/gemm.performance.py'
    nonlinear = 'zoo/mugi/designs/tensor/performance/nonlinear.performance.py'
    llama     = 'zoo/mugi/common/performance/model/llama.performance.py'
    model     = 'zoo/mugi/common/performance/model/model_architecture.performance.py'
    memory    = 'zoo/mugi/common/performance/memory/memory.performance.py'
    router    = 'zoo/mugi/common/performance/router/router.performance.py'

    gemm_hw   = ['input_gemm', 'weight_gemm', 'array_gemm', 'vector_gemm']
    mem       = ['isram_offchip_writes', 'isram_onchip_reads',
                 'wsram_offchip_writes', 'wsram_onchip_reads',
                 'osram_offchip_writes', 'osram_offchip_reads', 'osram_onchip_writes', 'osram_onchip_reads',
                 'dram_input_reads', 'dram_weight_reads', 'dram_output_reads', 'dram_output_writes']
    router_ev = ['irouter_mapping', 'wrouter_mapping', 'orouter_mapping']
    vector    = ['multiplier_vector', 'accumulator_vector', 'mac_register_vector', 'register_vector']

    # single-node event graph maps leaves to hardware + memory; multi-node adds the routers
    gemm_leaf      = gemm_hw + mem
    nonlinear_leaf = ['gemm_nonlinear', 'vector_nonlinear'] + mem

    # software tree (4 model roots collapse into one llama_2 root)
    event.add_event(name='llama_2',    subevent=['gemm', 'nonlinear'],                     performance=llama)
    event.add_event(name='gemm',       subevent=['projection', 'attention', 'ffn', 'output'], performance=model)
    event.add_event(name='nonlinear',  subevent=['softmax', 'silu'],                       performance=model)
    event.add_event(name='projection', subevent=['proj_q', 'proj_k', 'proj_v', 'proj_a'],  performance=model)
    event.add_event(name='attention',  subevent=['qkt', 'av'],                             performance=model)
    event.add_event(name='ffn',        subevent=['proj_up', 'proj_down', 'proj_gate'],     performance=model)

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

    # hardware events
    event.add_event(name='input_gemm',      subevent=['ififo', 'input_register'],                performance=gemm)
    event.add_event(name='weight_gemm',     subevent=['wfifo', 'int_to_fp', 'weight_register'],  performance=gemm)
    event.add_event(name='array_gemm',      subevent=['multiplier', 'pe_register', 'accumulator', 'accumulator_register', 'adder', 'ofifo'], performance=gemm)
    event.add_event(name='vector_gemm',     subevent=vector,                                     performance=gemm)
    event.add_event(name='gemm_nonlinear',  subevent=['int_to_fp', 'ififo', 'wfifo', 'input_register', 'weight_register',
                                                      'multiplier', 'pe_register', 'accumulator', 'accumulator_register',
                                                      'adder', 'ofifo'],                          performance=nonlinear)
    event.add_event(name='vector_nonlinear', subevent=vector,                                    performance=nonlinear)

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
    arch_inst['single'] = [1]

    # Memory parameters (single tensor-core size today)
    sram_width, sram_depth, fifo_depth = 2048, 512, 8

    # === Sweep parameters =======================================
    # NoC parameters
    node_inst = [[1, 1], [2, 1], [2, 2]]

    # Array parameters (8x16x16 tensor core)
    arch_inst['height'] = [[16, 16]]
    arch_inst['width']  = [[8, 16]]
    arch_inst['array']  = [[8, 16, 16]]
    arch_inst['wat']    = [[8, 16, 15]]

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
    isram = architecture.add_module(name='isram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': sram_width, 'depth': sram_depth})
    wsram = architecture.add_module(name='wsram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': sram_width, 'depth': sram_depth})
    osram = architecture.add_module(name='osram', instance=arch_inst['single'], tag=['onchip', 'memory', 'node_memory'], query={'interface': 'cacti7', 'class': 'sram', 'bank': 4, 'width': sram_width, 'depth': sram_depth})

    # === Fifo modules ===========================================
    ififo = architecture.add_module(name='ififo', instance=arch_inst['width'],  tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': fifo_depth})
    wfifo = architecture.add_module(name='wfifo', instance=arch_inst['height'], tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 4,  'depth': fifo_depth})
    ofifo = architecture.add_module(name='ofifo', instance=arch_inst['width'],  tag=['onchip', 'array', 'fifo'], query={'class': 'fifo', 'width': 16, 'depth': fifo_depth})

    # ---- pe cluster ----
    input_register  = architecture.add_module(name='input_register',  instance=arch_inst['array'],  tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})
    weight_register = architecture.add_module(name='weight_register', instance=arch_inst['array'],  tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})
    multiplier      = architecture.add_module(name='multiplier',      instance=arch_inst['array'],  tag=['onchip', 'array', 'pe'], query={'class': 'multiplierbf16'})
    pe_register     = architecture.add_module(name='pe_register',     instance=arch_inst['array'],  tag=['onchip', 'array', 'pe'], query={'class': 'register', 'width': 16})
    int_to_fp       = architecture.add_module(name='int_to_fp',       instance=arch_inst['height'], tag=['onchip', 'array', 'pe'], query={'class': 'int_to_bf16'})

    # ---- accumulator tree + accumulator adder ----
    accumulator          = architecture.add_module(name='accumulator',          instance=arch_inst['wat'],   tag=['onchip', 'array', 'pe'],          query={'class': 'adderbf16'})
    accumulator_register = architecture.add_module(name='accumulator_register', instance=arch_inst['width'], tag=['onchip', 'array', 'pe'],          query={'class': 'register', 'width': 16})
    adder                = architecture.add_module(name='adder',                instance=arch_inst['width'], tag=['onchip', 'array', 'accumulator'], query={'class': 'adderbf16'})

    # ---- vector lane ----
    multiplier_vector   = architecture.add_module(name='multiplier_vector',   instance=arch_inst['width'], tag=['onchip', 'array', 'vector'], query={'class': 'multiplierbf16'})
    accumulator_vector  = architecture.add_module(name='accumulator_vector',  instance=arch_inst['width'], tag=['onchip', 'array', 'vector'], query={'class': 'adderbf16'})
    mac_register_vector = architecture.add_module(name='mac_register_vector', instance=arch_inst['width'], tag=['onchip', 'array', 'vector'], query={'class': 'register', 'width': 16})
    register_vector     = architecture.add_module(name='register_vector',     instance=arch_inst['width'], tag=['onchip', 'array', 'vector'], query={'class': 'register', 'width': 16})

    # ---- routers (mapped only by the multi-node event graph) ----
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
    # Master sweep: every node-derived architecture instance locked to one index.
    agraph.direct_constraint([
        isram['instance'],
        wsram['instance'],
        osram['instance'],
        ififo['instance'],
        wfifo['instance'],
        ofifo['instance'],
        input_register['instance'],
        weight_register['instance'],
        multiplier['instance'],
        pe_register['instance'],
        int_to_fp['instance'],
        accumulator['instance'],
        accumulator_register['instance'],
        adder['instance'],
        multiplier_vector['instance'],
        accumulator_vector['instance'],
        mac_register_vector['instance'],
        register_vector['instance'],
        irouter['instance'],
        wrouter['instance'],
        orouter['instance'],
    ])

    # network: every software leaf picks the same event graph (single-node or multi-node)
    leaf_events = list(gemm_events.values()) + list(nonlinear_events.values())
    agraph.direct_constraint([e['subevent'] for e in leaf_events])

    # routers are mapped if and only if the design is multi-node
    agraph.conditional_constraint(
        m = multiplier['instance'],
        ev = gemm_events['proj_q_prefill']['subevent'],
        condition = lambda m, ev: ('irouter_mapping' in ev) == (m[0] != 1)
    )

    # batch size sweeps only for the single-node design; multi-node fixes batch size to 8
    agraph.conditional_constraint(
        m = multiplier['instance'],
        b = list(batch_size.values())[0]['parameter'],
        condition = lambda m, b: (m[0] == 1) or (b == 8)
    )

    # seq len sweeps only for the single-node design; multi-node fixes seq len to 4096
    agraph.conditional_constraint(
        m = multiplier['instance'],
        q = list(max_seq_len.values())[0]['parameter'],
        condition = lambda m, q: (m[0] == 1) or (q == 4096)
    )

    return agraph.generate()
