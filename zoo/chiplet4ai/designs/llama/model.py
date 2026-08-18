from collections import OrderedDict
from chiplet4ai.common.performance import mapping as common_mapping


LAYER_EVENTS_PF = ['proj_q_pf', 'proj_k_pf', 'proj_v_pf', 'qkt_pf', 'av_pf', 'a_proj_pf', 'up_proj_pf', 'gate_proj_pf', 'down_proj_pf']
LAYER_EVENTS_DC = ['proj_q_dc', 'proj_k_dc', 'proj_v_dc', 'qkt_dc', 'av_dc', 'a_proj_dc', 'up_proj_dc', 'gate_proj_dc', 'down_proj_dc']

def llama_2_7b(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)


def llama_2_13b(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)


def llama_2_70b(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)


def llama_2_70b_GQA(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)


def llama_3_1_70b(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)

def llama_3_1_8b(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)

def llama_3_1_405b(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)

def deepseek_v4(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return llama_model(architecture_dict=architecture_dict, workload_dict=workload_dict)

def llama_model(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    # 'llama' and 'llama_array' are two views of one computation, not two
    # computations: 'llama' charges array + SRAM + DRAM per GEMM, 'llama_array'
    # charges the array nodes of the same GEMMs with the same multiplicities.
    # Both edges are 'parallel' so the root takes max(llama, llama_array)
    # instead of their sum (the engine computes sequential_acc + parallel_max,
    # so a single sequential edge would still add the second view). The two
    # views share the '*_arr' subtree, and 'llama' adds only non-negative
    # '*_sram'/'*_dram' terms on top, so the maximum is 'llama' by
    # construction: the root aggregates the computation exactly once, and both
    # views stay individually queryable with their own numbers unchanged.
    # For 'dynamic_energy' (summation aggregation, which ignores edge
    # aggregation and sums over every workload->module path) the array view is
    # zero-weighted instead, so its energy is counted once via 'llama'.
    return OrderedDict({'subevent': OrderedDict({
        'llama': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'llama_array': OrderedDict({'count': 1, 'aggregation': 'parallel',
                                    'factor': {'dynamic_energy': 0.0}}),
    })})

def llama(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'prefill': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'decode': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def llama_array(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return OrderedDict({'subevent': OrderedDict({
        'llama_pf_array': OrderedDict({'count': cfg['layers'], 'aggregation': 'sequential'}),
        'llama_dc_array': OrderedDict({'count': cfg['layers'], 'aggregation': 'sequential'}),
        'lm_head_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'lm_head_dc_arr': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
    })})

def llama_pf_array(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    performance_dict = OrderedDict({'subevent': OrderedDict()})
    cfg = workload_dict['configuration']

    for event in LAYER_EVENTS_PF:
        performance_dict['subevent'][f'{event}_arr'] = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    
    return performance_dict

def llama_dc_array(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    performance_dict = OrderedDict({'subevent': OrderedDict()})
    cfg = workload_dict['configuration']

    for event in LAYER_EVENTS_DC:
        performance_dict['subevent'][f'{event}_arr'] = OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'})

    return performance_dict

def prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']

    return OrderedDict({'subevent': OrderedDict({
        'layer_pf': OrderedDict({'count': cfg['layers'], 'aggregation': 'sequential'}),
        'lm_head_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def decode(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']

    # per-run workload routing: MoE workloads (deepseek) charge layer_dc_moe and
    # dense llama charges layer_dc; the event graph carries both children, so the
    # unused variant must still be returned, at count 0 (out-neighbors are mandatory)
    moe = 'experts_per_tok' in cfg
    return OrderedDict({'subevent': OrderedDict({
        'layer_dc': OrderedDict({'count': 0 if moe else cfg['layers'], 'aggregation': 'sequential'}),
        'layer_dc_moe': OrderedDict({'count': cfg['layers'] if moe else 0, 'aggregation': 'sequential'}),
        # the lm_head GEMM runs once per decode step, same step count as every
        # other '_dc' branch (layer_dc's GEMMs, llama_dc_array, lm_head_dc_arr)
        'lm_head_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
    })})


def layer_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_k_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_v_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'qkt_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'av_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'a_proj_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'up_proj_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'gate_proj_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'down_proj_pf': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def layer_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'proj_k_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'proj_v_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'qkt_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'av_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'a_proj_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'up_proj_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'gate_proj_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'down_proj_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
    })})

def layer_dc_moe(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    # every event simulates under every workload; dense llama configs lack the MoE
    # keys, so default to one activated expert (== layer_dc, and charged at count 0)
    act = cfg.get('experts_per_tok', 1) + cfg.get('n_shared_experts', 0)
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'proj_k_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'proj_v_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'qkt_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'av_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'a_proj_dc': OrderedDict({'count': cfg['max_seq_len'] - cfg['prefill_seq_len'], 'aggregation': 'sequential'}),
        'up_proj_dc': OrderedDict({'count': (cfg['max_seq_len'] - cfg['prefill_seq_len']) * act, 'aggregation': 'sequential'}),
        'gate_proj_dc': OrderedDict({'count': (cfg['max_seq_len'] - cfg['prefill_seq_len']) * act, 'aggregation': 'sequential'}),
        'down_proj_dc': OrderedDict({'count': (cfg['max_seq_len'] - cfg['prefill_seq_len']) * act, 'aggregation': 'sequential'}),
    })})


def proj_q_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_q_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_q_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def proj_k_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_k_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_k_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_k_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def proj_v_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_v_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_v_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_v_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def qkt_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'qkt_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'qkt_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'qkt_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def av_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'av_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'av_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'av_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def a_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'a_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'a_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'a_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def up_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'up_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'up_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'up_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def gate_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'gate_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'gate_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'gate_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def down_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'down_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'down_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'down_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def lm_head_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'lm_head_pf_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'lm_head_pf_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'lm_head_pf_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def proj_q_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_q_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_q_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def proj_k_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_k_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_k_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_k_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def proj_v_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_v_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_v_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'proj_v_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def qkt_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'qkt_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'qkt_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'qkt_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def av_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'av_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'av_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'av_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def a_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'a_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'a_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'a_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def up_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'up_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'up_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'up_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def gate_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'gate_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'gate_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'gate_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def down_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'down_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'down_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'down_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def lm_head_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'lm_head_dc_arr': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'lm_head_dc_sram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
        'lm_head_dc_dram': OrderedDict({'count': 1, 'aggregation': 'sequential'}),
    })})


def proj_q_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'], 0, None)


def proj_q_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'], 0, None)


def proj_q_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'], 0, None)


def proj_k_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_k_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_k_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_v_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_v_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_v_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def qkt_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'], cfg['dim'] // cfg['heads'], cfg['prefill_seq_len'], 0, None)


def qkt_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'], cfg['dim'] // cfg['heads'], cfg['prefill_seq_len'], 0, None)


def qkt_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'], cfg['dim'] // cfg['heads'], cfg['prefill_seq_len'], 0, None)


def av_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'], cfg['prefill_seq_len'], cfg['dim'] // cfg['heads'], 0, None)


def av_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'], cfg['prefill_seq_len'], cfg['dim'] // cfg['heads'], 0, None)


def av_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'], cfg['prefill_seq_len'], cfg['dim'] // cfg['heads'], 0, None)


def a_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'], 0, None)


def a_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'], 0, None)


def a_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['dim'], 0, None)


def up_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['hidden_dim'], 0, None)


def up_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['hidden_dim'], 0, None)


def up_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['hidden_dim'], 0, None)


def gate_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['hidden_dim'], 0, None)


def gate_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['hidden_dim'], 0, None)


def gate_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['hidden_dim'], 0, None)


def down_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['hidden_dim'], cfg['dim'], 0, None)


def down_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['hidden_dim'], cfg['dim'], 0, None)


def down_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['hidden_dim'], cfg['dim'], 0, None)


def lm_head_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['vocab_size'], 0, None)


def lm_head_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['vocab_size'], 0, None)


def lm_head_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'] * cfg['prefill_seq_len'], cfg['dim'], cfg['vocab_size'], 0, None)


def proj_q_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'], 0, None)


def proj_q_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'], 0, None)


def proj_q_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'], 0, None)


def proj_k_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_k_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_k_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_v_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_v_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def proj_v_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'] * cfg['kv_heads'] // cfg['heads'], 0, None)


def qkt_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['heads'] // cfg['kv_heads'], cfg['dim'] // cfg['heads'], cfg['max_seq_len'], cfg['prefill_seq_len'], 'n')


def qkt_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['heads'] // cfg['kv_heads'], cfg['dim'] // cfg['heads'], cfg['max_seq_len'], cfg['prefill_seq_len'], 'n')


def qkt_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['heads'] // cfg['kv_heads'], cfg['dim'] // cfg['heads'], cfg['max_seq_len'], cfg['prefill_seq_len'], 'n')


def av_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['heads'] // cfg['kv_heads'], cfg['max_seq_len'], cfg['dim'] // cfg['heads'], cfg['prefill_seq_len'], 'k')


def av_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['heads'] // cfg['kv_heads'], cfg['max_seq_len'], cfg['dim'] // cfg['heads'], cfg['prefill_seq_len'], 'k')


def av_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, cfg['batch_size'] * cfg['kv_heads'], cfg['heads'] // cfg['kv_heads'], cfg['max_seq_len'], cfg['dim'] // cfg['heads'], cfg['prefill_seq_len'], 'k')


def a_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'], 0, None)


def a_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'], 0, None)


def a_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['dim'], 0, None)


def up_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['hidden_dim'], 0, None)


def up_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['hidden_dim'], 0, None)


def up_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['hidden_dim'], 0, None)


def gate_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['hidden_dim'], 0, None)


def gate_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['hidden_dim'], 0, None)


def gate_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['hidden_dim'], 0, None)


def down_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['hidden_dim'], cfg['dim'], 0, None)


def down_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['hidden_dim'], cfg['dim'], 0, None)


def down_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['hidden_dim'], cfg['dim'], 0, None)


def lm_head_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.gemm(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['vocab_size'], 0, None)


def lm_head_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['vocab_size'], 0, None)


def lm_head_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram(architecture_dict, 1, cfg['batch_size'], cfg['dim'], cfg['vocab_size'], 0, None)
