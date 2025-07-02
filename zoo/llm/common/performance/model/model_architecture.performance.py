from collections import OrderedDict
from zoo.llm.common.performance.mapping.mapping_performance import mapping
from zoo.llm.common.performance.utils import sum_subevents
from loguru import logger

# region functions
def gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    layers = next(iter(workload_dict.values()))['configuration']['layers']

    projection_dict = OrderedDict({'count': layers, 'aggregation': 'sequential'})
    attention_dict = OrderedDict({'count': layers, 'aggregation': 'sequential'})
    ffn_dict = OrderedDict({'count': layers, 'aggregation': 'sequential'})
    output_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'projection': projection_dict,
        'attention': attention_dict,
        'ffn': ffn_dict,
        'output': output_dict
    })

    return performance_dict

def nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    layers = next(iter(workload_dict.values()))['configuration']['layers']

    softmax_dict = OrderedDict({'count': layers, 'aggregation': 'sequential'})
    silu_dict = OrderedDict({'count': layers, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'softmax': softmax_dict,
        'silu': silu_dict
    })

    return performance_dict
# endregion

# region layers
def projection(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_q_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_k_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_v_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_a_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    
    performance_dict['subevent'] = OrderedDict({
        'proj_q': proj_q_dict,
        'proj_k': proj_k_dict,
        'proj_v': proj_v_dict,
        'proj_a': proj_a_dict
    })

    return performance_dict

def attention(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    qkt_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    av_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'qkt': qkt_dict,
        'av': av_dict
    })

    return performance_dict

def ffn(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_up_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_gate_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_down_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_up': proj_up_dict,
        'proj_gate': proj_gate_dict,
        'proj_down': proj_down_dict
    })

    return performance_dict

def output(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    output_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    output_decode_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'output_prefill': output_prefill_dict,
        'output_decode': output_decode_dict
    })

    return performance_dict
# endregion

# region sublayers
def proj_q(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_q_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_q_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_q_prefill': proj_q_prefill_dict,
        'proj_q_decode': proj_q_decode_dict
    })

    return performance_dict

def proj_k(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_k_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_k_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_k_prefill': proj_k_prefill_dict,
        'proj_k_decode': proj_k_decode_dict
    })

    return performance_dict

def proj_v(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_v_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_v_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_v_prefill': proj_v_prefill_dict,
        'proj_v_decode': proj_v_decode_dict
    })

    return performance_dict

def proj_a(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_a_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_a_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_a_prefill': proj_a_prefill_dict,
        'proj_a_decode': proj_a_decode_dict
    })

    return performance_dict

def qkt(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    qkt_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    qkt_decode_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'qkt_prefill': qkt_prefill_dict,
        'qkt_decode': qkt_decode_dict
    })

    return performance_dict

def av(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    av_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    av_decode_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'av_prefill': av_prefill_dict,
        'av_decode': av_decode_dict
    })

    return performance_dict

def proj_up(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_up_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_up_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_up_prefill': proj_up_prefill_dict,
        'proj_up_decode': proj_up_decode_dict
    })

    return performance_dict

def proj_gate(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_gate_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_gate_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_gate_prefill': proj_gate_prefill_dict,
        'proj_gate_decode': proj_gate_decode_dict
    })

    return performance_dict

def proj_down(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    proj_down_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    proj_down_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'proj_down_prefill': proj_down_prefill_dict,
        'proj_down_decode': proj_down_decode_dict
    })

    return performance_dict

def softmax(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    softmax_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    softmax_decode_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'softmax_prefill': softmax_prefill_dict,
        'softmax_decode': softmax_decode_dict
    })

    return performance_dict

def silu(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    silu_prefill_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    silu_decode_dict = OrderedDict({'count': 4096, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'silu_prefill': silu_prefill_dict,
        'silu_decode': silu_decode_dict
    })

    return performance_dict
# endregion

# region sublayer prefill/decode
def proj_q_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, dim)
    # output_shape = (batch_size, prefill_seq_len, dim)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    
    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    
    return performance_dict

def proj_q_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, dim)
    # output_shape = (batch_size, 1, dim)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_k_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, dim * kv_heads / heads)
    # output_shape = (batch_size, prefill_seq_len, dim * kv_heads / heads)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    heads = workload_config['heads']
    kv_heads = workload_config['kv_heads']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': dim * kv_heads / heads,
    })

    performance_dict = performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_k_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, dim * kv_heads / heads)
    # output_shape = (batch_size, 1, dim * kv_heads / heads)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']
    heads = workload_config['heads']
    kv_heads = workload_config['kv_heads']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': dim * kv_heads / heads,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_v_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, dim * kv_heads / heads)
    # output_shape = (batch_size, prefill_seq_len, dim * kv_heads / heads)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    heads = workload_config['heads']
    kv_heads = workload_config['kv_heads']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': dim * kv_heads / heads,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_v_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, dim * kv_heads / heads)
    # output_shape = (batch_size, 1, dim * kv_heads / heads)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']
    heads = workload_config['heads']
    kv_heads = workload_config['kv_heads']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': dim * kv_heads / heads,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_a_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, dim)
    # output_shape = (batch_size, prefill_seq_len, dim)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_a_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['proj_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, dim)
    # output_shape = (batch_size, 1, dim)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def qkt_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['k_avg_early_termination_cycles']})

    # attn_shape = (batch_size, heads, prefill_seq_len, dim/heads)
    # weight_shape = (batch_size, heads, dim/heads, prefill_seq_len)
    # output_shape = (batch_size, heads, prefill_seq_len, prefill_seq_len)

    # with gqa
    # attn_shape = (batch_size, kv_heads, heads / kv_heads * prefill_seq_len, dim/heads)
    # weight_shape = (batch_size, kv_heads, dim/heads, prefill_seq_len)
    # output_shape = (batch_size, kv_heads, heads / kv_heads * prefill_seq_len, prefill_seq_len)

    batch_size = workload_config['batch_size']
    heads = workload_config['heads']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    kv_heads = workload_config['kv_heads']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': batch_size * kv_heads,
        'm': (heads / kv_heads) * prefill_seq_len,
        'k': dim/heads,
        'n': prefill_seq_len,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def qkt_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['k_avg_early_termination_cycles']})

    # attn_shape = (batch_size, heads, 1, dim/heads)
    # weight_shape = (batch_size, heads, dim/heads, cur_seq_len)
    # output_shape = (batch_size, heads, 1, cur_seq_len)

    # with gqa
    # attn_shape = (batch_size, kv_heads, heads / kv_heads, dim/heads)
    # weight_shape = (batch_size, kv_heads, dim/heads, cur_seq_len)
    # output_shape = (batch_size, kv_heads, heads / kv_heads, cur_seq_len)

    batch_size = workload_config['batch_size']
    heads = workload_config['heads']
    prefill_seq_len = workload_config['prefill_seq_len']
    max_seq_len = workload_config['max_seq_len']
    dim = workload_config['dim']
    kv_heads = workload_config['kv_heads']

    for cur_seq_len in range(prefill_seq_len + 1, max_seq_len + 1):
        mapping_dict = OrderedDict({
            'event': 'gemm',
            'batch': batch_size * kv_heads,
            'm': (heads / kv_heads),
            'k': dim/heads,
            'n': cur_seq_len,
        })
        if not performance_dict:
            performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
        else:
            performance_dict = sum_subevents(performance_dict, mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config))

    return performance_dict

def av_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['v_avg_early_termination_cycles']})

    # attn_shape = (batch_size, heads, 1, prefill_seq_len)
    # weight_shape = (batch_size, heads, prefill_seq_len, dim/heads)
    # output_shape = (batch_size, heads, 1, dim/heads)

    # with gqa
    # attn_shape = (batch_size, kv_heads, heads / kv_heads * prefill_seq_len, prefill_seq_len)
    # weight_shape = (batch_size, kv_heads, prefill_seq_len, dim/heads)
    # output_shape = (batch_size, kv_heads, heads / kv_heads * prefill_seq_len, dim/heads)

    batch_size = workload_config['batch_size']
    heads = workload_config['heads']
    kv_heads = workload_config['kv_heads']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': batch_size * kv_heads,
        'm': (heads / kv_heads) * prefill_seq_len,
        'k': prefill_seq_len,
        'n': dim/heads,
    })


    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def av_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']
    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['v_avg_early_termination_cycles']})

    # attn_shape = (batch_size, heads, 1, cur_seq_len)
    # weight_shape = (batch_size, heads, cur_seq_len, dim/heads)
    # output_shape = (batch_size, heads, 1, dim/heads)

    # with gqa
    # attn_shape = (batch_size, kv_heads, heads / kv_heads, cur_seq_len)
    # weight_shape = (batch_size, kv_heads, cur_seq_len, dim/heads)
    # output_shape = (batch_size, kv_heads, heads / kv_heads, dim/heads)

    batch_size = workload_config['batch_size']
    heads = workload_config['heads']
    kv_heads = workload_config['kv_heads']
    max_seq_len = workload_config['max_seq_len']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']

    for cur_seq_len in range(prefill_seq_len + 1, max_seq_len + 1):
        mapping_dict = OrderedDict({
            'event': 'gemm',
            'batch': batch_size * kv_heads,
            'm': (heads / kv_heads),
            'k': cur_seq_len,
            'n': dim/heads,
        })

        if not performance_dict:
            performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
        else:
            performance_dict = sum_subevents(performance_dict, mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config))

    return performance_dict

def proj_up_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['ffn_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, hidden_dim)
    # output_shape = (batch_size, prefill_seq_len, hidden_dim)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    hidden_dim = workload_config['hidden_dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': hidden_dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_up_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['ffn_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, hidden_dim)
    # output_shape = (batch_size, 1, hidden_dim)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']
    hidden_dim = workload_config['hidden_dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': hidden_dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_gate_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['ffn_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, hidden_dim)
    # output_shape = (batch_size, prefill_seq_len, hidden_dim)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    hidden_dim = workload_config['hidden_dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': hidden_dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_gate_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['ffn_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, hidden_dim)
    # output_shape = (batch_size, 1, hidden_dim)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']
    hidden_dim = workload_config['hidden_dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': hidden_dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_down_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['ffn_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, hidden_dim)
    # weight_shape = (hidden_dim, dim)
    # output_shape = (batch_size, prefill_seq_len, dim)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    hidden_dim = workload_config['hidden_dim']
    dim = workload_config['dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': hidden_dim,
        'n': dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def proj_down_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['ffn_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, hidden_dim)
    # weight_shape = (hidden_dim, dim)
    # output_shape = (batch_size, 1, dim)

    batch_size = workload_config['batch_size']
    hidden_dim = workload_config['hidden_dim']
    dim = workload_config['dim']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': hidden_dim,
        'n': dim,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def softmax_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['nonlinear_avg_early_termination_cycles']})

    # attn_shape = (batch_size, heads, prefill_seq_len, prefill_seq_len)
    # output_shape = (batch_size, heads, prefill_seq_len, prefill_seq_len)

    # with gqa
    # attn_shape = (batch_size, kv_heads, heads / kv_heads * prefill_seq_len, cur_seq_len)
    # output_shape = (batch_size, kv_heads, heads / kv_heads * prefill_seq_len, cur_seq_len)

    batch_size = workload_config['batch_size']
    heads = workload_config['heads']
    prefill_seq_len = workload_config['prefill_seq_len']

    mapping_dict = OrderedDict({
        'event': 'nonlinear',
        'function': 'softmax',
        'batch': 1,
        'm': batch_size * heads * prefill_seq_len,
        'n': prefill_seq_len,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def softmax_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['nonlinear_avg_early_termination_cycles']})

    # attn_shape = (batch_size, heads, 1, cur_seq_len)
    # output_shape = (batch_size, heads, 1, cur_seq_len)

    # with gqa
    # attn_shape = (batch_size, kv_heads, heads / kv_heads, cur_seq_len)
    # output_shape = (batch_size, kv_heads, heads / kv_heads, cur_seq_len)

    batch_size = workload_config['batch_size']
    kv_heads = workload_config['kv_heads']
    heads = workload_config['heads']
    max_seq_len = workload_config['max_seq_len']
    prefill_seq_len = workload_config['prefill_seq_len']

    test = batch_size * heads * prefill_seq_len
    for cur_seq_len in range(prefill_seq_len + 1, max_seq_len + 1):
        test += batch_size * heads * cur_seq_len
        mapping_dict = OrderedDict({
            'event': 'nonlinear',
            'function': 'softmax',
            'batch': 1,
            'm': batch_size * heads,
            'n': cur_seq_len,
        })

        if not performance_dict:
            performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
        else:
            performance_dict = sum_subevents(performance_dict, mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config))

    return performance_dict

def silu_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['nonlinear_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, hidden_dim)
    # output_shape = (batch_size, prefill_seq_len, hidden_dim)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    hidden_dim = workload_config['hidden_dim']

    mapping_dict = OrderedDict({
        'event': 'nonlinear',
        'function': 'silu',
        'batch': 1,
        'm': hidden_dim,
        'n': batch_size * prefill_seq_len,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def silu_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['nonlinear_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, hidden_dim)
    # output_shape = (batch_size, 1, hidden_dim)

    batch_size = workload_config['batch_size']
    hidden_dim = workload_config['hidden_dim']

    mapping_dict = OrderedDict({
        'event': 'nonlinear',
        'function': 'silu',
        'batch': 1,
        'm': hidden_dim,
        'n': batch_size,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def output_prefill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['default_avg_early_termination_cycles']})

    # attn_shape = (batch_size, prefill_seq_len, dim)
    # weight_shape = (dim, vocab_size)
    # output_shape = (batch_size, prefill_seq_len, vocab_size)

    batch_size = workload_config['batch_size']
    prefill_seq_len = workload_config['prefill_seq_len']
    dim = workload_config['dim']
    vocab_size = workload_config['vocab_size']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size * prefill_seq_len,
        'k': dim,
        'n': vocab_size,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict

def output_decode(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    workload_config = next(iter(workload_dict.values()))['configuration']

    if workload_config['architecture'].lower() == 'mugi':
        workload_config.update({'early_termination_cycles': workload_config['default_avg_early_termination_cycles']})

    # attn_shape = (batch_size, 1, dim)
    # weight_shape = (dim, vocab_size)
    # output_shape = (batch_size, 1, vocab_size)

    batch_size = workload_config['batch_size']
    dim = workload_config['dim']
    vocab_size = workload_config['vocab_size']

    mapping_dict = OrderedDict({
        'event': 'gemm',
        'batch': 1,
        'm': batch_size,
        'k': dim,
        'n': vocab_size,
    })

    performance_dict = mapping(mapping_dict=mapping_dict, architecture_dict=architecture_dict, workload_dict=workload_config)
    return performance_dict
# endregion