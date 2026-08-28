from collections import OrderedDict
from chiplet4ai.common.performance import mapping as common_mapping


LAYER_EVENTS_PF = ['proj_q_pf', 'proj_k_pf', 'proj_v_pf', 'qkt_pf', 'av_pf', 'a_proj_pf', 'up_proj_pf', 'gate_proj_pf', 'down_proj_pf']
LAYER_EVENTS_DC = ['proj_q_dc', 'proj_k_dc', 'proj_v_dc', 'qkt_dc', 'av_dc', 'a_proj_dc', 'up_proj_dc', 'gate_proj_dc', 'down_proj_dc']

# The attention GEMMs are batched: one GEMM per (batch, kv head), each against its own
# slice of the KV cache. mapping.py describes a single GEMM, so the batch scales every
# count -- and with it the cycles and the energy, since the engine multiplies a child's
# value by the edge count. Folding the batch into M instead would be wrong: that would
# stream every batch element past ONE stationary weight tile and grant a weight reuse
# the hardware never gets.
def _batched(performance_dict: OrderedDict, batch: int) -> OrderedDict:
    for subevent in performance_dict['subevent'].values():
        subevent['count'] *= batch
    return performance_dict

# FUSED ATTENTION. qkt produces the score matrix and av consumes it immediately; it is an
# INTERMEDIATE, held on chip between the two GEMMs and never spilled to DRAM -- that is
# what every fused attention implementation does. mapping.py cannot know this, because it
# costs one GEMM at a time and has no idea whose output feeds whose input, so the pairing
# is applied here where the dataflow is known.
#
# Only the DRAM lanes are cleared. The matrix still moves on chip -- out of qkt's osram
# and into av's isram -- and those SRAM lanes stay charged.
#
# The effect is not marginal: at 128k context the score matrix is ~99% of both the ifmap
# and the ofmap DRAM traffic, and being one array tile wide it can never respond to SRAM
# capacity, so it flattens every curve it dominates.
def _onchip_output(performance_dict: OrderedDict) -> OrderedDict:
    for lane in ('dram_output_write', 'dram_output_read'):
        performance_dict['subevent'][lane]['count'] = 0
    return performance_dict

def _onchip_input(performance_dict: OrderedDict) -> OrderedDict:
    performance_dict['subevent']['dram_input_read']['count'] = 0
    return performance_dict

# qkt and av grow with the sequence, so their leaf functions walk every decode step
# internally (array_mapping_decode and friends) and already cover the whole decode
# phase. They are therefore charged ONCE per layer, while every other decode GEMM keeps
# its fixed shape and is charged once per decode step.
DECODE_STEP_EVENTS = [event for event in LAYER_EVENTS_DC if event not in ('qkt_dc', 'av_dc')]

# deepseek routes its decode layers through layer_dc_moe, where only the FFN GEMMs run
# once per activated expert; attention is computed once and shared across them
MOE_EXPERT_EVENTS = ['up_proj_dc', 'gate_proj_dc', 'down_proj_dc']

def _decode_gemm_counts(cfg: OrderedDict) -> OrderedDict:
    """Per-layer decode multiplicity of every '_dc' GEMM.

    ONE SOURCE FOR BOTH VIEWS. 'llama' reaches these GEMMs through layer_dc_moe and
    'llama_array' reaches their '_arr' nodes through llama_dc_array. The two views
    describe the same computation, so they must agree GEMM for GEMM -- fig_1 reads the
    array view's cycles and fig_3 divides the 'llama' view's MAC count by them, and any
    disagreement shows up there as an impossible utilization. Reading both from here is
    what keeps them in step.
    """
    steps = cfg['max_seq_len'] - cfg['prefill_seq_len']
    # every event simulates under every workload; dense llama configs lack the MoE keys,
    # so they default to one activated expert, which reproduces layer_dc exactly
    act = cfg.get('experts_per_tok', 1) + cfg.get('n_shared_experts', 0)

    counts = OrderedDict()
    for event in LAYER_EVENTS_DC:
        count = steps if event in DECODE_STEP_EVENTS else 1
        # only the FFN runs per activated expert; attention is shared across them
        if event in MOE_EXPERT_EVENTS:
            count *= act
        counts[event] = count

    return counts

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

    # Same multiplicities layer_dc/layer_dc_moe charge on the 'llama' side, including the
    # MoE expert factor on the FFN: qkt/av already walk every decode step inside their own
    # mapping, and the FFN really does run once per activated expert. Charging them here
    # without that factor would leave this view short of the array work it is meant to
    # describe.
    for event, count in _decode_gemm_counts(cfg).items():
        performance_dict['subevent'][f'{event}_arr'] = OrderedDict({'count': count, 'aggregation': 'sequential'})

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
    performance_dict = OrderedDict({'subevent': OrderedDict()})
    cfg = workload_dict['configuration']
    steps = cfg['max_seq_len'] - cfg['prefill_seq_len']

    for event in LAYER_EVENTS_DC:
        count = steps if event in DECODE_STEP_EVENTS else 1
        performance_dict['subevent'][event] = OrderedDict({'count': count, 'aggregation': 'sequential'})

    return performance_dict

def layer_dc_moe(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    performance_dict = OrderedDict({'subevent': OrderedDict()})
    cfg = workload_dict['configuration']

    for event, count in _decode_gemm_counts(cfg).items():
        performance_dict['subevent'][event] = OrderedDict({'count': count, 'aggregation': 'sequential'})

    return performance_dict


def proj_q_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_q_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_q_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def proj_k_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_k_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_k_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_k_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def proj_v_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_v_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_v_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_v_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def qkt_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'qkt_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'qkt_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'qkt_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def av_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'av_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'av_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'av_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def a_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'a_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'a_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'a_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def up_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'up_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'up_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'up_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def gate_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'gate_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'gate_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'gate_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def down_proj_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'down_proj_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'down_proj_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'down_proj_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def lm_head_pf(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'lm_head_pf_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'lm_head_pf_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'lm_head_pf_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def proj_q_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_q_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_q_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_q_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def proj_k_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_k_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_k_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_k_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def proj_v_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'proj_v_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_v_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'proj_v_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def qkt_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'qkt_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'qkt_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'qkt_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def av_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'av_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'av_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'av_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def a_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'a_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'a_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'a_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def up_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'up_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'up_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'up_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def gate_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'gate_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'gate_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'gate_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def down_proj_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'down_proj_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'down_proj_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'down_proj_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def lm_head_dc(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'lm_head_dc_arr': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'lm_head_dc_sram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
        'lm_head_dc_dram': OrderedDict({'count': 1, 'aggregation': 'parallel'}),
    })})


def proj_q_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def proj_q_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def proj_q_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def proj_k_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_k_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_k_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_v_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_v_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_v_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def qkt_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return _batched(common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'],
        'K': cfg['dim'] // cfg['heads'],
        'N': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads'])


def qkt_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return _batched(common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'],
        'K': cfg['dim'] // cfg['heads'],
        'N': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads'])


def qkt_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return _onchip_output(_batched(common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'],
        'K': cfg['dim'] // cfg['heads'],
        'N': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads']))


def av_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return _batched(common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'],
        'K': cfg['prefill_seq_len'],
        'N': cfg['dim'] // cfg['heads']})), cfg['batch_size'] * cfg['kv_heads'])


def av_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return _batched(common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'],
        'K': cfg['prefill_seq_len'],
        'N': cfg['dim'] // cfg['heads']})), cfg['batch_size'] * cfg['kv_heads'])


def av_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return _onchip_input(_batched(common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['prefill_seq_len'] * cfg['heads'] // cfg['kv_heads'],
        'K': cfg['prefill_seq_len'],
        'N': cfg['dim'] // cfg['heads']})), cfg['batch_size'] * cfg['kv_heads']))


def a_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def a_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def a_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def up_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def up_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def up_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def gate_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def gate_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def gate_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def down_proj_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['hidden_dim'],
        'N': cfg['dim']}))


def down_proj_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['hidden_dim'],
        'N': cfg['dim']}))


def down_proj_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['hidden_dim'],
        'N': cfg['dim']}))


def lm_head_pf_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['vocab_size']}))


def lm_head_pf_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['vocab_size']}))


def lm_head_pf_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'] * cfg['prefill_seq_len'],
        'K': cfg['dim'],
        'N': cfg['vocab_size']}))


def proj_q_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def proj_q_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def proj_q_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def proj_k_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_k_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_k_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_v_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_v_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def proj_v_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim'] * cfg['kv_heads'] // cfg['heads']}))


def qkt_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    tokens_per_step = cfg.get('tokens_per_step', 1)
    token_compression = cfg.get('token_compression', False)
    head_dim = cfg.get('head_dim', None)
    return _batched(common_mapping.array_mapping_decode('n', tokens_per_step, architecture_dict, OrderedDict({
        'M': tokens_per_step * cfg['heads'] // cfg['kv_heads'],
        'K': head_dim if head_dim is not None else cfg['dim'] // cfg['heads'],
        'N': ((cfg['max_seq_len'] - 128) / 4) + 128 if token_compression else cfg['max_seq_len'],
        'step_start': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads'])


def qkt_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    tokens_per_step = cfg.get('tokens_per_step', 1)
    token_compression = cfg.get('token_compression', False)
    head_dim = cfg.get('head_dim', None)
    return _batched(common_mapping.sram_mapping_decode('n', tokens_per_step, architecture_dict, OrderedDict({
        'M': tokens_per_step * cfg['heads'] // cfg['kv_heads'],
        'K': head_dim if head_dim is not None else cfg['dim'] // cfg['heads'],
        'N': ((cfg['max_seq_len'] - 128) / 4) + 128 if token_compression else cfg['max_seq_len'],
        'step_start': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads'])


def qkt_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    tokens_per_step = cfg.get('tokens_per_step', 1)
    token_compression = cfg.get('token_compression', False)
    head_dim = cfg.get('head_dim', None)
    return _onchip_output(_batched(common_mapping.dram_mapping_decode('n', tokens_per_step, architecture_dict, OrderedDict({
        'M': tokens_per_step * cfg['heads'] // cfg['kv_heads'],
        'K': head_dim if head_dim is not None else cfg['dim'] // cfg['heads'],
        'N': ((cfg['max_seq_len'] - 128) / 4) + 128 if token_compression else cfg['max_seq_len'],
        'step_start': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads']))


def av_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    tokens_per_step = cfg.get('tokens_per_step', 1)
    token_compression = cfg.get('token_compression', False)
    head_dim = cfg.get('head_dim', None)
    return _batched(common_mapping.array_mapping_decode('k', tokens_per_step, architecture_dict, OrderedDict({
        'M': tokens_per_step * cfg['heads'] // cfg['kv_heads'],
        'K': ((cfg['max_seq_len'] - 128) / 4) + 128 if token_compression else cfg['max_seq_len'],
        'N': head_dim if head_dim is not None else cfg['dim'] // cfg['heads'],
        'step_start': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads'])


def av_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    tokens_per_step = cfg.get('tokens_per_step', 1)
    token_compression = cfg.get('token_compression', False)
    head_dim = cfg.get('head_dim', None)
    return _batched(common_mapping.sram_mapping_decode('k', tokens_per_step, architecture_dict, OrderedDict({
        'M': tokens_per_step * cfg['heads'] // cfg['kv_heads'],
        'K': ((cfg['max_seq_len'] - 128) / 4) + 128 if token_compression else cfg['max_seq_len'],
        'N': head_dim if head_dim is not None else cfg['dim'] // cfg['heads'],
        'step_start': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads'])


def av_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    tokens_per_step = cfg.get('tokens_per_step', 1)
    token_compression = cfg.get('token_compression', False)
    head_dim = cfg.get('head_dim', None)
    return _onchip_input(_batched(common_mapping.dram_mapping_decode('k', tokens_per_step, architecture_dict, OrderedDict({
        'M': tokens_per_step * cfg['heads'] // cfg['kv_heads'],
        'K': ((cfg['max_seq_len'] - 128) / 4) + 128 if token_compression else cfg['max_seq_len'],
        'N': head_dim if head_dim is not None else cfg['dim'] // cfg['heads'],
        'step_start': cfg['prefill_seq_len']})), cfg['batch_size'] * cfg['kv_heads']))


def a_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def a_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def a_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['dim']}))


def up_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def up_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def up_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def gate_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def gate_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def gate_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['hidden_dim']}))


def down_proj_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['hidden_dim'],
        'N': cfg['dim']}))


def down_proj_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['hidden_dim'],
        'N': cfg['dim']}))


def down_proj_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['hidden_dim'],
        'N': cfg['dim']}))


def lm_head_dc_arr(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.array_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['vocab_size']}))


def lm_head_dc_sram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.sram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['vocab_size']}))


def lm_head_dc_dram(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    cfg = workload_dict['configuration']
    return common_mapping.dram_mapping(architecture_dict, OrderedDict({
        'M': cfg['batch_size'],
        'K': cfg['dim'],
        'N': cfg['vocab_size']}))

