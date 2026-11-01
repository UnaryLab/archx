from collections import OrderedDict

from archx.utils import get_prod

from zoo.agraph.designs.mugi.mapping import (
    gemm_mapping,
    nonlinear_mapping,
    sum_subevents,
)


def _router_dim(architecture_dict):
    # routers exist only in multi-node architectures; work splits across the nodes
    return (
        get_prod(architecture_dict["irouter"]["instance"])
        if "irouter" in architecture_dict
        else 1
    )


def _performance(cycle_count, frequency, **subevent_counts):
    return OrderedDict(
        {
            "cycle_count": OrderedDict({"value": cycle_count, "unit": "cycle"}),
            "runtime": OrderedDict(
                {"value": cycle_count / 1000 / frequency, "unit": "ms"}
            ),
            "subevent": OrderedDict(
                {
                    name: OrderedDict({"count": count})
                    for name, count in subevent_counts.items()
                }
            ),
        }
    )


def _sequential(**counts):
    return OrderedDict(
        {
            "subevent": OrderedDict(
                {
                    name: OrderedDict({"count": count, "aggregation": "sequential"})
                    for name, count in counts.items()
                }
            )
        }
    )


# region model
def llama_2(architecture_dict, workload_dict=None):
    return _sequential(gemm=1, nonlinear=1)


def gemm(architecture_dict, workload_dict=None):
    layers = workload_dict["configuration"]["layers"]
    return _sequential(projection=layers, attention=layers, ffn=layers, output=1)


def nonlinear(architecture_dict, workload_dict=None):
    layers = workload_dict["configuration"]["layers"]
    return _sequential(softmax=layers, silu=layers)


def projection(architecture_dict, workload_dict=None):
    return _sequential(proj_q=1, proj_k=1, proj_v=1, proj_a=1)


def attention(architecture_dict, workload_dict=None):
    return _sequential(qkt=1, av=1)


def ffn(architecture_dict, workload_dict=None):
    return _sequential(proj_up=1, proj_gate=1, proj_down=1)


def output(architecture_dict, workload_dict=None):
    return _sequential(output_prefill=1, output_decode=1)


# quirk: decode counts are fixed at 4096 tokens, not max_seq_len - prefill_seq_len (qkt, av and softmax decode loop over the real range)
DECODE_TOKENS = 4096


def proj_q(architecture_dict, workload_dict=None):
    return _sequential(proj_q_prefill=1, proj_q_decode=DECODE_TOKENS)


def proj_k(architecture_dict, workload_dict=None):
    return _sequential(proj_k_prefill=1, proj_k_decode=DECODE_TOKENS)


def proj_v(architecture_dict, workload_dict=None):
    return _sequential(proj_v_prefill=1, proj_v_decode=DECODE_TOKENS)


def proj_a(architecture_dict, workload_dict=None):
    return _sequential(proj_a_prefill=1, proj_a_decode=DECODE_TOKENS)


def qkt(architecture_dict, workload_dict=None):
    return _sequential(qkt_prefill=1, qkt_decode=1)


def av(architecture_dict, workload_dict=None):
    return _sequential(av_prefill=1, av_decode=1)


def proj_up(architecture_dict, workload_dict=None):
    return _sequential(proj_up_prefill=1, proj_up_decode=DECODE_TOKENS)


def proj_gate(architecture_dict, workload_dict=None):
    return _sequential(proj_gate_prefill=1, proj_gate_decode=DECODE_TOKENS)


def proj_down(architecture_dict, workload_dict=None):
    return _sequential(proj_down_prefill=1, proj_down_decode=DECODE_TOKENS)


def softmax(architecture_dict, workload_dict=None):
    return _sequential(softmax_prefill=1, softmax_decode=1)


def silu(architecture_dict, workload_dict=None):
    return _sequential(silu_prefill=1, silu_decode=DECODE_TOKENS)


# endregion


# region prefill/decode layers
# prefill runs the prompt (batch_size x prefill_seq_len tokens) at once, decode one token per batch entry;
# GEMM shapes are (m x k) @ (k x n), with grouped-query attention sharing each kv head across heads / kv_heads heads
def _decode_steps(w, step):
    """Sum of step(cur_seq_len) over every decode position."""
    total = None
    for cur_seq_len in range(w["prefill_seq_len"] + 1, w["max_seq_len"] + 1):
        events = step(cur_seq_len)
        total = events if total is None else sum_subevents(total, events)
    return total


def proj_q_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    # quirk: no early termination here, unlike the other projections (a misspelled key in the original model); equal while proj_avg_early_termination_cycles == cycles
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        w["dim"],
        architecture_dict,
        w,
        early_termination_cycles=None,
    )


def proj_q_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        w["dim"],
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def proj_k_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    n = w["dim"] * w["kv_heads"] / w["heads"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        n,
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def proj_k_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    n = w["dim"] * w["kv_heads"] / w["heads"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        n,
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def proj_v_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    n = w["dim"] * w["kv_heads"] / w["heads"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        n,
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def proj_v_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    n = w["dim"] * w["kv_heads"] / w["heads"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        n,
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def proj_a_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        w["dim"],
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def proj_a_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        w["dim"],
        architecture_dict,
        w,
        w["proj_avg_early_termination_cycles"],
    )


def qkt_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        batch=w["batch_size"] * w["kv_heads"],
        m=(w["heads"] / w["kv_heads"]) * w["prefill_seq_len"],
        k=w["dim"] / w["heads"],
        n=w["prefill_seq_len"],
        architecture_dict=architecture_dict,
        workload=w,
        early_termination_cycles=w["k_avg_early_termination_cycles"],
    )


def qkt_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return _decode_steps(
        w,
        lambda cur_seq_len: gemm_mapping(
            batch=w["batch_size"] * w["kv_heads"],
            m=(w["heads"] / w["kv_heads"]),
            k=w["dim"] / w["heads"],
            n=cur_seq_len,
            architecture_dict=architecture_dict,
            workload=w,
            early_termination_cycles=w["k_avg_early_termination_cycles"],
        ),
    )


def av_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        batch=w["batch_size"] * w["kv_heads"],
        m=(w["heads"] / w["kv_heads"]) * w["prefill_seq_len"],
        k=w["prefill_seq_len"],
        n=w["dim"] / w["heads"],
        architecture_dict=architecture_dict,
        workload=w,
        early_termination_cycles=w["v_avg_early_termination_cycles"],
    )


def av_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return _decode_steps(
        w,
        lambda cur_seq_len: gemm_mapping(
            batch=w["batch_size"] * w["kv_heads"],
            m=(w["heads"] / w["kv_heads"]),
            k=cur_seq_len,
            n=w["dim"] / w["heads"],
            architecture_dict=architecture_dict,
            workload=w,
            early_termination_cycles=w["v_avg_early_termination_cycles"],
        ),
    )


def proj_up_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        w["hidden_dim"],
        architecture_dict,
        w,
        w["ffn_avg_early_termination_cycles"],
    )


def proj_up_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        w["hidden_dim"],
        architecture_dict,
        w,
        w["ffn_avg_early_termination_cycles"],
    )


def proj_gate_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        w["hidden_dim"],
        architecture_dict,
        w,
        w["ffn_avg_early_termination_cycles"],
    )


def proj_gate_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        w["hidden_dim"],
        architecture_dict,
        w,
        w["ffn_avg_early_termination_cycles"],
    )


def proj_down_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["hidden_dim"],
        w["dim"],
        architecture_dict,
        w,
        w["ffn_avg_early_termination_cycles"],
    )


def proj_down_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["hidden_dim"],
        w["dim"],
        architecture_dict,
        w,
        w["ffn_avg_early_termination_cycles"],
    )


def softmax_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return nonlinear_mapping(
        "softmax",
        1,
        w["batch_size"] * w["heads"] * w["prefill_seq_len"],
        w["prefill_seq_len"],
        architecture_dict,
        w,
    )


def softmax_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return _decode_steps(
        w,
        lambda cur_seq_len: nonlinear_mapping(
            "softmax",
            1,
            w["batch_size"] * w["heads"],
            cur_seq_len,
            architecture_dict,
            w,
        ),
    )


def silu_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return nonlinear_mapping(
        "silu",
        1,
        w["hidden_dim"],
        w["batch_size"] * w["prefill_seq_len"],
        architecture_dict,
        w,
    )


def silu_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return nonlinear_mapping(
        "silu", 1, w["hidden_dim"], w["batch_size"], architecture_dict, w
    )


def output_prefill(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"] * w["prefill_seq_len"],
        w["dim"],
        w["vocab_size"],
        architecture_dict,
        w,
        w["default_avg_early_termination_cycles"],
    )


def output_decode(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    return gemm_mapping(
        1,
        w["batch_size"],
        w["dim"],
        w["vocab_size"],
        architecture_dict,
        w,
        w["default_avg_early_termination_cycles"],
    )


# endregion


# region hardware: shared by gemm and nonlinear
def instruction(architecture_dict, workload_dict=None):
    return _performance(
        1 / _router_dim(architecture_dict),
        architecture_dict["imux"]["query"]["frequency"],
        imux=architecture_dict["imux"]["instance"][-1],
        wmux=architecture_dict["wmux"]["instance"][-1],
    )


def counter_reuse(architecture_dict, workload_dict=None):
    return _performance(
        workload_dict["configuration"]["cycles"] / _router_dim(architecture_dict),
        architecture_dict["counter"]["query"]["frequency"],
        counter=architecture_dict["counter"]["instance"][-1],
        counter_register=get_prod(
            architecture_dict["counter_register"]["instance"][-2:]
        ),
    )


def vector(architecture_dict, workload_dict=None):
    return _performance(
        1 / _router_dim(architecture_dict),
        architecture_dict["multiplier"]["query"]["frequency"],
        multiplier_vector=architecture_dict["multiplier_vector"]["instance"][-1],
        register_vector=architecture_dict["register_vector"]["instance"][-1],
    )


# endregion


# region hardware: gemm
def input_gemm(architecture_dict, workload_dict=None):
    return _performance(
        1 / _router_dim(architecture_dict),
        architecture_dict["ififo"]["query"]["frequency"],
        ififo=architecture_dict["ififo"]["instance"][-1],
    )


def input_reuse_gemm(architecture_dict, workload_dict=None):
    cycle_count = workload_dict["configuration"]["cycles"] / _router_dim(
        architecture_dict
    )
    return _performance(
        cycle_count,
        architecture_dict["multiplier"]["query"]["frequency"],
        multiplier=architecture_dict["multiplier"]["instance"][-1] * cycle_count,
        multiplier_register=get_prod(
            architecture_dict["multiplier_register"]["instance"][-2:]
        )
        * cycle_count,
    )


def weight_gemm(architecture_dict, workload_dict=None):
    return _performance(
        1 / _router_dim(architecture_dict),
        architecture_dict["wfifo"]["query"]["frequency"],
        wfifo=architecture_dict["wfifo"]["instance"][-1],
        magnitude_register=architecture_dict["magnitude_register"]["instance"][-1],
        sign_register=architecture_dict["sign_register"]["instance"][-1],
        sign_fifo=architecture_dict["sign_fifo"]["instance"][-1],
    )


def weight_reuse_gemm(architecture_dict, workload_dict=None):
    return _performance(
        1 / _router_dim(architecture_dict),
        architecture_dict["comparator"]["query"]["frequency"],
        comparator=architecture_dict["comparator"]["instance"][-1],
    )


def array_gemm(architecture_dict, workload_dict=None):
    cycles = workload_dict["configuration"]["cycles"]
    return _performance(
        cycles / _router_dim(architecture_dict),
        architecture_dict["temporal_register"]["query"]["frequency"],
        temporal_register=get_prod(
            architecture_dict["temporal_register"]["instance"][-2:]
        ),
        and_gate=get_prod(architecture_dict["and_gate"]["instance"][-2:]),
        or_gate=get_prod(architecture_dict["or_gate"]["instance"][-3:-1]),
        sign_xor=architecture_dict["sign_xor"]["instance"][-1] * cycles,
        adder=architecture_dict["adder"]["instance"][-1] * cycles,
        ofifo=architecture_dict["ofifo"]["instance"][-1] * cycles,
    )


def array_fifo_gemm(architecture_dict, workload_dict=None):
    cycles = workload_dict["configuration"]["cycles"]
    return _performance(
        cycles / _router_dim(architecture_dict),
        architecture_dict["pe_fifo"]["query"]["frequency"],
        # quirk: [-3:1] is an empty slice, so or_gate counts 1 (get_prod of []), not the height x width gates
        or_gate=get_prod(architecture_dict["or_gate"]["instance"][-3:1]),
        pe_fifo=architecture_dict["pe_fifo"]["instance"][-1] * cycles,
    )


def nonlinear_gemm(architecture_dict, workload_dict=None):
    # the nonlinear modules sit idle during a gemm; only the active subarchitecture's modules are in the graph
    names = [
        "round",
        "sign_mantissa_register",
        "exponent_register",
        "exp_clamp",
        "exp_norm",
        "exp_select",
        "max_tree",
        "max_tree_register",
    ]
    names += [
        name
        for name in ("window_select", "lut_register", "lut_decoder")
        if name in architecture_dict
    ]
    performance_dict = OrderedDict()
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    performance_dict["subevent"] = OrderedDict(
        {name: OrderedDict({"count": 0}) for name in names}
    )
    return performance_dict


# endregion


# region hardware: nonlinear
def input_nonlinear(architecture_dict, workload_dict=None):
    lut_height = workload_dict["configuration"]["lut_height"]
    return _performance(
        lut_height / _router_dim(architecture_dict),
        architecture_dict["ififo"]["query"]["frequency"],
        ififo=architecture_dict["ififo"]["instance"][-1] * lut_height,
    )


def weight_nonlinear(architecture_dict, workload_dict=None):
    w = workload_dict["configuration"]
    lut_height = w["lut_height"]
    # only the active subarchitecture's modules are in the architecture; the other one counts 0
    if w["subarch"] == "lut":
        window_select = 0
        # quirk: instance [..., height // 8, 2, 8, 12] gives 8 * 2 = 16 registers on every height
        lut_register = (
            architecture_dict["lut_register"]["instance"][-2]
            * architecture_dict["lut_register"]["instance"][-4]
        )
        lut_decoder = architecture_dict["lut_decoder"]["instance"][-1] * lut_height
    else:  # vlp
        window_select = architecture_dict["window_select"]["instance"][-1]
        lut_register = 0
        lut_decoder = 0
    return _performance(
        lut_height / _router_dim(architecture_dict),
        architecture_dict["round"]["query"]["frequency"],
        round=architecture_dict["round"]["instance"][-1],
        sign_mantissa_register=architecture_dict["sign_mantissa_register"]["instance"][
            -1
        ],
        exponent_register=architecture_dict["exponent_register"]["instance"][-1],
        exp_clamp=architecture_dict["exp_clamp"]["instance"][-1],
        wfifo=architecture_dict["wfifo"]["instance"][-1],
        max_tree=architecture_dict["max_tree"]["instance"][-1],
        max_tree_register=architecture_dict["max_tree_register"]["instance"][-1],
        magnitude_register=architecture_dict["magnitude_register"]["instance"][-1],
        sign_register=architecture_dict["sign_register"]["instance"][-1],
        sign_fifo=architecture_dict["sign_fifo"]["instance"][-1],
        exp_norm=architecture_dict["exp_norm"]["instance"][-1],
        exp_select=architecture_dict["exp_select"]["instance"][-1],
        ofifo=architecture_dict["ofifo"]["instance"][-1],
        window_select=window_select,
        lut_register=lut_register,
        lut_decoder=lut_decoder,
    )


def weight_reuse_nonlinear(architecture_dict, workload_dict=None):
    lut_height = workload_dict["configuration"]["lut_height"]
    return _performance(
        lut_height / _router_dim(architecture_dict),
        architecture_dict["comparator"]["query"]["frequency"],
        comparator=architecture_dict["comparator"]["instance"][-1] * lut_height,
    )


def array_nonlinear(architecture_dict, workload_dict=None):
    return _performance(
        workload_dict["configuration"]["cycles"] / _router_dim(architecture_dict),
        architecture_dict["temporal_register"]["query"]["frequency"],
        temporal_register=get_prod(
            architecture_dict["temporal_register"]["instance"][-2:]
        ),
        and_gate=get_prod(architecture_dict["and_gate"]["instance"][-2:]),
        or_gate=get_prod(architecture_dict["or_gate"]["instance"][-3:-1]),
    )


def array_fifo_nonlinear(architecture_dict, workload_dict=None):
    return _performance(
        workload_dict["configuration"]["cycles"] / _router_dim(architecture_dict),
        architecture_dict["pe_fifo"]["query"]["frequency"],
        # quirk: [-3:1] is an empty slice, so or_gate counts 1 (get_prod of []), not the height x width gates
        or_gate=get_prod(architecture_dict["or_gate"]["instance"][-3:1]),
        pe_fifo=architecture_dict["pe_fifo"]["instance"][-1],
    )


def summation(architecture_dict, workload_dict=None):
    return _performance(
        workload_dict["configuration"]["cycles"] / _router_dim(architecture_dict),
        architecture_dict["adder"]["query"]["frequency"],
        adder=architecture_dict["adder"]["instance"][-1],
    )


def gemm_nonlinear(architecture_dict, workload_dict=None):
    # the gemm modules sit idle during a nonlinear op
    performance_dict = OrderedDict()
    performance_dict["cycle_count"] = OrderedDict({"value": 0, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict({"value": 0, "unit": "ms"})
    names = [
        "multiplier",
        "multiplier_register",
        "sign_register",
        "sign_fifo",
        "sign_xor",
    ]
    performance_dict["subevent"] = OrderedDict(
        {name: OrderedDict({"count": 0}) for name in names}
    )
    return performance_dict


# endregion


# region memory
def _dram_event(width, operation, architecture_dict):
    gigabyte = 2**33  # bits
    megabyte = 2**23  # bits
    # quirk: multi-node designs time dram transfers at the router bandwidth (64 x 2^23), not the dram bandwidth (128 x 2^33)
    if "irouter" in architecture_dict:
        bandwidth = architecture_dict["irouter"]["query"]["bandwidth"] * megabyte
    else:
        bandwidth = architecture_dict["dram"]["query"]["bandwidth"] * gigabyte
    frequency = architecture_dict["dram"]["query"]["frequency"]
    return OrderedDict(
        {
            "cycle_count": OrderedDict({"value": width / bandwidth, "unit": "cycle"}),
            "runtime": OrderedDict(
                {"value": (width / bandwidth) / 1000 / frequency, "unit": "ms"}
            ),
            "subevent": OrderedDict(
                {
                    "dram": OrderedDict(
                        {"operation": OrderedDict({"dynamic_energy": operation})}
                    )
                }
            ),
        }
    )


def dram_input_reads(architecture_dict, workload_dict=None):
    return _dram_event(
        architecture_dict["isram"]["query"]["width"], "read", architecture_dict
    )


def dram_weight_reads(architecture_dict, workload_dict=None):
    return _dram_event(
        architecture_dict["wsram"]["query"]["width"], "read", architecture_dict
    )


def dram_output_reads(architecture_dict, workload_dict=None):
    return _dram_event(
        architecture_dict["osram"]["query"]["width"], "read", architecture_dict
    )


def dram_output_writes(architecture_dict, workload_dict=None):
    return _dram_event(
        architecture_dict["osram"]["query"]["width"], "write", architecture_dict
    )


def _sram_event(operation, sram, architecture_dict):
    cycle_count = 1 / _router_dim(architecture_dict)
    frequency = architecture_dict[sram]["query"]["frequency"]
    return OrderedDict(
        {
            "cycle_count": OrderedDict({"value": cycle_count, "unit": "cycle"}),
            "runtime": OrderedDict(
                {"value": cycle_count / 1000 / frequency, "unit": "ms"}
            ),
            "subevent": OrderedDict(
                {
                    sram: OrderedDict(
                        {"operation": OrderedDict({"dynamic_energy": operation})}
                    )
                }
            ),
        }
    )


def isram_offchip_writes(architecture_dict, workload_dict=None):
    return _sram_event("write", "isram", architecture_dict)


def isram_onchip_reads(architecture_dict, workload_dict=None):
    return _sram_event("read", "isram", architecture_dict)


def wsram_offchip_writes(architecture_dict, workload_dict=None):
    return _sram_event("write", "wsram", architecture_dict)


def wsram_onchip_reads(architecture_dict, workload_dict=None):
    return _sram_event("read", "wsram", architecture_dict)


def osram_offchip_reads(architecture_dict, workload_dict=None):
    return _sram_event("read", "osram", architecture_dict)


def osram_offchip_writes(architecture_dict, workload_dict=None):
    return _sram_event("write", "osram", architecture_dict)


def osram_onchip_reads(architecture_dict, workload_dict=None):
    return _sram_event("read", "osram", architecture_dict)


def osram_onchip_writes(architecture_dict, workload_dict=None):
    return _sram_event("write", "osram", architecture_dict)


# endregion


# region router
def _router_event(router, architecture_dict):
    router_width = architecture_dict[router]["instance"][-1]
    router_bandwidth = architecture_dict[router]["query"]["bandwidth"] * 2**23
    return _performance(
        1 / router_width / router_bandwidth,
        architecture_dict[router]["query"]["frequency"],
        **{router: 1},
    )


def irouter_mapping(architecture_dict, workload_dict=None):
    return _router_event("irouter", architecture_dict)


def wrouter_mapping(architecture_dict, workload_dict=None):
    return _router_event("wrouter", architecture_dict)


def orouter_mapping(architecture_dict, workload_dict=None):
    return _router_event("orouter", architecture_dict)


# endregion
