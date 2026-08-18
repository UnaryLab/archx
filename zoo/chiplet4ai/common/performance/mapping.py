from math import ceil, floor, ulp
from collections import OrderedDict

from chiplet4ai.common.performance.utils import _step_config, _step_dims, _sram_bits

def _architecture(architecture_dict: OrderedDict) -> OrderedDict:
    if 'architecture' in architecture_dict and 'pe' in architecture_dict['architecture']:
        return architecture_dict['architecture']
    if 'architecture' in architecture_dict and 'module' in architecture_dict['architecture']:
        modules = architecture_dict['architecture']['module']
    else:
        modules = architecture_dict

    return OrderedDict({
        'pe': modules['pe']['instance'],
        'isram': modules['isram'],
        'wsram': modules['wsram'],
        'osram': modules['osram'],
        'dram': modules['dram'],
    })

def _frequency_mhz(arch: OrderedDict) -> float:
    query = arch['dram'].get('query', {})
    if 'frequency' in query:
        return float(query['frequency'])
    return 1000.0

def _dram_bytes_per_cycle(arch: OrderedDict) -> float:
    # unit convention: the YAML 'dram: bandwidth:' field is DECIMAL GB/s, so 1e9 (not 2**30).
    # 256 -> 256.0 B/cy at 1 GHz. The only 2**30 kept in this file is in
    # _bandwidth_gib_per_second, whose reported figure is explicitly labelled GiB/s.
    bandwidth_gbs = float(arch['dram']['query']['bandwidth'])
    frequency_hz = _frequency_mhz(arch) * 1e6
    return max(1e-12, bandwidth_gbs * 1e9 / frequency_hz)

def _cycle_runtime_ms(arch: OrderedDict) -> float:
    return 1 / (1000 * _frequency_mhz(arch))

def _bandwidth_gib_per_second(arch: OrderedDict, bytes_count: float, cycles: float) -> float:
    if bytes_count <= 0 or cycles <= 0:
        return 0
    frequency_hz = _frequency_mhz(arch) * 1e6
    # unit convention: binary GiB/s, matching the 'GiB/s' unit declared on the bandwidth metric.
    return bytes_count / cycles * frequency_hz / 2**30

def _active_fraction(module: OrderedDict) -> float:
    query = module.get('query', {})
    fraction = float(query.get('active_fraction', query.get('active_buffer_fraction', 0.5)))
    return min(1.0, max(0.0, fraction))

def _sram_elements(arch: OrderedDict, sram_name: str) -> int:
    width = int(arch[sram_name]['query']['width'])
    return max(1, _sram_bits(OrderedDict({'architecture': arch}), sram_name) // width)

def _active_banks(arch: OrderedDict, sram_name: str) -> int:
    # SINGLE rounding rule for the active half of the double buffer. A bank is the
    # granularity of the split (you cannot activate a fraction of a bank), so the active
    # bank count floors to a whole bank and is clamped to [1, bank]. Both the capacity
    # view (_buffer_elements) and the port view (_sram_bytes_per_cycle) derive from this
    # one number, so they cannot diverge for non-0.5 fractions or non-power-of-2 banks.
    #
    # DEGENERACY, documented deliberately rather than asserted away (audit Open 4). The
    # `max(1, ...)` clamp is required (a zero-wide port would divide by zero and a
    # zero-element active buffer is meaningless), but it has two consequences a caller
    # must know about:
    #   * bank == 1: active == total for EVERY fraction, including 0.0. _buffer_elements
    #     then clamps prefetch up to 1, so there is NO double buffer at all -- effective
    #     capacity is the whole SRAM, which feeds the all-or-nothing input fit test and
    #     DRAM traffic. A single-bank SRAM is modelled as single-buffered
    #     with full capacity, not as a half-and-half split.
    #   * odd or non-dyadic bank counts: the floor bites. bank == 3 at fraction 0.5 gives
    #     one active bank of three (33%, not 50%).
    # An `assert bank >= 2` was considered and rejected: a single-bank SRAM is a legal
    # architecture to model, so the degeneracy above is documented rather than rejected --
    # the assert would refuse an architecture the model can represent perfectly well, for a
    # double-buffering nicety. The shipped sweep is unaffected either way: description.py
    # builds sram_banks = [64, 128, 256, 512, 1024] and no YAML sets active_fraction.
    # Callers sweeping bank == 1 or odd bank counts must read the two bullets above.
    #
    # 'bank' is indexed, not defaulted. It is hard-required by every SRAM consumer --
    # utils._sram_bits and the cacti7 interface both index query['bank'] directly -- and
    # _sram_bits runs before this function on every path that reaches it, so a bank-less
    # query has already raised KeyError. A dict-get with a default of 1 here would therefore
    # be dead code that could only ever mis-model a real SRAM as single-bank in silence.
    banks = max(1, int(arch[sram_name]['query']['bank']))
    return max(1, min(banks, floor(banks * _active_fraction(arch[sram_name]))))

def _buffer_elements(arch: OrderedDict, sram_name: str) -> tuple[int, int]:
    total = _sram_elements(arch, sram_name)
    banks = max(1, int(arch[sram_name]['query']['bank']))
    # active elements = active banks x elements-per-bank, from the shared _active_banks rule
    active = max(1, min(total, _active_banks(arch, sram_name) * (total // banks)))
    prefetch = max(1, total - active)
    return active, prefetch

def _sram_bytes_per_cycle(arch: OrderedDict, sram_name: str) -> float:
    # Port width is the active half of the double buffer, derived from the same
    # _active_banks rule that _buffer_elements uses, so the two cannot diverge.
    # At the default fraction 0.5 the whole-bank floor matches the old hardcoded bank/2
    # exactly on dyadic bank counts (so every shipped config is unchanged), but not in
    # general: at bank == 3 the floor bites, giving 1 of 3 banks (2.0 B/cy, not 3.0).
    width_bytes = float(arch[sram_name]['query']['width']) / 8
    return max(1e-12, _active_banks(arch, sram_name) * width_bytes)

def _service_cycles(bytes_count: float, bytes_per_cycle: float) -> int:
    if bytes_count <= 0:
        return 0
    return ceil(bytes_count / max(1e-12, bytes_per_cycle))

# Absolute byte-conservation tolerance. One byte is the finest quantity the model can
# move, so the invariant is asserted ABSOLUTELY at half a byte rather than with the former
# 1e-6 RELATIVE bound (~232 KB of slack at the reference scale, enough to hide the loss of
# a whole small stream). The phase assignment is a fixed number of float multiply-adds
# over a total, so its residual is a few ULP of that total; the `4 * ulp` term only ever
# binds above ~2^51 bytes, where a double can no longer represent a single byte and a
# sub-byte absolute bound is not expressible at all. Below that scale the tolerance is a
# flat 0.5 B.
def _byte_tolerance(total: float) -> float:
    return max(0.5, 4.0 * ulp(max(1.0, abs(total))))

def _split_integer(total: int, weights, fallback) -> list:
    """Split an integer `total` across three phases by `weights` (largest-remainder, so
    the parts are non-negative integers summing to exactly `total`). Closed form: no
    iteration over phases beyond the fixed three. `fallback` is used when every weight is
    zero, and a final fallback puts everything in the steady phase."""
    total = int(total)
    if total <= 0:
        return [0, 0, 0]
    share = [max(0.0, float(w)) for w in weights]
    if sum(share) <= 0:
        share = [max(0.0, float(w)) for w in fallback]
    if sum(share) <= 0:
        return [0, total, 0]
    scale = total / sum(share)
    raw = [w * scale for w in share]
    parts = [int(v) for v in raw]
    remainder = total - sum(parts)
    order = sorted(range(3), key=lambda i: (-(raw[i] - parts[i]), i))
    for i in order[:remainder]:
        parts[i] += 1
    return parts

def _slack_assign(total_bytes: float, bytes_per_cycle: float, eligible, windows, slacks) -> list:
    """Slack-aware assignment of one prorated byte stream across the three phases
    (audit Open 1(d)). `s = total_bytes / rate` is the stream's service demand in cycles.
    If it fits in the eligible phases' combined slack it is assigned proportional to that
    slack and adds no stall; otherwise each eligible phase is filled to its slack and the
    remainder is spread by window length. Closed form -- one three-way expression, no
    water-filling loop, because there are exactly three phases and filling every eligible
    phase to its slack is unconditional in the overflow branch.

    A zero-length window has zero slack and zero length, so it can never receive bytes.
    If every eligible window is zero (e.g. a 1x1 array has no fill and no tail skew, so
    the weight stream's whole eligible set collapses) the stream falls back to length
    proration over all non-zero windows: no byte stream may ever vanish."""
    if total_bytes <= 0:
        return [0.0, 0.0, 0.0]
    window = [windows[p] if eligible[p] else 0.0 for p in range(3)]
    slack = [slacks[p] if eligible[p] else 0.0 for p in range(3)]
    if sum(window) <= 0:
        window = [float(w) for w in windows]
        slack = [0.0, 0.0, 0.0]
    if sum(window) <= 0:
        # no compute span at all (degenerate shape): park the stream in the steady phase
        return [0.0, total_bytes, 0.0]
    total_slack = sum(slack)
    service = total_bytes / max(1e-12, bytes_per_cycle)
    if service <= total_slack:
        return [total_bytes * (slack[p] / total_slack) for p in range(3)]
    spill = service - total_slack
    window_sum = sum(window)
    return [(slack[p] + spill * window[p] / window_sum) * bytes_per_cycle for p in range(3)]

def _cycle_event(architecture_dict: OrderedDict) -> OrderedDict:
    arch = _architecture(architecture_dict)
    return OrderedDict({
        'cycle_count': OrderedDict({'value': 1, 'unit': 'cycle'}),
        'runtime': OrderedDict({'value': _cycle_runtime_ms(arch), 'unit': 'ms'}),
        'subevent': OrderedDict({'pe': OrderedDict({'count': 0})}),
    })

def cycle_reference(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _cycle_event(architecture_dict)

def _stall_event(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'cycle_reference': OrderedDict({'count': 1, 'aggregation': 'sequential'})
    })})

# The six stall events are named for the PHASE they measure (fill / steady / tail), not
# for an operand: the SRAM side is per-phase real-time pressure and the DRAM side is the
# whole-span floor spread back over the phases, so the quantities are phase-shaped. The
# former per-operand names misattributed them (under k_outer the steady phase's bytes are
# ~96% output spill, and in the reference config the tail phase is ~67% weight bytes).
#
# The '_dram'/'_sram' PREFIXES on these six names are NOT load-bearing and carry no
# bucketing meaning. query_cycle_breakdown (results/query/utils.py) buckets on endswith
# '_arr' / '_sram' / '_dram' over the PARENT composite events and adds each matched
# node's whole subtree; these stall events are counted because they sit inside the
# '*_dram' / '*_sram' wrappers' subtrees, never because of their own prefix. The prefixes
# are kept only as human-readable provenance for which wrapper emits them (dram() below
# emits the three 'dram_*', sram() the three 'sram_*'), and renaming them would not move
# a single cycle between buckets.
def dram_fill_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def dram_steady_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def dram_tail_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def sram_fill_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def sram_steady_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def sram_tail_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def _range_sum(first: int, last: int) -> int:
    if last < first:
        return 0
    return (first + last) * (last - first + 1) // 2

def _ceil_sum(first: int, last: int, divisor: int) -> int:
    if last < first:
        return 0
    return _ceil_sum_to(last, divisor) - _ceil_sum_to(first - 1, divisor)

def _ceil_sum_to(n: int, divisor: int) -> int:
    if n <= 0:
        return 0
    q, r = divmod(n, divisor)
    return divisor * q * (q + 1) // 2 + (q + 1) * r

def _ceil_minus_one_sum(first: int, last: int, divisor: int) -> int:
    count = max(0, last - first + 1)
    return _ceil_sum(first, last, divisor) - count

def _ws_schedule(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None, loop_order: str = 'auto') -> OrderedDict:
    arch = _architecture(architecture_dict)
    step_dim, min_step, max_step, total_steps = _step_config(M, K, N, step_start, step_dim)

    array_rows = arch['pe'][0]
    array_cols = arch['pe'][1]

    input_width = arch['isram']['query']['width']
    weight_width = arch['wsram']['query']['width']
    output_width = arch['osram']['query']['width']

    input_bytes = input_width / 8
    weight_bytes = weight_width / 8
    output_bytes = output_width / 8

    isram_active_elements = _buffer_elements(arch, 'isram')[0]
    wsram_active_elements = _buffer_elements(arch, 'wsram')[0]
    osram_active_elements = _buffer_elements(arch, 'osram')[0]

    # One shared DRAM channel: the architecture declares a single 'dram' with a single
    # bandwidth, so all operands contend for `bytes_per_cycle` inside a phase (see the
    # per-phase stall model below). The three SRAMs are physically separate memories.
    bytes_per_cycle = _dram_bytes_per_cycle(arch)
    isram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'isram')
    wsram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'wsram')
    osram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'osram')

    first_step = min_step + 1
    last_step = max_step
    if step_dim == 'm':
        sum_M = _range_sum(first_step, last_step)
        sum_K = K * total_steps
        sum_N = N * total_steps
        sum_k_folds = ceil(K / array_rows) * total_steps
        sum_n_folds = ceil(N / array_cols) * total_steps
        fold_count = ceil(K / array_rows) * ceil(N / array_cols) * total_steps
        compute_cycles = batch * ceil(K / array_rows) * ceil(N / array_cols) * (sum_M + total_steps * (array_rows + array_cols - 2))
        input_unique_elements = batch * K * sum_M
        input_sram_elements = input_unique_elements * ceil(N / array_cols)
        weight_elements = batch * K * N * total_steps
        output_write_elements = batch * N * sum_M * ceil(K / array_rows)
        output_read_elements = batch * N * sum_M * max(0, ceil(K / array_rows) - 1)
        max_M, max_K, max_N = last_step, K, N
    elif step_dim == 'k':
        sum_M = M * total_steps
        sum_K = _range_sum(first_step, last_step)
        sum_N = N * total_steps
        sum_k_folds = _ceil_sum(first_step, last_step, array_rows)
        sum_n_folds = ceil(N / array_cols) * total_steps
        fold_count = ceil(N / array_cols) * sum_k_folds
        compute_cycles = batch * ceil(N / array_cols) * sum_k_folds * (M + array_rows + array_cols - 2)
        input_unique_elements = batch * M * sum_K
        input_sram_elements = input_unique_elements * ceil(N / array_cols)
        weight_elements = batch * N * sum_K
        output_write_elements = batch * M * N * sum_k_folds
        output_read_elements = batch * M * N * _ceil_minus_one_sum(first_step, last_step, array_rows)
        max_M, max_K, max_N = M, last_step, N
    elif step_dim == 'n':
        sum_M = M * total_steps
        sum_K = K * total_steps
        sum_N = _range_sum(first_step, last_step)
        sum_k_folds = ceil(K / array_rows) * total_steps
        sum_n_folds = _ceil_sum(first_step, last_step, array_cols)
        fold_count = ceil(K / array_rows) * sum_n_folds
        compute_cycles = batch * ceil(K / array_rows) * sum_n_folds * (M + array_rows + array_cols - 2)
        input_unique_elements = batch * M * K * total_steps
        input_sram_elements = batch * M * K * sum_n_folds
        weight_elements = batch * K * sum_N
        output_write_elements = batch * M * sum_N * ceil(K / array_rows)
        output_read_elements = batch * M * sum_N * max(0, ceil(K / array_rows) - 1)
        max_M, max_K, max_N = M, K, last_step
    else:
        k_folds = ceil(K / array_rows)
        n_folds = ceil(N / array_cols)
        sum_M = M
        sum_K = K
        sum_N = N
        sum_k_folds = k_folds
        sum_n_folds = n_folds
        fold_count = k_folds * n_folds
        compute_cycles = batch * fold_count * (M + array_rows + array_cols - 2)
        input_unique_elements = batch * M * K
        input_sram_elements = input_unique_elements * n_folds
        weight_elements = batch * K * N
        output_write_elements = batch * M * N * k_folds
        output_read_elements = batch * M * N * max(0, k_folds - 1)
        max_M, max_K, max_N = M, K, N

    # SCALE-Sim-reconciled traffic laws (user ruling 2026-08-17, validated against ~90
    # USER-mode SCALE-Sim reference points; see scalesim_audit.md).
    #
    # INPUT, all-or-nothing: SCALE-Sim's WS traversal re-streams the whole ifmap once per
    # n-fold, and its read buffer is a rotating sequential prefetch window, so a re-pass
    # hits only when the ENTIRE input matrix sits in the active half of the double buffer.
    # No partial credit, and no strip-residency mode exists, so the law is loop-order
    # independent: reads = unique (M*K) if the matrix fits the active half, else the full
    # streamed ceiling (M*K * ceil(N/C)). Capacity is the exact active half from
    # _buffer_elements: SCALE-Sim's set-granular window can hold up to ~2% less at the
    # boundary, but a 0.98 margin would be a magic constant for a sub-2% edge case; the
    # exact half keeps one capacity definition across the model. For stepped decode GEMMs
    # the fit is tested once per event at the LARGEST step's working set (max_M * max_K),
    # the same granularity the replaced law used.
    max_input_matrix = batch * max_M * max_K
    input_fits = max_input_matrix <= isram_active_elements
    input_read_elements = input_unique_elements if input_fits else input_sram_elements

    # One final write per output element per step; the rest are partial-sum re-accesses.
    output_final_elements = output_write_elements - output_read_elements
    output_accum_elements = output_read_elements

    # OUTPUT, write-through: SCALE-Sim's write buffer is a pure drain (write_buffer.py:256)
    # -- every osram write (final + partial-sum update) drains to DRAM, M*N*k_folds writes
    # in total, independent of osram size, and partial sums are never read back from DRAM.
    # This replaces the osram-fit-gated spill law; the ruling is recorded in
    # scalesim_audit.md.
    dram_output_write_elements = output_write_elements
    dram_output_read_elements = 0.0

    # DRAM traffic is now loop-order independent (the input law is fixed to SCALE-Sim's
    # traversal and the output side is write-through), so 'auto' has nothing to choose;
    # it resolves to 'n_outer', the order whose semantics the input law matches. The
    # parameter is kept for interface stability.
    if loop_order == 'auto':
        loop_order = 'n_outer'
    assert loop_order in ('k_outer', 'n_outer'), \
        f"loop_order must be one of ['auto', 'k_outer', 'n_outer'], but got {loop_order}"

    weight_read_elements = weight_elements

    input_read_bytes = input_read_elements * input_bytes
    weight_read_bytes = weight_read_elements * weight_bytes
    output_read_bytes = dram_output_read_elements * output_bytes
    output_write_bytes = dram_output_write_elements * output_bytes
    input_sram_bytes = input_sram_elements * input_bytes
    weight_sram_bytes = weight_read_elements * weight_bytes
    # accumulation always reads and writes osram; DRAM only sees the non-resident fraction
    osram_read_bytes = output_accum_elements * output_bytes
    osram_write_bytes = output_write_elements * output_bytes
    output_sram_bytes = osram_read_bytes + osram_write_bytes

    input_count = input_read_elements / isram_active_elements
    weight_count = weight_read_elements / wsram_active_elements
    output_read_count = output_accum_elements / osram_active_elements
    output_write_count = output_write_elements / osram_active_elements

    # ------------------------------------------------------------------------------
    # Hybrid stall model (closed form; folds enter only as multipliers).
    #
    # The three phase windows are disjoint and exhaustive: they sum exactly to
    # compute_cycles = (M + R + C - 2) * folds, asserted below.
    #
    #   fill   = (array_rows - 1) * folds   weight-load skew
    #   steady = M * folds                  input streaming / psum accumulate
    #   tail   = (array_cols - 1) * folds   output drain skew
    #
    # The model has two terms combined with max():
    #
    #   (a) SPLIT-FREE FLOOR. For each resource r in {DRAM channel, isram, wsram, osram},
    #       service_r = ceil(total_bytes_r / rate_r) over the WHOLE GEMM with a SINGLE
    #       ceil, and floor = max(0, max_r service_r - compute_cycles). This is the
    #       roofline: no phase split can beat it and none may be charged above it for a
    #       purely global reason.
    #
    #   (b) REAL-TIME PHASE PRESSURE, SRAM ONLY.
    #           rt = SUM_p max(0, max_b ceil(bytes_{b,p} / rate_b) - window_p)
    #       DRAM contributes to rt NOT AT ALL. The physics: DRAM is decoupled from
    #       instantaneous demand by the prefetch half of the double buffer, so charging
    #       it per phase would model a barrier the hardware does not have -- it is
    #       represented entirely by the floor. The array's SRAM demand, by contrast, is
    #       genuinely real-time (the active half must supply its words in the cycle they
    #       are consumed), so per-phase resolution is meaningful there and is retained.
    #
    #   (c) stall_cycles = max(floor, rt).
    #
    # Because stall >= floor by construction, runtime >= max(compute_cycles, max_r
    # service_r) always, and where DRAM binds and no SRAM phase binds the runtime is
    # EXACTLY max(compute_cycles, ceil(total_dram_bytes / rate)) -- a single whole-span
    # ceil, with none of the per-phase double-ceil rounding the old aggregator paid.
    #
    # Residual pessimism, stated honestly: rt can exceed floor when a buffer's real-time
    # demand inside one phase exceeds that phase's window even though its whole-span
    # service fits. That is the physically correct behaviour (the array does stall), but
    # the phase boundaries are themselves a three-way discretization of the WS pipeline,
    # so the residual is a modelling artifact and is reported, never hidden. Two limits
    # remain UNADDRESSED here: DRAM elasticity is treated as unbounded (real prefetch is
    # capped by the prefetch buffer's capacity), and the all-or-nothing input fit is
    # tested once per event at the largest step's working set, not per decode step.
    #
    # Byte-to-phase assignment is slack-aware (see _slack_assign) rather than by window
    # length, so an idle resource in one phase lends capacity to a starved phase --
    # exactly what double buffering and cross-fold pipelining do in real hardware. Each
    # stream declares an eligible phase set, every one justified by the WS overlap that
    # makes SCALE-Sim's materialized demand matrix `T + 2*arr_row + arr_col - 2` rows tall
    # while wall-clock is `M + R + C - 2` (bw_audit_scalesim_spec.md 6.3):
    #
    #   weight load (wsram, DRAM)     fill + tail           weight-load prefix of a fold
    #                                                       overlaps the previous fold's
    #                                                       drain
    #   input stream (isram)          steady                consumed in real time, not
    #                                                       deferrable
    #   osram accumulate (rd + wr)    steady                 partial sums are produced and
    #                                                       consumed in the steady region
    #   final output drain (osram,    fill + steady + tail  outputs retire column by
    #     DRAM)                                             column throughout, and fold i's
    #                                                       drain overlaps fold i+1's fill
    #   DRAM, all streams             whole span            the prefetch buffer decouples
    #                                                       DRAM from instantaneous demand
    # ------------------------------------------------------------------------------
    fill_window = batch * max(0, array_rows - 1) * fold_count
    steady_window = (
        batch * ceil(K / array_rows) * ceil(N / array_cols) * sum_M
        if step_dim == 'm'
        else batch * M * fold_count
    )
    tail_window = batch * max(0, array_cols - 1) * fold_count
    assert fill_window + steady_window + tail_window == compute_cycles, \
        ("phase windows must partition the compute span exactly: "
         f"{fill_window} + {steady_window} + {tail_window} != {compute_cycles}")
    windows = (fill_window, steady_window, tail_window)

    # DRAM output traffic is write-through (every osram write drains, no read-back), so
    # output_read_bytes is 0 and output_write_bytes covers finals plus partial-sum
    # updates. output_final_bytes below is the OSRAM-side final drain only, used for the
    # slack assignment of the on-chip drain stream.
    output_final_bytes = output_final_elements * output_bytes
    dram_total_bytes = input_read_bytes + weight_read_bytes + output_read_bytes + output_write_bytes
    osram_total_bytes = osram_read_bytes + osram_write_bytes
    # osram accumulate = every osram access except the one final write per output element
    osram_accum_bytes = osram_total_bytes - output_final_bytes

    # Exclusive (non-prorated) demand per phase, in cycles, PER RESOURCE. Only the steady
    # phase owns any at all: the input stream (isram) and the accumulate traffic (osram).
    # wsram and the DRAM channel own none anywhere.
    isram_exclusive = (0.0, input_sram_bytes / isram_bytes_per_cycle, 0.0)
    osram_exclusive = (0.0, osram_accum_bytes / osram_bytes_per_cycle, 0.0)

    # Slack is what is left of a phase once a resource's OWN exclusive demand is served.
    # Two properties, both required to stop bandwidth being stranded:
    #   * Slack is PER RESOURCE, for the same reason the real-time term takes a MAX over
    #     buffers rather than a sum -- the three SRAMs are physically separate memories, so
    #     a saturated isram does not consume osram's capacity to retire finals.
    #   * The phase a resource has to work with is its REALIZED length, not its nominal
    #     window: a phase whose binding buffer is already over its window lasts that long
    #     for every buffer in it, so a non-binding buffer's room extends to the realized
    #     length. Without this the model strands exactly the capacity the whole fix exists
    #     to reclaim. Measured over a 57,820-point stress sweep, points whose runtime moves
    #     UPWARD against the old aggregator: 1,855 with phase-wide slack, 854 with
    #     per-resource slack on the nominal window, 71 with both properties (worst case
    #     +35.9% -> +2.6%). All three satisfy the split-free floor; the difference is
    #     purely how much slack is left stranded.
    # A zero-length window has zero exclusive demand and therefore zero slack either way,
    # so no assignment rule below can put bytes into one.
    realized_window = tuple(
        max(float(windows[p]), isram_exclusive[p], osram_exclusive[p]) for p in range(3))
    wsram_slacks = realized_window
    osram_slacks = tuple(max(0.0, realized_window[p] - osram_exclusive[p]) for p in range(3))

    wsram_phase_bytes = _slack_assign(
        weight_sram_bytes, wsram_bytes_per_cycle, (True, False, True), windows, wsram_slacks)
    osram_final_phase_bytes = _slack_assign(
        output_final_bytes, osram_bytes_per_cycle, (True, True, True), windows, osram_slacks)
    # DRAM is whole-span eligible with no exclusive per-phase demand, so its slack is the
    # window itself and both branches of _slack_assign collapse to length proration. The
    # per-phase DRAM bytes below therefore carry no stall weight (see (b)); they exist to
    # shape the reported dram_*_stall attribution and to be conservation-checked.
    dram_phase_bytes = _slack_assign(
        dram_total_bytes, bytes_per_cycle, (True, True, True), windows, windows)

    isram_phase_bytes = [0.0, input_sram_bytes, 0.0]
    osram_phase_bytes = [
        osram_final_phase_bytes[0],
        osram_accum_bytes + osram_final_phase_bytes[1],
        osram_final_phase_bytes[2],
    ]

    # Byte conservation: no stream may be dropped or duplicated by the phase routing, and
    # no bytes may land in a zero-length window. Absolute sub-byte tolerance, not relative.
    for name, parts, total in (
        ('DRAM', dram_phase_bytes, dram_total_bytes),
        ('wsram', wsram_phase_bytes, weight_sram_bytes),
        ('isram', isram_phase_bytes, input_sram_bytes),
        ('osram', osram_phase_bytes, osram_total_bytes),
    ):
        assert abs(sum(parts) - total) <= _byte_tolerance(total), \
            (f"per-phase {name} bytes must sum to the total: {parts} sums to {sum(parts)}, "
             f"expected {total}")
        assert all(parts[p] >= 0.0 for p in range(3)), \
            f"per-phase {name} bytes must be non-negative, got {parts}"
        assert all(parts[p] <= _byte_tolerance(total) for p in range(3) if windows[p] == 0) \
            or compute_cycles == 0, \
            f"per-phase {name} bytes must not land in a zero-length window: {parts} vs {windows}"

    # (a) split-free floor -- one ceil per resource over whole-GEMM bytes
    dram_service_cycles = _service_cycles(dram_total_bytes, bytes_per_cycle)
    isram_service_cycles = _service_cycles(input_sram_bytes, isram_bytes_per_cycle)
    wsram_service_cycles = _service_cycles(weight_sram_bytes, wsram_bytes_per_cycle)
    osram_service_cycles = _service_cycles(osram_total_bytes, osram_bytes_per_cycle)
    sram_service_cycles = max(isram_service_cycles, wsram_service_cycles, osram_service_cycles)
    bound_cycles = max(compute_cycles, dram_service_cycles, sram_service_cycles)
    floor_stall_cycles = max(0, max(dram_service_cycles, sram_service_cycles) - compute_cycles)

    # (b) real-time phase pressure, SRAM only
    phase_rt_stall = [
        max(0, max(
            _service_cycles(isram_phase_bytes[p], isram_bytes_per_cycle),
            _service_cycles(wsram_phase_bytes[p], wsram_bytes_per_cycle),
            _service_cycles(osram_phase_bytes[p], osram_bytes_per_cycle),
        ) - windows[p])
        for p in range(3)
    ]
    realtime_stall_cycles = sum(phase_rt_stall)

    # (c) combine
    stall_cycles = max(floor_stall_cycles, realtime_stall_cycles)

    # Attribution of the six reported per-phase stall events. The real-time term is
    # already per-phase and per-SRAM, so it lands on the sram_* events unchanged. Any
    # excess of the floor over it is a whole-span quantity and is attributed to the
    # resource that actually sets the floor: to the dram_* events when the DRAM channel
    # binds, otherwise to the sram_* events. It is spread over the phases in proportion
    # to that resource's own per-phase bytes (falling back to window length), by
    # largest-remainder so the six events sum to stall_cycles exactly.
    excess_cycles = stall_cycles - realtime_stall_cycles
    if dram_service_cycles >= sram_service_cycles:
        excess_weights = dram_phase_bytes
        excess_to_dram = True
    else:
        excess_to_dram = False
        if osram_service_cycles >= max(isram_service_cycles, wsram_service_cycles):
            excess_weights = osram_phase_bytes
        elif isram_service_cycles >= wsram_service_cycles:
            excess_weights = isram_phase_bytes
        else:
            excess_weights = wsram_phase_bytes
    excess_split = _split_integer(excess_cycles, excess_weights, windows)

    dram_fill_stall_cycles, dram_steady_stall_cycles, dram_tail_stall_cycles = (
        excess_split if excess_to_dram else [0, 0, 0])
    sram_fill_stall_cycles, sram_steady_stall_cycles, sram_tail_stall_cycles = (
        [phase_rt_stall[p] + (0 if excess_to_dram else excess_split[p]) for p in range(3)])
    assert (dram_fill_stall_cycles + dram_steady_stall_cycles + dram_tail_stall_cycles
            + sram_fill_stall_cycles + sram_steady_stall_cycles + sram_tail_stall_cycles
            ) == stall_cycles, \
        "the six per-phase stall events must sum to stall_cycles exactly"
    assert compute_cycles + stall_cycles >= bound_cycles, \
        ("runtime must never fall below the split-free bound: "
         f"{compute_cycles} + {stall_cycles} < {bound_cycles}")

    # traffic overlaps compute; bandwidths below are averages over the compute span
    input_transfer_window_cycles = compute_cycles if input_read_bytes > 0 else 0
    weight_transfer_window_cycles = compute_cycles if weight_read_bytes > 0 else 0
    output_read_transfer_window_cycles = compute_cycles if output_read_bytes > 0 else 0
    output_transfer_window_cycles = compute_cycles if output_write_bytes > 0 else 0
    # SRAM fill/drain events: every count/bandwidth pair below describes the SAME byte
    # volume, i.e. count * active_elements * width/8 == the bandwidth numerator.
    input_sram_window_cycles = compute_cycles if input_read_bytes > 0 else 0
    weight_sram_window_cycles = compute_cycles if weight_sram_bytes > 0 else 0
    output_read_sram_window_cycles = compute_cycles if osram_read_bytes > 0 else 0
    output_write_sram_window_cycles = compute_cycles if osram_write_bytes > 0 else 0
    # Collapsed-window invariant, asserted at its one honest home -- here, beside the
    # window definitions. All eight demand windows are the same literal expression
    # (`compute_cycles if <that stream's bytes> > 0 else 0`), which is why the query layer
    # legitimately collapses them into a single `demand_window_cycles` column. Anyone who
    # gives one of them a distinct definition trips this assert instead of silently
    # invalidating that collapse.
    _demand_windows = (
        input_transfer_window_cycles, weight_transfer_window_cycles,
        output_read_transfer_window_cycles, output_transfer_window_cycles,
        input_sram_window_cycles, weight_sram_window_cycles,
        output_read_sram_window_cycles, output_write_sram_window_cycles,
    )
    assert all(w in (0, compute_cycles) for w in _demand_windows), \
        (f"every demand window must be compute_cycles or 0, got {_demand_windows} "
         f"against compute_cycles={compute_cycles}")

    mapping_efficiency = (sum_K / (sum_k_folds * array_rows)) * (sum_N / (sum_n_folds * array_cols)) if sum_k_folds > 0 and sum_n_folds > 0 else 0
    compute_utilization = (batch * sum_M * sum_K * sum_N / total_steps) / (compute_cycles * array_rows * array_cols) if compute_cycles > 0 else 0

    return OrderedDict({
        'total_steps': total_steps,
        'loop_order': loop_order,
        'input_count': input_count,
        'weight_count': weight_count,
        'output_read_count': output_read_count,
        'output_write_count': output_write_count,
        'input_read_bytes': input_read_bytes,
        'weight_read_bytes': weight_read_bytes,
        'output_read_bytes': output_read_bytes,
        'output_write_bytes': output_write_bytes,
        'input_sram_bytes': input_sram_bytes,
        'weight_sram_bytes': weight_sram_bytes,
        'osram_read_bytes': osram_read_bytes,
        'osram_write_bytes': osram_write_bytes,
        'output_sram_bytes': output_sram_bytes,
        'input_transfer_window_cycles': input_transfer_window_cycles,
        'weight_transfer_window_cycles': weight_transfer_window_cycles,
        'output_read_transfer_window_cycles': output_read_transfer_window_cycles,
        'output_transfer_window_cycles': output_transfer_window_cycles,
        'input_sram_window_cycles': input_sram_window_cycles,
        'weight_sram_window_cycles': weight_sram_window_cycles,
        'output_read_sram_window_cycles': output_read_sram_window_cycles,
        'output_write_sram_window_cycles': output_write_sram_window_cycles,
        'compute_cycles': compute_cycles,
        'fill_window_cycles': fill_window,
        'steady_window_cycles': steady_window,
        'tail_window_cycles': tail_window,
        'dram_fill_stall_cycles': dram_fill_stall_cycles,
        'dram_steady_stall_cycles': dram_steady_stall_cycles,
        'dram_tail_stall_cycles': dram_tail_stall_cycles,
        'sram_fill_stall_cycles': sram_fill_stall_cycles,
        'sram_steady_stall_cycles': sram_steady_stall_cycles,
        'sram_tail_stall_cycles': sram_tail_stall_cycles,
        'stall_cycles': stall_cycles,
        'input_sram_bandwidth': _bandwidth_gib_per_second(arch, input_read_bytes, input_sram_window_cycles),
        'weight_sram_bandwidth': _bandwidth_gib_per_second(arch, weight_sram_bytes, weight_sram_window_cycles),
        'output_read_sram_bandwidth': _bandwidth_gib_per_second(arch, osram_read_bytes, output_read_sram_window_cycles),
        'output_write_sram_bandwidth': _bandwidth_gib_per_second(arch, osram_write_bytes, output_write_sram_window_cycles),
        'input_dram_bandwidth': _bandwidth_gib_per_second(arch, input_read_bytes, input_transfer_window_cycles),
        'weight_dram_bandwidth': _bandwidth_gib_per_second(arch, weight_read_bytes, weight_transfer_window_cycles),
        'output_read_dram_bandwidth': _bandwidth_gib_per_second(arch, output_read_bytes, output_read_transfer_window_cycles),
        'output_write_dram_bandwidth': _bandwidth_gib_per_second(arch, output_write_bytes, output_transfer_window_cycles),
        'mapping_efficiency': mapping_efficiency,
        'compute_utilization': compute_utilization,
        'cycle_runtime_ms': _cycle_runtime_ms(arch),
    })

def gemm(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None, loop_order: str = 'auto') -> OrderedDict:
    performance_dict = OrderedDict()
    arch = _architecture(architecture_dict)
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim, loop_order)

    total_steps = schedule['total_steps']
    input_count = 0
    weight_count = 0
    compute_count = 0

    step_dim, min_step, max_step, _ = _step_config(M, K, N, step_start, step_dim)
    array_rows = arch['pe'][0]
    array_cols = arch['pe'][1]

    for step in range(min_step + 1, max_step + 1):
        M_step, K_step, N_step = _step_dims(M, K, N, step, step_dim)

        m_tiles = ceil(M_step / array_rows)
        k_tiles = ceil(K_step / array_rows)
        n_tiles = ceil(N_step / array_cols)

        m_util = M_step / (m_tiles * array_rows)
        k_util = K_step / (k_tiles * array_rows)
        n_util = N_step / (n_tiles * array_cols)

        input_util = m_util * k_util
        weight_util = k_util * n_util
        compute_util = m_util * k_util * n_util

        input_tiles = batch * m_tiles * k_tiles * n_tiles * input_util
        weight_tiles = batch * k_tiles * n_tiles * weight_util
        compute_tiles = batch * m_tiles * k_tiles * n_tiles * compute_util

        input_count += input_tiles
        weight_count += weight_tiles
        compute_count += compute_tiles

    performance_dict['subevent'] = OrderedDict({
        'array_input_mapping': {
            'count': input_count / total_steps,
            'factor': {
                'cycle_count': schedule['compute_cycles'] / input_count if input_count > 0 else 0,
                'runtime': schedule['compute_cycles'] / input_count if input_count > 0 else 0
                }
        },
        'array_weight_mapping': {
            'count': weight_count / total_steps,
            'factor': {
                'cycle_count': schedule['compute_cycles'] / weight_count if weight_count > 0 else 0,
                'runtime': schedule['compute_cycles'] / weight_count if weight_count > 0 else 0
            }
        },
        'array_compute_mapping': {
            'count': compute_count / total_steps,
            'factor': {
                'cycle_count': schedule['compute_cycles'] / compute_count if compute_count > 0 else 0,
                'runtime': schedule['compute_cycles'] / compute_count if compute_count > 0 else 0
            }
        }
    })
    return performance_dict

def sram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None, loop_order: str = 'auto') -> OrderedDict:
    performance_dict = OrderedDict()
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim, loop_order)
    total_steps = schedule['total_steps']

    performance_dict['subevent'] = OrderedDict({
        'sram_input_write_mapping': OrderedDict({
            'count': schedule['input_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['input_sram_bandwidth'],
            }
        }),
        'sram_weight_write_mapping': OrderedDict({
            'count': schedule['weight_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['weight_sram_bandwidth'],
            }
        }),
        'sram_output_read_mapping': OrderedDict({
            'count': schedule['output_read_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['output_read_sram_bandwidth'],
            }
        }),
        'sram_output_write_mapping': OrderedDict({
            'count': schedule['output_write_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['output_write_sram_bandwidth'],
            }
        }),
        # SRAM-attributable share of each phase stall (the DRAM share is emitted by
        # dram()). These carry the per-phase real-time SRAM pressure, plus the split-free
        # floor's excess over it when an SRAM -- not the DRAM channel -- sets the floor.
        # fed by the FILL phase (weight-load skew, (R-1)*folds): wsram + osram drain
        'sram_fill_stall': OrderedDict({
            'count': schedule['sram_fill_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        # fed by the STEADY phase (M*folds): isram stream + osram accumulate + osram drain
        # (+ wsram in the 1x1 fallback, where there is no fill or tail window to hold it)
        'sram_steady_stall': OrderedDict({
            'count': schedule['sram_steady_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        # fed by the TAIL phase (output drain skew, (C-1)*folds): osram drain + wsram
        'sram_tail_stall': OrderedDict({
            'count': schedule['sram_tail_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
    })

    return performance_dict

def dram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None, loop_order: str = 'auto') -> OrderedDict:
    performance_dict = OrderedDict()
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim, loop_order)

    total_steps = schedule['total_steps']
    input_read_count = schedule['input_read_bytes']
    weight_read_count = schedule['weight_read_bytes']
    output_read_count = schedule['output_read_bytes']
    output_write_count = schedule['output_write_bytes']

    performance_dict['subevent'] = OrderedDict({
        # Traffic edges are zero-cycle (cycle_count and runtime factors are 0), so they
        # overlap compute and contribute no time; 'parallel' is applied uniformly to all
        # four, matching the SRAM traffic events.
        'dram_input_read': OrderedDict({
            'count': input_read_count / total_steps,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['input_dram_bandwidth']}
        }),
        'dram_weight_read': OrderedDict({
            'count': weight_read_count / total_steps,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['weight_dram_bandwidth']}
        }),
        'dram_output_read': OrderedDict({
            'count': output_read_count / total_steps,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['output_read_dram_bandwidth']}
        }),
        'dram_output_write': OrderedDict({
            'count': output_write_count / total_steps,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['output_write_dram_bandwidth']}
        }),
        # DRAM-attributable share of each phase stall (the SRAM share is emitted by
        # sram()). DRAM never enters the per-phase real-time term -- the prefetch half of
        # the double buffer decouples it from instantaneous demand -- so these three carry
        # the whole-span split-free floor, spread back over the phases in proportion to
        # each phase's DRAM bytes, and are identically 0 whenever an SRAM sets the floor.
        # fed by the FILL phase ((R-1)*folds): shared-channel DRAM share in fill
        'dram_fill_stall': OrderedDict({
            'count': schedule['dram_fill_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        # fed by the STEADY phase (M*folds): shared-channel DRAM share in steady
        'dram_steady_stall': OrderedDict({
            'count': schedule['dram_steady_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        # fed by the TAIL phase ((C-1)*folds): shared-channel DRAM share in tail
        'dram_tail_stall': OrderedDict({
            'count': schedule['dram_tail_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
    })

    return performance_dict
