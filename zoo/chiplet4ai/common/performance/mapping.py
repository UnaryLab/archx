import math
from collections import OrderedDict
from functools import lru_cache

from chiplet4ai.common.performance.utils import _step_config, _step_dims

# MEMOIZATION, and why it is split the way it is. Both helpers below are pure arithmetic
# over a handful of integers, and both are called many times with repeating arguments --
# a GEMM's array/SRAM/DRAM events each re-derive the same tiling, and a decode walk
# re-derives it once per step. The cached halves therefore take SCALARS, never the
# architecture dict.
#
# That split is not cosmetic. The engine hands performance models a tracked mapping and
# records which architecture fields each one reads, so it knows when a cached result is
# stale. Caching on the dict would skip those reads and hide the dependency; caching on
# scalars keeps every read on the uncached wrapper, where the tracker still sees it.
@lru_cache(maxsize=None)
def _buffer_elements_from(bank: int, depth: int) -> tuple[int, int]:
    # Double buffered: half the banks are active while the other half prefetch the next
    # tile, so a tile is sized against the active half. A transfer is `width` bits and
    # every active bank does one per cycle, so the active bank count is both the capacity
    # divisor and the bandwidth. An odd bank count floors (bank == 3 gives 1 of 3).
    active_banks = max(1, bank // 2)
    return active_banks * depth, active_banks

def _buffer_elements(architecture_dict: OrderedDict, sram_name: str) -> tuple[int, int]:
    query = architecture_dict[sram_name]['query']
    return _buffer_elements_from(int(query['bank']), int(query['depth']))

def _dram_fills(elements: int, passes: int, sram_elements: int) -> int:
    # PARTIAL RESIDENCY. An operand is read `passes` times over the GEMM -- the input once
    # per N tile, the weight matrix once per M sweep. The first pass always comes from
    # DRAM. What the later passes cost depends on how much of the operand is still on
    # chip: DRAM fills in bursts, so the part that fits the active half is served locally
    # and only the remainder is re-fetched.
    #
    # This is graded, not all or nothing. Every element of capacity buys back one element
    # of re-fetch on every later pass, so traffic falls inversely with SRAM size and
    # bottoms out at `elements` once the operand is fully resident.
    resident = min(elements, sram_elements)
    return elements + (elements - resident) * (passes - 1)

def _tiling(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> tuple:
    # The dict-reading half: every architecture and workload field this model depends on
    # is read HERE, on every call, so the engine's dependency tracking stays complete.
    # The arithmetic is cached below on the scalars alone.
    return _tiling_from(
        int(architecture_dict['pe']['instance'][0]),
        int(architecture_dict['pe']['instance'][1]),
        _buffer_elements(architecture_dict, 'isram')[0],
        _buffer_elements(architecture_dict, 'wsram')[0],
        _buffer_elements(architecture_dict, 'osram')[0],
        int(workload_dict['M']),
        int(workload_dict['K']),
        int(workload_dict['N']),
    )

# Bounded: a decode walk asks for one tiling per step, so an unbounded cache would grow
# to the sequence length (a million entries at DeepSeek's context). This size covers a
# 131072-token walk outright -- letting the sram and dram walks reuse what the array walk
# computed -- and degrades to plain LRU behaviour beyond it.
@lru_cache(maxsize=1 << 18)
def _tiling_from(array_m: int, array_n: int, isram_elements: int, wsram_elements: int,
                 osram_elements: int, M: int, K: int, N: int) -> tuple:
    # WEIGHT STATIONARY. The array holds the weight tile, so its rows are the K
    # reduction and its columns are N: a Kt x Nt tile sits in the PEs while M streams
    # through. Tiles start at the array and shrink monotonically until every operand
    # fits its SRAM.
    Kt = min(K, array_m)
    Nt = min(N, array_n)

    # wsram holds the stationary Kt x Nt weight tile. Nt shrinks first: an extra n tile
    # only re-streams inputs, while an extra k fold leaves a partial sum to accumulate.
    if Kt * Nt > wsram_elements:
        Nt = max(1, wsram_elements // Kt)
        if Kt * Nt > wsram_elements:
            Kt = wsram_elements

    # Mt >= 1 needs one streamed input row and one output row resident. Both only shrink
    # further, so the wsram fit above still holds.
    Kt = min(Kt, isram_elements)
    Nt = min(Nt, osram_elements)

    # M streams through the array, so its chunk is set by the buffers rather than by the
    # array: isram holds an Mt x Kt input chunk, osram an Mt x Nt output chunk.
    Mt = min(M, isram_elements // Kt, osram_elements // Nt)

    num_m_tiles = math.ceil(M / Mt)
    num_k_tiles = math.ceil(K / Kt)
    num_n_tiles = math.ceil(N / Nt)

    # SRAM BLOCKING -- the second level of tiling, and the one that bends the curve.
    #
    # The array tile is fixed by the PE grid, so capacity can never make it bigger. What
    # capacity CAN do is hold many array tiles at once and let the array walk them on
    # chip, so DRAM is visited once per BLOCK instead of once per tile. That puts
    # capacity in the denominator of the pass count: traffic falls as 1/C rather than by
    # a fixed amount per added element, which is the difference between a hyperbola and
    # a descending line.
    #
    # Each buffer bounds one block, and so owns one lane:
    #   wsram  holds the weight tiles of an N block -> the input is re-read once per N
    #          BLOCK rather than once per N tile
    #   isram  holds the input chunks of an M block -> the weight matrix is re-read once
    #          per M block
    #   osram  holds the live partial sums (see the output lane in dram_mapping)
    #
    # A block is a whole number of array tiles, because that is what the array actually
    # iterates. The rounding is what makes the curve a staircase rather than smooth.
    n_tiles_per_block = max(1, wsram_elements // (Kt * Nt))
    m_tiles_per_block = max(1, isram_elements // (Mt * Kt))
    num_n_blocks = math.ceil(num_n_tiles / n_tiles_per_block)
    num_m_blocks = math.ceil(num_m_tiles / m_tiles_per_block)

    return Mt, Kt, Nt, num_m_tiles, num_k_tiles, num_n_tiles, num_m_blocks, num_n_blocks

def _folded_counts(subevents_at, num_steps: int) -> tuple[OrderedDict, OrderedDict]:
    # PLATEAU SUMMATION. A decode walk asks the same question at every sequence length,
    # but the tiling only changes shape a few thousand times over a million steps: between
    # those changes each count is either constant or advances by a fixed amount per step.
    # Such a run collapses to `n*first + delta*n*(n-1)/2` -- one arithmetic series instead
    # of n evaluations.
    #
    # The linearity is VERIFIED, never assumed. A run is collapsed only if its own
    # endpoints confirm it: first, first+delta, and first+(n-1)*delta must land exactly on
    # the values actually computed at those steps. Anything that breaks linearity inside
    # the run -- a tile count stepping, `_dram_fills` crossing its capacity, `_tiling`
    # switching between spilling and re-walking, even the per-step byte rounding -- fails
    # that check, and the run is bisected until the pieces do pass or are summed the long
    # way. Every count here is an integer, so the check and the series are exact: the
    # result is bit-identical to summing every step.
    #
    # This is why the array walk does not use it: its counts are floats, and regrouping
    # float additions changes the last bits.
    evaluated = {}

    def evaluate(index):
        if index not in evaluated:
            evaluated[index] = subevents_at(index)
        return evaluated[index]

    counts = OrderedDict()

    def accumulate(name, value):
        counts[name] = counts.get(name, 0) + value

    # explicit stack rather than recursion: a walk can be a million steps deep
    pending = [(0, num_steps - 1)]
    while pending:
        low, high = pending.pop()
        span = high - low + 1

        # too short to be worth a linearity probe: three evaluations would cost as much
        if span <= 3:
            for index in range(low, high + 1):
                for name, subevent in evaluate(index).items():
                    accumulate(name, subevent['count'])
            continue

        first = evaluate(low)
        second = evaluate(low + 1)
        last = evaluate(high)

        deltas = {name: second[name]['count'] - subevent['count']
                  for name, subevent in first.items()}
        linear = all(subevent['count'] + deltas[name] * (span - 1) == last[name]['count']
                     for name, subevent in first.items())

        if linear:
            for name, subevent in first.items():
                accumulate(name, span * subevent['count']
                           + deltas[name] * span * (span - 1) // 2)
        else:
            middle = (low + high) // 2
            pending.append((low, middle))
            pending.append((middle + 1, high))

    return counts, evaluate(num_steps - 1)

def array_mapping_decode(dim: str, tokens: int, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # workload parameters
    M = workload_dict['M']
    N = workload_dict['N']
    K = workload_dict['K']

    # ---------------------------------------------------------
    # 1. Steps
    #
    # A decode GEMM grows by one row (or column, or reduction step) per generated token,
    # and the graph has no way to say that: an edge carries one count, not a count per
    # token. So walk `dim` from the prefill length to its final value, map every length
    # on its own, and fold the results into a single event. `step_start` is where the
    # walk begins -- the prefill sequence length -- and dim None maps the GEMM once, at
    # its stated size.
    # ---------------------------------------------------------

    step_dim, min_step, max_step, total_steps = _step_config(
        M, K, N, workload_dict.get('step_start', 0), dim)

    # ---------------------------------------------------------
    # 2. Aggregation
    #
    # Counts SUM: every step is work that really happens. Utilizations AVERAGE over the
    # steps, since each one describes how well a single mapping filled the array, and
    # the folded event needs one representative figure. array_mapping writes the factor
    # as 1 / utilization, so the utilization is read back out of it.
    # ---------------------------------------------------------

    counts = OrderedDict()
    utilizations = OrderedDict()
    # `tokens` per step means the walk takes fewer, longer strides, so the mean divides
    # by the strides actually taken -- not by total_steps, which counts single tokens and
    # would shrink every utilization by a factor of `tokens`.
    sampled_steps = 0

    for step in range(min_step + 1, max_step + 1, tokens):
        sampled_steps += 1
        step_m, step_k, step_n = _step_dims(M, K, N, step, step_dim)
        step_dict = array_mapping(
            architecture_dict, OrderedDict({'M': step_m, 'K': step_k, 'N': step_n}))

        for name, subevent in step_dict['subevent'].items():
            counts[name] = counts.get(name, 0) + subevent['count']
            utilizations[name] = utilizations.get(name, 0) + 1 / subevent['factor']['cycle_count']

    # ---------------------------------------------------------
    # 3. Hardware events
    # ---------------------------------------------------------

    performance_dict['subevent'] = OrderedDict({
        name: {
            'count': counts[name],
            'factor': {'cycle_count': sampled_steps / utilizations[name],
                       'runtime': sampled_steps / utilizations[name]},
        }
        for name in counts
    })

    return performance_dict


def array_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    array_m = architecture_dict['pe']['instance'][0]
    array_n = architecture_dict['pe']['instance'][1]

    # workload parameters
    M = workload_dict['M']
    N = workload_dict['N']
    K = workload_dict['K']

    Mt, Kt, Nt, num_m_tiles, num_k_tiles, num_n_tiles, num_m_blocks, _ = _tiling(architecture_dict, workload_dict)

    # ---------------------------------------------------------
    # 1. Mapping counts
    #
    # One mapping = one full-array load/stream of that operand's tile.
    #
    # WEIGHT STATIONARY, M innermost:
    #   A Kt x Nt weight tile is held in the array while the m tiles of one block sweep
    #   past it, so it is loaded once per (k, n, m block); isram capacity sets how far a
    #   block reaches (see _tiling). An input chunk is streamed for every (m, k, n).
    # ---------------------------------------------------------

    num_weight_mappings = num_k_tiles * num_n_tiles * num_m_blocks
    num_input_mappings = num_m_tiles * num_k_tiles * num_n_tiles
    num_mac_mappings = num_m_tiles * num_k_tiles * num_n_tiles

    # ---------------------------------------------------------
    # 2. Utilization
    #
    # Partial tiling is carried by the utilizations, not by fractional mapping counts:
    # a partial edge tile still costs a whole mapping. K and N are measured against the
    # PHYSICAL array they occupy, so they also capture a GEMM smaller than the array and
    # any tile the SRAM fit shrank below it; M is streamed, so it is measured against
    # the buffered chunk it is folded into.
    # ---------------------------------------------------------

    m_utilization = M / (num_m_tiles * Mt)
    k_utilization = K / (num_k_tiles * array_m)
    n_utilization = N / (num_n_tiles * array_n)

    weight_utilization = k_utilization * n_utilization
    input_utilization = k_utilization * m_utilization
    mac_utilization = k_utilization * n_utilization * m_utilization

    # ---------------------------------------------------------
    # 3. Cycles
    #
    # A mapping streams its M chunk through the array one row per cycle; a weight load
    # shifts the full array depth. Systolic fill and drain are ignored.
    # ---------------------------------------------------------

    input_cycle_count = num_input_mappings * Mt
    weight_cycle_count = num_weight_mappings * array_m
    mac_cycle_count = num_mac_mappings * Mt

    # ---------------------------------------------------------
    # 4. Hardware events
    #
    # Every subevent bills a FULLY UTILIZED array, so the count passed is the
    # fully-utilized equivalent -- cycles x utilization -- which keeps the module event
    # counts (and so the energy) on useful work only, while the cycle_count and runtime
    # factors scale that count back up to the cycles actually spent. The three edges are
    # parallel: streaming, weight loading and compute overlap, so the engine takes the
    # slowest rather than their sum.
    # ---------------------------------------------------------

    performance_dict['subevent'] = OrderedDict({
        'array_input': {
            'count': input_cycle_count * input_utilization,
            'factor': {'cycle_count': 1 / input_utilization, 'runtime': 1 / input_utilization},
        },
        'array_weight': {
            'count': weight_cycle_count * weight_utilization,
            'factor': {'cycle_count': 1 / weight_utilization, 'runtime': 1 / weight_utilization},
        },
        'array_compute': {
            'count': mac_cycle_count * mac_utilization,
            'factor': {'cycle_count': 1 / mac_utilization, 'runtime': 1 / mac_utilization},
        },
    })

    return performance_dict

def sram_mapping_decode(dim: str, tokens: int, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # workload parameters
    M = workload_dict['M']
    N = workload_dict['N']
    K = workload_dict['K']

    # ---------------------------------------------------------
    # 1. Steps
    #
    # Same walk array_mapping_decode makes: a decode GEMM grows `dim` by one per
    # generated token, which the graph cannot express, so every length is mapped on its
    # own and folded into a single event. dim None maps the GEMM once.
    # ---------------------------------------------------------

    step_dim, min_step, max_step, _ = _step_config(
        M, K, N, workload_dict.get('step_start', 0), dim)

    # ---------------------------------------------------------
    # 2. Aggregation
    #
    # Only the element counts move with the step: an SRAM event is per element and its
    # factor is the active bank count, which the architecture fixes. So the counts SUM
    # and the edge properties carry over unchanged -- there is no utilization to average
    # here, and the fold is exact. Being integer counts, they go through _folded_counts,
    # which collapses the linear runs and is bit-identical to summing every step.
    # ---------------------------------------------------------

    steps = range(min_step + 1, max_step + 1, tokens)

    def subevents_at(index):
        step_m, step_k, step_n = _step_dims(M, K, N, steps[index], step_dim)
        return sram_mapping(
            architecture_dict, OrderedDict({'M': step_m, 'K': step_k, 'N': step_n}))['subevent']

    counts, step_subevents = _folded_counts(subevents_at, len(steps))

    # ---------------------------------------------------------
    # 3. Hardware events
    # ---------------------------------------------------------

    performance_dict['subevent'] = OrderedDict({
        name: {**step_subevents[name], 'count': counts[name]}
        for name in counts
    })

    return performance_dict

def sram_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    isram_elements, isram_banks = _buffer_elements(architecture_dict, 'isram')
    wsram_elements, wsram_banks = _buffer_elements(architecture_dict, 'wsram')
    osram_banks = _buffer_elements(architecture_dict, 'osram')[1]

    # workload parameters
    M = workload_dict['M']
    N = workload_dict['N']
    K = workload_dict['K']

    _, _, _, _, num_k_tiles, num_n_tiles, num_m_blocks, num_n_blocks = _tiling(
        architecture_dict, workload_dict)

    # ---------------------------------------------------------
    # 1. Elements
    #
    # The USEFUL (unpadded) elements moved: an edge tile transfers only the data it
    # holds, while the padding it maps into the array is the array's business.
    #
    # The array READS the input once per N tile and a weight tile once per (k, n, m
    # block), always -- those are on-chip and set by the PE grid. What each SRAM is
    # FILLED with is set by the BLOCK counts instead, and reduced further by whatever
    # survived between passes: see _dram_fills. Every K fold drains a PARTIAL output
    # tile, with the running sum read back on every fold after the first.
    # ---------------------------------------------------------

    input_read_elements = M * K * num_n_tiles
    input_fill_elements = _dram_fills(M * K, num_n_blocks, isram_elements)
    weight_read_elements = K * N * num_m_blocks
    weight_fill_elements = _dram_fills(K * N, num_m_blocks, wsram_elements)
    output_write_elements = M * N * num_k_tiles
    output_read_elements = M * N * (num_k_tiles - 1)

    # ---------------------------------------------------------
    # 2. Hardware events
    #
    # memory.py bills ONE ELEMENT per event, at one cycle each. An SRAM moves one
    # width-bit transfer per active bank per cycle, so the cycle_count and runtime
    # factors divide by the active bank count. The counts need no utilization scaling:
    # unlike the array, a memory event is per element, and these element counts are
    # already the useful ones. Every edge is parallel -- the three SRAMs are separate
    # memories, and the active half of each serves the array while the other half is
    # filled -- so the engine takes the slowest stream rather than their sum.
    # ---------------------------------------------------------

    performance_dict['subevent'] = OrderedDict({
        'sram_input_write': {
            'count': input_fill_elements,
            'factor': {'cycle_count': 1 / isram_banks, 'runtime': 1 / isram_banks},
        },
        'sram_input_read': {
            'count': input_read_elements,
            'factor': {'cycle_count': 1 / isram_banks, 'runtime': 1 / isram_banks},
        },
        'sram_weight_write': {
            'count': weight_fill_elements,
            'factor': {'cycle_count': 1 / wsram_banks, 'runtime': 1 / wsram_banks},
        },
        'sram_weight_read': {
            'count': weight_read_elements,
            'factor': {'cycle_count': 1 / wsram_banks, 'runtime': 1 / wsram_banks},
        },
        'sram_output_write': {
            'count': output_write_elements,
            'factor': {'cycle_count': 1 / osram_banks, 'runtime': 1 / osram_banks},
        },
        'sram_output_read': {
            'count': output_read_elements,
            'factor': {'cycle_count': 1 / osram_banks, 'runtime': 1 / osram_banks},
        },
    })

    return performance_dict

def dram_mapping_decode(dim: str, tokens: int, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # workload parameters
    M = workload_dict['M']
    N = workload_dict['N']
    K = workload_dict['K']

    # ---------------------------------------------------------
    # 1. Steps
    #
    # Same walk array_mapping_decode makes: a decode GEMM grows `dim` by one per
    # generated token, which the graph cannot express, so every length is mapped on its
    # own and folded into a single event. dim None maps the GEMM once.
    # ---------------------------------------------------------

    step_dim, min_step, max_step, _ = _step_config(
        M, K, N, workload_dict.get('step_start', 0), dim)

    # ---------------------------------------------------------
    # 2. Aggregation
    #
    # Only the byte counts move with the step: a DRAM event is per byte and carries its
    # own runtime at the module's bandwidth. So the counts SUM and the edge properties
    # carry over unchanged -- there is no utilization to average here, and the fold is
    # exact. Each step rounds its lanes up to a whole byte, so a sub-byte element width
    # rounds per step rather than once over the whole walk -- and _folded_counts preserves
    # that too, since a rounding that breaks linearity simply fails its check and is
    # summed the long way.
    # ---------------------------------------------------------

    steps = range(min_step + 1, max_step + 1, tokens)

    def subevents_at(index):
        step_m, step_k, step_n = _step_dims(M, K, N, steps[index], step_dim)
        return dram_mapping(
            architecture_dict, OrderedDict({'M': step_m, 'K': step_k, 'N': step_n}))['subevent']

    counts, step_subevents = _folded_counts(subevents_at, len(steps))

    # ---------------------------------------------------------
    # 3. Hardware events
    # ---------------------------------------------------------

    performance_dict['subevent'] = OrderedDict({
        name: {**step_subevents[name], 'count': counts[name]}
        for name in counts
    })

    return performance_dict

def dram_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # one lane per SRAM, each carrying that SRAM's element width
    isram_width = architecture_dict['isram']['query']['width']
    wsram_width = architecture_dict['wsram']['query']['width']
    osram_width = architecture_dict['osram']['query']['width']

    bandwidth = architecture_dict['dram']['query']['bandwidth']
    dram_frequency = architecture_dict['dram']['query']['frequency']
    isram_elements = _buffer_elements(architecture_dict, 'isram')[0]
    wsram_elements = _buffer_elements(architecture_dict, 'wsram')[0]
    osram_elements = _buffer_elements(architecture_dict, 'osram')[0]

    # workload parameters
    M = workload_dict['M']
    N = workload_dict['N']
    K = workload_dict['K']

    _, _, _, _, num_k_tiles, _, num_m_blocks, num_n_blocks = _tiling(
        architecture_dict, workload_dict)

    # ---------------------------------------------------------
    # 1. Events
    #
    # memory.py bills ONE BYTE per DRAM event. Each lane is a pass count set by its
    # buffer's BLOCK, cut further by what stayed resident between passes:
    #   INPUT   one pass per N block (wsram bounds the block), minus what survived isram
    #   WEIGHT  one pass per M block (isram bounds the block), minus what survived wsram
    #   OUTPUT  one write per K fold, minus the partials that stayed live in osram. The
    #           first pass is the M*N finals, which always reach DRAM; every later fold
    #           only moves the partial tiles osram could not keep, and reads back exactly
    #           what it wrote. A big enough osram holds every live partial and the lane
    #           collapses to the finals alone.
    # ---------------------------------------------------------

    input_bytes = math.ceil(_dram_fills(M * K, num_n_blocks, isram_elements) * isram_width / 8)
    weight_bytes = math.ceil(_dram_fills(K * N, num_m_blocks, wsram_elements) * wsram_width / 8)

    output_write_elements = _dram_fills(M * N, num_k_tiles, osram_elements)
    output_read_elements = output_write_elements - M * N
    output_write_bytes = math.ceil(output_write_elements * osram_width / 8)
    output_read_bytes = math.ceil(output_read_elements * osram_width / 8)

    # ---------------------------------------------------------
    # 2. Hardware events
    #
    # A DRAM event already carries its own runtime at the module's bandwidth, so only
    # cycle_count needs a factor to turn bytes into cycles. Bandwidth is decimal GB/s
    # and frequency is MHz. The lanes run concurrently, each at the full bandwidth, so
    # every edge is parallel and the engine takes the slowest lane rather than the sum.
    # ---------------------------------------------------------

    bytes_per_cycle = bandwidth * 1e9 / (dram_frequency * 1e6)

    performance_dict['subevent'] = OrderedDict({
        'dram_input_read': {
            'count': input_bytes,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 1 / bytes_per_cycle},
        },
        'dram_weight_read': {
            'count': weight_bytes,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 1 / bytes_per_cycle},
        },
        'dram_output_write': {
            'count': output_write_bytes,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 1 / bytes_per_cycle},
        },
        'dram_output_read': {
            'count': output_read_bytes,
            'aggregation': 'parallel',
            'factor': {'cycle_count': 1 / bytes_per_cycle},
        },
    })

    return performance_dict
