from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from collections import OrderedDict

def query_performance_metrics(event_graph, metric_dict, workload, event, module, tag, metrics) -> OrderedDict:

    performance_metrics_dict = OrderedDict()

    if 'cycle_count' in metrics:
        cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
        performance_metrics_dict['cycle_count'] = cycle_count
    if any(metric in metrics for metric in ['cycle_breakdown', 'compute_cycle_count', 'dram_stall_cycle_count', 'sram_cycle_count']):
        cycle_breakdown = query_cycle_breakdown(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
        if 'cycle_breakdown' in metrics:
            performance_metrics_dict['cycle_breakdown'] = cycle_breakdown
        if 'compute_cycle_count' in metrics:
            performance_metrics_dict['compute_cycle_count'] = cycle_breakdown['compute_cycle_count']
        if 'dram_stall_cycle_count' in metrics:
            performance_metrics_dict['dram_stall_cycle_count'] = cycle_breakdown['dram_stall_cycle_count']
        if 'sram_cycle_count' in metrics:
            performance_metrics_dict['sram_cycle_count'] = cycle_breakdown['sram_cycle_count']
    if 'execution_time' in metrics or 'throughput' in metrics:
        execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
        performance_metrics_dict['execution_time'] = execution_time
    if 'flops' in metrics or 'throughput' in metrics:
        pe_count = aggregate_event_count(event_graph=event_graph, workload=workload, event=module)
        flops = pe_count * 2 / 10**9 # GFLOPS
        performance_metrics_dict['flops'] = flops
    if 'energy' in metrics:
        dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=workload, tag=tag)
        performance_metrics_dict['energy'] = dynamic_energy
    if 'power' in metrics:
        leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=workload, tag=tag)
        performance_metrics_dict['power'] = leakage_power
    if 'throughput' in metrics:
        flops = pe_count * 2 / 10**9 # GFLOPS
        throughput = flops / execution_time
        performance_metrics_dict['throughput'] = throughput

    return performance_metrics_dict

def query_bandwidth(event_graph, metric_dict, workload, array_event, dram_event, width):
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=array_event)
    data_moved = aggregate_event_count(event_graph=event_graph, workload=workload, event=dram_event)

    bandwidth = (data_moved / execution_time) / 2**30

    return bandwidth, execution_time, data_moved

def query_cycle_count(event_graph, metric_dict, workload, event) -> OrderedDict:
    cycle_count_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='cycle_count', workload=workload, event=event)
    return cycle_count_dict['value']

def query_cycle_breakdown(event_graph, metric_dict, workload, event) -> OrderedDict:
    total_cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
    local_cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=None, event=event)
    component_event_counts = _component_event_counts(event_graph, event)
    scale = total_cycle_count / local_cycle_count if local_cycle_count != 0 else 0

    compute_cycle_count = sum(
        count * query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=None, event=name)
        for name, count in component_event_counts.items()
        if name.endswith('_arr')
    ) * scale
    sram_cycle_count = sum(
        count * query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=None, event=name)
        for name, count in component_event_counts.items()
        if name.endswith('_sram')
    ) * scale
    dram_stall_cycle_count = sum(
        count * query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=None, event=name)
        for name, count in component_event_counts.items()
        if name.endswith('_dram')
    ) * scale

    return OrderedDict({
        'cycle_count': total_cycle_count,
        'compute_cycle_count': compute_cycle_count,
        'sram_cycle_count': sram_cycle_count,
        'dram_stall_cycle_count': dram_stall_cycle_count,
        'unattributed_cycle_count': total_cycle_count - compute_cycle_count - sram_cycle_count - dram_stall_cycle_count,
    })

def _component_event_counts(event_graph, event):
    """Total multiplicity of every event reachable from `event`.

    Each node's count is the sum over all paths of the product of edge counts. This is
    accumulated in topological order so a node is expanded exactly once, after all of its
    parents have contributed. The previous version keyed its dedup on `(name, path_count)`
    and skipped expanding a node on a repeat arrival even though it had already added that
    arrival's count, so a node reached by two paths of equal multiplicity was counted twice
    but its subtree only once.
    """
    children_of = OrderedDict()
    stack = [event]
    while stack:
        name = stack.pop()
        if name in children_of:
            continue
        try:
            children = list(event_graph.get_out_neighbors(name))
        except ValueError:
            children = []
        children_of[name] = children
        stack.extend(children)

    pending_parents = {name: 0 for name in children_of}
    for name, children in children_of.items():
        for child in children:
            pending_parents[child] += 1

    component_counts = OrderedDict({event: 1})
    ready = [name for name, parents in pending_parents.items() if parents == 0]

    expanded = 0
    while ready:
        name = ready.pop()
        expanded += 1
        count = component_counts.get(name, 0)
        for child in children_of[name]:
            try:
                child_count = count * event_graph.get_edge_count(name, child)
            except ValueError:
                child_count = count
            component_counts[child] = component_counts.get(child, 0) + child_count
            pending_parents[child] -= 1
            if pending_parents[child] == 0:
                ready.append(child)

    if expanded != len(children_of):
        raise ValueError(
            f'event graph reachable from {event!r} contains a cycle; '
            f'{len(children_of) - expanded} of {len(children_of)} nodes were not expanded'
        )

    return component_counts

def query_execution_time(event_graph, metric_dict, workload, event, tag=None) -> OrderedDict:
    execution_time_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='runtime', workload=workload, event=event)
    return execution_time_dict['value'] / 10**3 # ms -> s

def query_dynamic_energy(event_graph, metric_dict, workload, event=None, tag=None) -> OrderedDict:
    if tag is None:
        dynamic_energy_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='dynamic_energy', workload=workload, event=event)
    else:
        dynamic_energy_dict = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='dynamic_energy', workload=workload, tag=tag)
    return dynamic_energy_dict['value'] / 10**9 # nJ -> J

def query_leakage_power(event_graph, metric_dict, workload, event=None, tag=None) -> OrderedDict:
    if tag is None:
        leakage_power = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='leakage_power', workload=workload, event=event)
    else:
        leakage_power =  aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='leakage_power', workload=workload, tag=tag)
    return leakage_power['value'] / 10**3 # mW -> W

def query_area(event_graph, metric_dict, workload=None, tag=None, module=None) -> float:

    if module is not None:
        area = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='area', workload=workload, event=module)['value']

    elif tag is not None:
        area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='area', workload=workload, tag=tag)['value']

    return area


# ---------------------------------------------------------------------------------------
# DRAM bandwidth demand. Shared by fig_4_query, which reports it for one design point, and
# fig_6_query, which reports it across the whole array-shape grid so fig_4's point can be
# selected on it. One implementation, so the two can never disagree about what the number
# means -- fig_4_query cross-checks its own result against fig_6's CSV for the point it
# picks.
# ---------------------------------------------------------------------------------------

# memory.py bills one DRAM event per byte, so an edge count is a byte count.
DRAM_LANES = ['dram_input_read', 'dram_weight_read', 'dram_output_read', 'dram_output_write']

def gemm_phase(gemm):
    """'prefill' or 'decode' for a GEMM event name.

    model.py names every GEMM with a `_pf` / `_dc` suffix, which is the only phase marker
    in the graph; anything else is a naming change this function must be told about.
    """
    if gemm.endswith('_pf'):
        return 'prefill'
    if gemm.endswith('_dc'):
        return 'decode'
    raise ValueError(f'GEMM {gemm!r} has neither a _pf nor a _dc suffix; '
                     f'it cannot be placed in a phase')

def gemm_demand(event_graph, metric_dict, workload_name):
    """Per-GEMM (name, phase, bytes, window cycles), the tuple every reduction is built on.

    The window is max(array compute cycles, SRAM port cycles). DRAM is deliberately
    EXCLUDED, being the thing being sized: including it would let the window stretch to
    accommodate whatever traffic there is, and the answer would always be 'the declared
    channel is adequate'.
    """
    demand = []

    for name in sorted(event_graph.get_all_node_names()):
        if not name.endswith('_dram'):
            continue
        gemm = name[:-len('_dram')]

        # Multiplicity comes off '_dram', which hangs under `llama` alone. '_arr' is
        # reachable through the `llama_array` view as well, and aggregate_event_count
        # sums over every path, so reading it there would double the GEMM.
        multiplicity = aggregate_event_count(
            event_graph=event_graph, workload=workload_name, event=name)
        if multiplicity <= 0:
            continue

        gemm_bytes = sum(event_graph.get_edge_count(name, lane)
                         for lane in DRAM_LANES) * multiplicity
        # workload=None is one instance of the event; the multiplicity scales it
        array_cycles = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                         workload=None, event=f'{gemm}_arr')
        sram_cycles = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                        workload=None, event=f'{gemm}_sram')
        window = max(array_cycles, sram_cycles) * multiplicity
        demand.append((gemm, gemm_phase(gemm), gemm_bytes, window))

    return demand

def bandwidth_gbs(data_moved, window_cycles, frequency_mhz):
    """Decimal GB/s, matching the unit the declared dram bandwidth is written in."""
    if data_moved <= 0 or window_cycles <= 0:
        return 0
    seconds = window_cycles / (float(frequency_mhz) * 1e6)
    return (data_moved / seconds) / 1e9

def bandwidth_summary(demand, frequency_mhz):
    """{average, peak, peak_phase, burst, burst_gemm, total_bytes, total_window}.

    average is pooled over every GEMM; peak is pooled per PHASE and the hungrier phase
    taken. The peak is NOT a per-GEMM maximum: double buffering (mapping.py charges only
    `bank // 2` as usable, because the other half is filling) smooths traffic across GEMM
    boundaries, and a per-GEMM row is not contiguous execution anyway -- its multiplicity
    is how many times it runs across the model, so a decode GEMM is one small instance per
    token, interleaved with its neighbours. `burst` keeps the superseded per-GEMM number
    so the gap between the two stays visible.
    """
    total_bytes = sum(gemm_bytes for _, _, gemm_bytes, _ in demand)
    total_window = sum(window for _, _, _, window in demand)

    phase_totals = {}
    for _, phase, gemm_bytes, window in demand:
        totals = phase_totals.setdefault(phase, [0.0, 0.0])
        totals[0] += gemm_bytes
        totals[1] += window
    peak_phase, peak = max(
        ((phase, bandwidth_gbs(phase_bytes, phase_window, frequency_mhz))
         for phase, (phase_bytes, phase_window) in phase_totals.items()),
        key=lambda pair: pair[1])

    burst_gemm, burst = max(
        ((gemm, bandwidth_gbs(gemm_bytes, window, frequency_mhz))
         for gemm, _, gemm_bytes, window in demand),
        key=lambda pair: pair[1])

    return {
        'total_bytes': total_bytes,
        'total_window': total_window,
        'average': bandwidth_gbs(total_bytes, total_window, frequency_mhz),
        'peak': peak,
        'peak_phase': peak_phase,
        'burst': burst,
        'burst_gemm': burst_gemm,
    }


# ---------------------------------------------------------------------------------------
# fig_4's selection criteria. fig_4 is drawn three times, once per criterion, because
# "the best design" is not one question -- and the three disagree sharply, which is the
# point of showing all three rather than picking one.
#
#   runtime     argmin of total wall-clock. Does NOT normalise for work: prefill runs
#               M = batch_size * prefill_seq_len rows and decode M = batch_size
#               (model.py), so a batch-32 point does a sixteenth of the work of a
#               batch-512 one and wins on raw time by default. This is the literal
#               "finishes soonest" reading, and it is here to be contrasted with
#               throughput rather than to stand alone.
#   avg_band    argmax of required average DRAM bandwidth. A DEMAND, not a speed: a shape
#               can score high by being fast OR by having worse reuse and moving more
#               bytes, so this selects the most memory-hungry design -- what the channel
#               must be sized to survive.
#   throughput  argmax of generated tokens per second, which credits a bigger batch for
#               the extra work it does instead of penalising it for the extra time. This
#               is the serving metric, and the one that answers "which machine would you
#               build". Being work-per-time it is the exact inverse of
#               runtime_ms_per_sequence, so it and the runtime criterion above bracket the
#               batch question from both ends.
#
# Both scripts read this list, so a criterion cannot exist in the query and not the figure.
# ---------------------------------------------------------------------------------------
FIG_4_CRITERIA = [
    {
        'key': 'runtime',
        'column': 'runtime_ms',
        'direction': 'min',
        'label': 'lowest total runtime',
        'note': 'not work-normalised: favours small batches',
    },
    {
        'key': 'avg_band',
        'column': 'average_bandwidth',
        'direction': 'max',
        'label': 'highest average DRAM bandwidth',
        'note': 'most memory-hungry, not fastest',
    },
    {
        'key': 'throughput',
        'column': 'throughput_tokens_per_s',
        'direction': 'max',
        'label': 'highest throughput',
        'note': 'generated tokens per second, work-normalised',
    },
]

def fig_4_paths(criterion_key):
    """(csv, scientific csv, figure) for one fig_4 criterion."""
    return (f'zoo/chiplet4ai/results/csv/dram_bandwidth_metrics_{criterion_key}.csv',
            f'zoo/chiplet4ai/results/csv/dram_bandwidth_metrics_{criterion_key}_scientific.csv',
            f'zoo/chiplet4ai/results/figs/fig_4_{criterion_key}.pdf')


# ---------------------------------------------------------------------------------------
# Design-point selection: the SMALLEST array within a margin of the best, not the best.
#
# WHY NOT JUST THE ARGMAX/ARGMIN. These metrics are close to flat across the shape grid --
# at DeepSeek's max context the top few configurations sit within 0.1% of each other while
# spanning a 4x range in PE count -- so a strict extremum reports a 262,144-PE array as
# "the answer" when a 131,072-PE one was 0.04% behind it. Taking the smallest array inside
# a tolerance band reports the configuration that is actually worth building.
#
# THE MARGIN IS LOAD-BEARING, so it is one constant rather than a per-criterion tweak, and
# it is worth re-checking against the data when the model changes. At 1%:
#   avg_band     DeepSeek 512x512 -> 256x512 (half the PEs, 0.04% less bandwidth); the
#                Llamas are unmoved at 32x512, their runners-up being 4-10% behind.
#   runtime      collapses to 4K-32K PE arrays for every model. That is not a bug: at
#                batch 32 these designs are DRAM-bound and the array barely matters, so
#                the honest answer is that a large one buys nothing.
#   throughput   DeepSeek 512x512 -> 256x128, the Llamas 128K-256K -> 16K-256K.
# Set it to 0 to recover the strict extremum.
SELECTION_MARGIN = 0.01

def select_design_point(group, column, direction, margin=SELECTION_MARGIN):
    """The row with the fewest PEs whose `column` is within `margin` of the best.

    Ties on PE count are broken by the metric itself, so among equally sized arrays the
    better one still wins. `group` must carry a `pe_count` column.
    """
    best = group[column].max() if direction == 'max' else group[column].min()
    # a relative band, so it means the same thing for a GB/s column and a milliseconds one
    within = (group[column] >= best * (1 - margin) if direction == 'max'
              else group[column] <= best * (1 + margin))
    band = group[within]
    band = band.sort_values(['pe_count', column], ascending=[True, direction == 'min'])
    return band.iloc[0], best
