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
