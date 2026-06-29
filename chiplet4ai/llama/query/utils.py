from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from collections import OrderedDict

def query_performance_metrics(event_graph, metric_dict, workload, event, module, tag, metrics) -> OrderedDict:

    performance_metrics_dict = OrderedDict()

    if 'cycle_count' in metrics:
        cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
        performance_metrics_dict['cycle_count'] = cycle_count
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

def query_area(event_graph, metric_dict, workload=None, tag=None, module=None) -> np.float64:

    if module is not None:
        area = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='area', workload=workload, event=module)['value']

    elif tag is not None:
        area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='area', workload=workload, tag=tag)['value']

    return area