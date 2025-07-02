from archx.metric import query_module_metric, aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from collections import OrderedDict
from archx.utils import get_prod, read_yaml
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
import statistics
import numpy as np

def geomean(dict_list: list[OrderedDict]) -> OrderedDict:
    geomean_dict = OrderedDict()

    for dict in dict_list:
        for key, value in dict.items():
            if key not in geomean_dict:
                if isinstance(value, str):
                    geomean_dict[key] = value
                else:
                    geomean_dict[key] = []

            if not isinstance(value, str):
                geomean_dict[key].append(value)

    for key, value in geomean_dict.items():
        if isinstance(value, list):
            geomean_dict[key] = statistics.geometric_mean(value)

    return geomean_dict

def query_cycle_count(event_graph, metric_dict, workload, event) -> OrderedDict:
    cycle_count_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='cycle_count', workload=workload, event=event)
    return cycle_count_dict['value']

def query_execution_time(event_graph, metric_dict, workload, event) -> OrderedDict:
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

def query_tag_power(tag, event_graph, metric_dict, workload, event) -> OrderedDict:
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=workload, tag=tag)
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=workload, tag=tag)
    
    power = (leakage_power + (dynamic_energy / execution_time)) * 10**3 # W -> mW

    return power

def compute_throughput(performance_metrics_dict: OrderedDict) -> OrderedDict:

    flops = performance_metrics_dict['flops']
    execution_time = performance_metrics_dict['execution_time']
    throughput = flops/execution_time
    performance_metrics_dict['throughput'] = throughput

    return performance_metrics_dict

def compute_throughput_efficiancy(performance_metrics_dict: OrderedDict) -> OrderedDict:

    flops = performance_metrics_dict['flops']
    energy = performance_metrics_dict['energy']
    power = performance_metrics_dict['power']
    execution_time = performance_metrics_dict['execution_time']

    throughput = flops/execution_time
    energy_efficiancy = throughput / (energy + (power * execution_time))
    power_efficiancy = throughput / (power + (energy / execution_time))
    
    performance_metrics_dict['throughput'] = throughput
    performance_metrics_dict['energy_efficiency'] = energy_efficiancy
    performance_metrics_dict['power_efficiency'] = power_efficiancy

    return performance_metrics_dict

def query_throughput_metrics(event_graph, metric_dict, module, workload, event)-> OrderedDict:
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = aggregate_event_count(event_graph=event_graph, workload=event, event=module)

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time
    })

    return performance_metrics_dict

def query_performance_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = aggregate_event_count(event_graph=event_graph, workload=event, event=module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
        'power': leakage_power,
        'cycle_count': cycle_count,
    })

    return performance_metrics_dict

def query_performance_gemm_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = aggregate_event_count(event_graph=event_graph, workload=event, event=module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
        'power': leakage_power,
    })

    return performance_metrics_dict

def query_performance_nonlinear_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = aggregate_event_count(event_graph=event_graph, workload=event, event=module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)

    flops = pe_count * 3 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
        'power': leakage_power,
        'cycle_count': cycle_count,
    })

    return performance_metrics_dict

def load_yaml(path):
    yaml_dict = OrderedDict({
        'architecture_dict': load_architecture_dict(path + '/architecture.yaml'),
        'event_graph': load_event_graph(path + '/checkpoint.gt'),
        'metric_dict': load_metric_dict(path + '/metric.yaml'),
        'event_dict': read_yaml(path + '/event.yaml'),
    })

    return yaml_dict