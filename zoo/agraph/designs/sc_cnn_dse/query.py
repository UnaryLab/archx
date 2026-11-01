from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from archx.workload import load_workload_dict
from loguru import logger
import pandas as pd
import os

logger.remove()

base_path = 'zoo/agraph/designs/sc_cnn_dse/'
event = 'cnn'
tag = 'array'

configurations = pd.read_csv(f'{base_path}description/configurations.csv')

rows = []
for row in configurations.itertuples():
    path = row.run_path
    event_graph = load_event_graph(f'{path}/checkpoint.json')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    bitwidth = load_workload_dict(f'{path}/workload.yaml')['configuration']['bitwidth']

    array_dim = architecture['mac']['instance']

    area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=event, tag=tag)['value']
    # runtime is in ms, leakage_power in mW, dynamic_energy in nJ
    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=event, event=event)['value']
    latency = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=event, event=event)['value'] / 1e3
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=event, tag=tag)['value'] / 1e3
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=event, tag=tag)['value'] / 1e9
    count = aggregate_event_count(event_graph=event_graph, workload=event, event='mac')

    dynamic_power = dynamic_energy / latency
    leakage_energy = leakage_power * latency

    rows.append({
        'workload': row.workload_name,
        'arch': 'x'.join([str(dim) for dim in array_dim]),
        'dim_x': array_dim[0],
        'dim_y': array_dim[1],
        'dim_z': array_dim[2],
        'area jj': area,
        'dynamic_power W': dynamic_power,
        'leakage_power W': leakage_power,
        'power W': dynamic_power + leakage_power,
        'dynamic_energy J': dynamic_energy,
        'leakage_energy J': leakage_energy,
        'energy J': dynamic_energy + leakage_energy,
        'throughput TMAC/s': count / latency / 1e12,
        'latency s': latency,
        'utilization': count * 2**(bitwidth - 1) / (cycle_count * array_dim[0] * array_dim[1] * array_dim[2]),
    })

os.makedirs(f'{base_path}results', exist_ok=True)
dse_df = pd.DataFrame(rows).sort_values(['workload', 'dim_x', 'dim_y', 'dim_z'])
dse_df.to_csv(f'{base_path}results/dse.csv', index=False)
