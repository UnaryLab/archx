from archx.metric import aggregate_event_metric, aggregate_tag_metric
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from loguru import logger
import pandas as pd
import os

if not os.path.exists('agraph_casestudy/fft_course_grain/results/'):
    os.makedirs('agraph_casestudy/fft_course_grain/results/')

logger.remove()

configs = ['config_0', 'config_1', 'config_2', 'config_3']

area_df = None
energy_df = None
power_df = None
cycle_df = None
runtime_df = None

csv_path = 'agraph_casestudy/fft_course_grain/results/'
area_path = csv_path + 'area.csv'
energy_path = csv_path + 'energy.csv'
power_path = csv_path + 'power.csv'
cycle_path = csv_path + 'cycle.csv'
runtime_path = csv_path + 'runtime.csv'

area_dict = {
    '8x3': 24198.020, #um^2
    '16x4': 64192.450,
    '32x5': 160661.340,
    '64x6': 386278.284
}

leakage_power_dict = {
    '8x3': 0.57966627, #um^2
    '16x4': 1.52456516,
    '32x5': 3.80941356,
    '64x6': 9.17620493,
}

dynamic_power_dict = {
    '8x3': 32.5927692, #um^2
    '16x4': 88.17718782,
    '32x5': 222.90268963,
    '64x6': 543.0962468,
}

for config in configs:
    path = f'agraph_casestudy/fft_course_grain/description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    # workload = load_workload_dict(f'{path}/workload.yaml')

    array_dim = architecture['pe']['instance']
    array_dim = [array_dim[0] * 2, array_dim[1]]
    array_dim_str = 'x'.join([str(dim) for dim in array_dim])

    workload = 'butterfly'
    event = 'butterfly'
    tag = 'array'

    # area
    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=workload, event=event)
    runtime = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, event=event)
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag=tag)
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag=tag)
    area = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, event=event)
    
    syn_area = area_dict[array_dim_str] * 1e-6
    syn_power = leakage_power_dict[array_dim_str]
    syn_energy = dynamic_power_dict[array_dim_str] * runtime['value'] * 1e3

    if area_df is None:
        norm_syn_area = syn_area
        area_df = pd.DataFrame(columns=['arch', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if energy_df is None:
        norm_syn_energy = syn_energy
        energy_df = pd.DataFrame(columns=['arch', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if power_df is None:
        norm_syn_power = syn_power
        power_df = pd.DataFrame(columns=['arch', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if cycle_df is None:
        cycle_df = pd.DataFrame(columns=['arch', f'archx'])

    if runtime_df is None:
        runtime_df = pd.DataFrame(columns=['arch', f'archx'])

    area_df.loc[len(area_df)] = [
        array_dim_str,
        area["value"],
        syn_area,
        ((area["value"] - syn_area) / syn_area * 100),
        area["value"] / norm_syn_area,
        syn_area / norm_syn_area
    ]
    energy_df.loc[len(energy_df)] = [
        array_dim_str,
        dynamic_energy["value"],
        syn_energy,
        ((dynamic_energy["value"] - syn_energy) / syn_energy * 100),
        dynamic_energy["value"] / norm_syn_energy,
        syn_energy / norm_syn_energy
    ]
    power_df.loc[len(power_df)] = [
        array_dim_str,
        leakage_power["value"],
        syn_power,
        ((leakage_power["value"] - syn_power) / syn_power * 100),
        leakage_power["value"] / norm_syn_power,
        syn_power / norm_syn_power
    ]
    cycle_df.loc[len(cycle_df)] = [
        array_dim_str,
        cycle_count["value"]
    ]
    runtime_df.loc[len(runtime_df)] = [
        array_dim_str,
        runtime["value"]
    ]

area_df.to_csv(area_path, index=False)
energy_df.to_csv(energy_path, index=False)
power_df.to_csv(power_path, index=False)
cycle_df.to_csv(cycle_path, index=False)
runtime_df.to_csv(runtime_path, index=False)