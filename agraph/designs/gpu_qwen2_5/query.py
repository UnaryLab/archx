from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from archx.workload import load_workload_dict
from loguru import logger
import pandas as pd
import os

logger.remove()

configs = ['config_0']

area_df = None
energy_df = None
power_df = None
cycle_df = None
runtime_df = None
memory_df = None

base_path = 'agraph/designs/gpu_qwen2_5/'

csv_path = f'{base_path}results/'
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

area_path = csv_path + 'area.csv'
energy_path = csv_path + 'energy.csv'
power_path = csv_path + 'power.csv'
cycle_path = csv_path + 'cycle.csv'
runtime_path = csv_path + 'runtime.csv'
memory_path = csv_path + 'memory.csv'

total_kernel_time = 86086.8

for config in configs:
    path = f'{base_path}description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    workload = load_workload_dict(f'{path}/workload.yaml')

    array_dim_str = 1

    workload = 'qwen2_5'
    event = 'qwen2_5'
    tag = 'kernel'

    # area
    print('energy')
    energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='energy', workload=workload, tag=tag)
    print('runtime')
    runtime = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, tag=tag)

    power = energy['value'] / runtime['value']

    energy['value'] *= 10**-6
    energy['unit'] = 'J'
    runtime['value'] *= 10**-3
    runtime['unit'] = 's'

    syn_power = 442.553
    syn_energy = 36.30764064
    syn_runtime = 86.0868

    if power_df is None:
        norm_syn_power = syn_power
        power_df = pd.DataFrame(columns=['arch', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if energy_df is None:
        norm_syn_energy = syn_energy
        energy_df = pd.DataFrame(columns=['arch', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if runtime_df is None:
        norm_syn_runtime = syn_runtime
        runtime_df = pd.DataFrame(columns=['arch', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    power_df.loc[len(power_df)] = [
        array_dim_str,
        power,
        syn_power,
        ((power - syn_power) / syn_power * 100),
        power / norm_syn_power,
        syn_power / norm_syn_power
    ]

    energy_df.loc[len(energy_df)] = [
        array_dim_str,
        energy['value'],
        syn_energy,
        ((energy['value'] - syn_energy) / syn_energy * 100),
        energy['value'] / norm_syn_energy,
        syn_energy / norm_syn_energy
    ]

    runtime_df.loc[len(runtime_df)] = [
        array_dim_str,
        runtime['value'],
        syn_runtime,
        ((runtime['value'] - syn_runtime) / syn_runtime * 100),
        runtime['value'] / norm_syn_runtime,
        syn_runtime / norm_syn_runtime
    ]

power_df.to_csv(power_path, index=False)
energy_df.to_csv(energy_path, index=False)
runtime_df.to_csv(runtime_path, index=False)