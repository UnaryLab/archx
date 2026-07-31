from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from archx.workload import load_workload_dict
from loguru import logger
import pandas as pd
import os

logger.remove()

configs = [f'config_{i}' for i in range(9)]

area_df = None
energy_df = None
power_df = None
cycle_df = None
runtime_df = None
memory_df = None

base_path = 'agraph/designs/mugi/'

csv_path = f'{base_path}results/'
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

area_path = csv_path + 'area.csv'
energy_path = csv_path + 'energy.csv'
power_path = csv_path + 'power.csv'
cycle_path = csv_path + 'cycle.csv'
runtime_path = csv_path + 'runtime.csv'
memory_path = csv_path + 'memory.csv'

area_dict = {
    '8x8': 31408.748,
    '8x16': 122808.210,
    '8x32': 492027.648,
}

leakage_power_dict = {
    '8x8': 0.63970840,
    '8x16': 2.52812598,
    '8x32': 10.22371720,
}

dynamic_power_dict = {
    '8x8': 21.8501604,
    '8x16': 90.22697716,
    '8x32': 370.03671122,
}

for config in configs:
    path = f'{base_path}description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    workload = load_workload_dict(f'{path}/workload.yaml')

    array_dim = architecture['and_gate']['instance']
    seq_len = workload['llama_2_7b']['configuration']['max_seq_len']
    array_dim_str = 'x'.join([str(dim) for dim in array_dim])

    workload = 'llama_2_7b'
    event = 'gemm'
    tag = 'array'

    # area
    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=workload, event=event)
    runtime = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, event=event)
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag=tag)
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag=tag)
    area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag=tag)
    count = aggregate_event_count(event_graph=event_graph, workload=workload, event='and_gate')

    memory_area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag='memory')
    memory_dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag='memory')
    memory_leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag='memory')

    syn_area = area_dict[array_dim_str] * 1e-6
    syn_power = leakage_power_dict[array_dim_str]
    syn_energy = dynamic_power_dict[array_dim_str] * runtime['value'] * 1e3

    if area_df is None:
        norm_syn_area = syn_area
        area_df = pd.DataFrame(columns=['arch', 'seq_len', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if energy_df is None:
        norm_syn_energy = syn_energy
        energy_df = pd.DataFrame(columns=['arch', 'seq_len', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if power_df is None:
        norm_syn_power = syn_power
        power_df = pd.DataFrame(columns=['arch', 'seq_len', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if cycle_df is None:
        cycle_df = pd.DataFrame(columns=['arch', 'seq_len', f'archx'])

    if runtime_df is None:
        runtime_df = pd.DataFrame(columns=['arch', 'seq_len', f'archx'])

    if memory_df is None:
        memory_df = pd.DataFrame(columns=['arch', 'seq_len', 'memory_area', 'memory_dynamic_energy', 'memory_leakage_power'])

    area_df.loc[len(area_df)] = [
        array_dim_str,
        seq_len,
        area["value"],
        syn_area,
        ((area["value"] - syn_area) / syn_area * 100),
        area["value"] / norm_syn_area,
        syn_area / norm_syn_area
    ]
    energy_df.loc[len(energy_df)] = [
        array_dim_str,
        seq_len,
        dynamic_energy["value"],
        syn_energy,
        ((dynamic_energy["value"] - syn_energy) / syn_energy * 100),
        dynamic_energy["value"] / norm_syn_energy,
        syn_energy / norm_syn_energy
    ]
    power_df.loc[len(power_df)] = [
        array_dim_str,
        seq_len,
        leakage_power["value"],
        syn_power,
        ((leakage_power["value"] - syn_power) / syn_power * 100),
        leakage_power["value"] / norm_syn_power,
        syn_power / norm_syn_power
    ]
    cycle_df.loc[len(cycle_df)] = [
        array_dim_str,
        seq_len,
        cycle_count["value"]
    ]
    runtime_df.loc[len(runtime_df)] = [
        array_dim_str,
        seq_len,    
        runtime["value"]
    ]
    memory_df.loc[len(memory_df)] = [
        array_dim_str,
        seq_len,
        memory_area["value"],
        memory_dynamic_energy["value"],
        memory_leakage_power["value"]
    ]

area_df.to_csv(area_path, index=False)
energy_df.to_csv(energy_path, index=False)
power_df.to_csv(power_path, index=False)
cycle_df.to_csv(cycle_path, index=False)
runtime_df.to_csv(runtime_path, index=False)
memory_df.to_csv(memory_path, index=False)