from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from loguru import logger
import pandas as pd
import os

# systolic array case study
# area
# leakage power
# dynamic power
# cycle count

if not os.path.exists('agraph_casestudy/systolic_pe/results/'):
    os.makedirs('agraph_casestudy/systolic_pe/results/')

logger.remove()

configs = ['config_0', 'config_1', 'config_2', 'config_3']

area_df = None
energy_df = None
power_df = None
cycle_df = None
runtime_df = None

csv_path = 'agraph_casestudy/systolic_pe/results/'
area_path = csv_path + 'area.csv'
energy_path = csv_path + 'energy.csv'
power_path = csv_path + 'power.csv'
cycle_path = csv_path + 'cycle.csv'
runtime_path = csv_path + 'runtime.csv'

area_dict = {
    '4': 111.720,
    '8': 326.400,
    '16': 942.684,
    '32': 3032.382
}

leakage_power_dict = {
    '4': 0.00178350,
    '8': 0.00520932,
    '16': 0.01704102,
    '32': 0.05582214
}

dynamic_power_dict = {
    '4': 0.04743902,
    '8': 0.15073312,
    '16': 0.71114695,
    '32': 3.40805862
}

for config in configs:
    path = f'agraph_casestudy/systolic_pe/description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    # workload = load_workload_dict(f'{path}/workload.yaml')

    bitwidth_str = str(architecture['multiplier']['query']['width'])

    workload = 'gemm'
    event = 'gemm'
    tag = 'array'

    # area
    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=workload, event=event)
    runtime = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, event=event)
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag=tag)
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag=tag)
    area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag=tag)
    count = aggregate_event_count(event_graph=event_graph, workload=workload, event='multiplier')

    syn_area = area_dict[bitwidth_str] * 1e-6
    syn_power = leakage_power_dict[bitwidth_str]
    syn_energy = dynamic_power_dict[bitwidth_str] * runtime['value'] * 1e3

    if area_df is None:
        norm_syn_area = syn_area
        area_df = pd.DataFrame(columns=['bitwidth', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if energy_df is None:
        norm_syn_energy = syn_energy
        energy_df = pd.DataFrame(columns=['bitwidth', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if power_df is None:
        norm_syn_power = syn_power
        power_df = pd.DataFrame(columns=['bitwidth', f'archx', f'pnr', 'percent_error', 'archx_norm', 'syn_norm'])

    if cycle_df is None:
        cycle_df = pd.DataFrame(columns=['bitwidth', f'archx'])

    if runtime_df is None:
        runtime_df = pd.DataFrame(columns=['arch', f'archx'])

    area_df.loc[len(area_df)] = [
        bitwidth_str,
        area["value"],
        syn_area,
        ((area["value"] - syn_area) / syn_area * 100),
        area["value"] / norm_syn_area,
        syn_area / norm_syn_area
    ]
    energy_df.loc[len(energy_df)] = [
        bitwidth_str,
        dynamic_energy["value"],
        syn_energy,
        ((dynamic_energy["value"] - syn_energy) / syn_energy * 100),
        dynamic_energy["value"] / norm_syn_energy,
        syn_energy / norm_syn_energy
    ]
    power_df.loc[len(power_df)] = [
        bitwidth_str,
        leakage_power["value"],
        syn_power,
        ((leakage_power["value"] - syn_power) / syn_power * 100),
        leakage_power["value"] / norm_syn_power,
        syn_power / norm_syn_power
    ]
    cycle_df.loc[len(cycle_df)] = [
        bitwidth_str,
        cycle_count["value"]
    ]
    runtime_df.loc[len(runtime_df)] = [
        bitwidth_str,
        runtime["value"]
    ]

area_df.to_csv(area_path, index=False)
energy_df.to_csv(energy_path, index=False)
power_df.to_csv(power_path, index=False)
cycle_df.to_csv(cycle_path, index=False)
runtime_df.to_csv(runtime_path, index=False)