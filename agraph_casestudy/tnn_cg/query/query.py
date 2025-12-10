from archx.metric import aggregate_event_metric, aggregate_tag_metric
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from loguru import logger
import pandas as pd

# systolic array case study
# area
# leakage power
# dynamic power
# cycle count



configs = ['config_0', 'config_1', 'config_2']

area_df = None
energy_df = None
power_df = None
cycle_df = None
runtime_df = None

csv_path = 'agraph_casestudy/tnn_cg/results/'
area_path = csv_path + 'area.csv'
energy_path = csv_path + 'energy.csv'
power_path = csv_path + 'power.csv'
cycle_path = csv_path + 'cycle.csv'
runtime_path = csv_path + 'runtime.csv'

area_dict = {
    '96x2': 15684.581 * 1e-6,
    '152x2': 24706.452 * 1e-6,
    '343x2': 55506.177 * 1e-6
}

leakage_power_dict = {
    '96x2': 0.00071134,
    '152x2': 0.00112181,
    '343x2': 0.00252238
}

dynamic_power_dict = {
    '96x2': 0.43742493,
    '152x2': 0.70338774,
    '343x2': 1.49137133
}

arch_area_norm = area_dict['96x2']
arch_power_norm = leakage_power_dict['96x2']
arch_energy_norm = 0

for config in configs:
    path = f'agraph_casestudy/tnn_cg/description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    # workload = load_workload_dict(f'{path}/workload.yaml')

    array_dim = architecture['stdp']['instance']
    array_dim_str = 'x'.join([str(dim) for dim in array_dim])
    workload = 'tnn'
    event = 'tnn'
    tag = 'column'

    # area
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag=tag)
    logger.remove()
    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=workload, event=event)
    runtime = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, event=event)
    
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag=tag)
    area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag=tag)


    syn_area = area_dict[array_dim_str]
    syn_power = leakage_power_dict[array_dim_str]
    syn_energy = dynamic_power_dict[array_dim_str]  * runtime['value'] * 1e3

    if arch_energy_norm == 0:
        arch_energy_norm = syn_energy

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
        area['value'] / arch_area_norm,
        syn_area / arch_area_norm
    ]
    energy_df.loc[len(energy_df)] = [
        array_dim_str,
        dynamic_energy["value"],
        syn_energy,
        ((dynamic_energy["value"] - syn_energy) / syn_energy * 100),
        dynamic_energy['value'] / arch_energy_norm,
        syn_energy / arch_energy_norm
    ]
    power_df.loc[len(power_df)] = [
        array_dim_str,
        leakage_power["value"],
        syn_power,
        ((leakage_power["value"] - syn_power) / syn_power * 100),
        leakage_power['value'] / arch_power_norm,
        syn_power / arch_power_norm
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