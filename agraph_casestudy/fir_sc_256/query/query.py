from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
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

logger.remove()

configs = [f'config_{i}' for i in range(0, 7)]

area_df = None
energy_df = None
split_power_df = None
power_df = None
cycle_df = None
throughput_df = None

csv_path = 'agraph_casestudy/fir_sc_256/results/'
area_path = csv_path + 'area.csv'
power_path = csv_path + 'power.csv'
split_power_path = csv_path + 'split_power.csv'
cycle_path = csv_path + 'cycle.csv'
throughput_path = csv_path + 'throughput.csv'

# area_dict = {
#     '4': 11000,
#     '6': 12500,
#     '8': 14000,
#     '10': 15500,
#     '12': 17000,
#     '14': 18500,
#     '16': 20000
# }

area_dict_base = {
    '1024': 552004,
    '16': 8692
}

area_dict_slope = (area_dict_base['1024'] - area_dict_base['16']) / (int(list(area_dict_base.keys())[0]) - int(list(area_dict_base.keys())[1]))

area_dict = {
    '256': area_dict_base['16'] + area_dict_slope * (256 - 16),
    '32': area_dict_base['16'] + area_dict_slope * (32 - 16),
}


throughput_dict = {
    '4': 14,
    '6': 4.3,
    '8': 0.85,
    '10': .18,
    '12': .04,
    '14': .008,
    '16': .0018
}

dynamic_power_dict = {
    '32': 0.0000084,
}

leakage_power_dict = {
    '32': .0048,
}

power_dict = {
    '32': 0.0048 + 0.0000084,
}

# power_dict = {
#     '4': 3.9
# }

for config in configs:
    path = f'agraph_casestudy/fir_sc_256/description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    # workload = load_workload_dict(f'{path}/workload.yaml')

    array_dim_str = str(architecture['pnm']['instance'][0])
    bitwidth_dim_str = str(architecture['pnm']['query']['width'])

    workload = 'fir'
    event = 'fir'
    tag = 'tap'

    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=workload, event=event)
    runtime = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, event=event)
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag='mac')
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag='mac')
    area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag=tag)
    count = aggregate_event_count(event_graph=event_graph, workload=workload, event='mult')

    dynamic_energy['value'] = dynamic_energy['value'] / 1e9
    dynamic_energy['unit'] = 'J'

    leakage_power['value'] = leakage_power['value'] / 1e3
    leakage_power['unit'] = 'W'

    runtime['value'] = runtime['value'] / 1e3
    runtime['unit'] = 's'

    dynamic_power = {
        'value': dynamic_energy['value'] / runtime['value'],
        'unit': 'W'
    }

    power = {'value': leakage_power['value'] + dynamic_power['value'], 'unit': 'W'}

    area_eff = {
        'value': count / runtime['value'] / area['value'],
        'unit': 'ops/s/J'
    }

    throughput = {
        'value': count / runtime['value'] / 1e9,
        'unit': 'GOPs'
    }

    if area_df is None:
        area_df = pd.DataFrame(columns=['arch', 'bitwidth', f'area {area["unit"]}', f'baseline area {area["unit"]}', 'area_percent_dif'])

    if throughput_df is None:
        throughput_df = pd.DataFrame(columns=['arch', 'bitwidth', f'throughput {throughput["unit"]}', f'baseline_throughput {throughput["unit"]}', 'throughput_percent_dif'])

    if split_power_df is None:
        split_power_df = pd.DataFrame(columns=['arch', 'bitwidth', f'dynamic_power {power["unit"]}', f'leakage_power {power['unit']}', f'baseline dynamic power {power["unit"]}', 
                                         f'baseline leakage power {power['unit']}', 'dynamic power percent_dif', 'leakage power percent_dif'])
        
    if power_df is None:
        power_df = pd.DataFrame(columns=['arch', 'bitwidth', f'power {power["unit"]}', f'baseline power {power["unit"]}', 'power_percent_dif'])

    if array_dim_str in area_dict and array_dim_str != '16':
        area_df.loc[len(area_df)] = [
            array_dim_str,
            bitwidth_dim_str,
            area["value"],
            area_dict[array_dim_str],
            ((area["value"] - area_dict[array_dim_str]) / area_dict[array_dim_str] * 100)
        ]

    if bitwidth_dim_str in throughput_dict and array_dim_str != '16':
        throughput_df.loc[len(throughput_df)] = [
            array_dim_str,
            bitwidth_dim_str,
            throughput["value"],
            throughput_dict[bitwidth_dim_str],
            ((throughput["value"] - throughput_dict[bitwidth_dim_str]) / throughput_dict[bitwidth_dim_str] * 100)
        ]
    
    if array_dim_str in dynamic_power_dict and array_dim_str != '16':
        split_power_df.loc[len(split_power_df)] = [
            array_dim_str,
            bitwidth_dim_str,
            dynamic_power['value'],
            leakage_power['value'],
            dynamic_power_dict[array_dim_str],
            leakage_power_dict[array_dim_str],
            ((dynamic_power["value"] - dynamic_power_dict[array_dim_str]) / dynamic_power_dict[array_dim_str] * 100),
            ((leakage_power["value"] - leakage_power_dict[array_dim_str]) / leakage_power_dict[array_dim_str] * 100)
        ]

    if array_dim_str in power_dict and array_dim_str != '16':
        power_df.loc[len(power_df)] = [
            array_dim_str,
            bitwidth_dim_str,
            power["value"],
            power_dict[array_dim_str],
            ((power["value"] - power_dict[array_dim_str]) / power_dict[array_dim_str] * 100)
        ]


area_df.to_csv(area_path, index=False)
throughput_df.to_csv(throughput_path, index=False)
split_power_df.to_csv(split_power_path, index=False)
power_df.to_csv(power_path, index=False)
