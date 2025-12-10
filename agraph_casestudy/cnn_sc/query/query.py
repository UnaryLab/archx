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

configs = ['config_0']

area_df = None
power_df = None
throughput_df = None
power_breakdown_df = None
area_breakdown_df = None

csv_path = 'agraph_casestudy/cnn_sc/results/'
area_path = csv_path + 'area.csv'
power_path = csv_path + 'power.csv'
throughtput_path = csv_path + 'throughput.csv'
power_breakdown = csv_path + 'power_breakdown.csv'
area_breakdown = csv_path + 'area_breakdown.csv'

area_dict = {
    '64x64x32': 15400000
}

power_dict = {
    '64x64x32': 3.9
}

throughput_dict = {
    '64x64x32': 409
}

modules = ['mac', 'mac_splitter', 'weight_splitter', 'input_splitter', 'nrdo']

for config in configs:
    path = f'agraph_casestudy/cnn_sc/description/runs/{config}'
    event_graph = load_event_graph(f'{path}/checkpoint.gt')
    architecture = load_architecture_dict(f'{path}/architecture.yaml')
    metric = load_metric_dict(f'{path}/metric.yaml')
    # workload = load_workload_dict(f'{path}/workload.yaml')

    array_dim = architecture['mac']['instance']
    array_dim_str = 'x'.join([str(dim) for dim in array_dim])

    workload = 'vgg16'
    event = 'vgg16'
    tag = 'mac'

    cycle_count = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='cycle_count', workload=workload, event=event)
    runtime = aggregate_event_metric(event_graph=event_graph, metric_dict=metric, metric='runtime', workload=workload, event=event)
    dynamic_energy = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag=tag)
    leakage_power = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag=tag)
    area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag='pe')
    overhead_area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag='array')
    count = aggregate_event_count(event_graph=event_graph, workload=workload, event='mac')

    mod_area = {}
    mod_power = {}

    runtime['value'] = runtime['value'] / 1e3
    runtime['unit'] = 's'

    for mod in modules:
        print(mod)
        mod_area[mod] = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='area', workload=workload, tag=mod)['value']
        leakage = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='leakage_power', workload=workload, tag=mod)['value']
        dynamic = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric, metric='dynamic_energy', workload=workload, tag=mod)['value']

        mod_power[mod] = (leakage / 1e3)

        print(dynamic)
        print(leakage)
        print(runtime)
        print(mod_power[mod])
        print()

    throughtput = (count / 1e12) / (runtime['value'] / 1e3)

    dynamic_energy['value'] = dynamic_energy['value'] / 1e9
    dynamic_energy['unit'] = 'J'

    leakage_power['value'] = leakage_power['value'] / 1e3
    leakage_power['unit'] = 'W'

    

    dynamic_power = {
        'value': dynamic_energy['value'] / runtime['value'],
        'unit': 'W'
    }
    dynamic_power['value'] = 0

    power = {'value': leakage_power['value'] + dynamic_power['value'], 'unit': 'W'}

    print(dynamic_energy)
    print
    print(leakage_power)
    print(runtime)
    print(power)

    if area_df is None:
        area_df = pd.DataFrame(columns=['config', 'arch', f'area {area["unit"]}', f'baseline area {area["unit"]}', 'area_percent_dif'])

    if power_df is None:
        power_df = pd.DataFrame(columns=['arch', f'dynamic_power {power["unit"]}', f'leakage_power {power["unit"]}', f'power {power["unit"]}', f'baseline_power {power["unit"]} {power["unit"]}', 'power_percent_dif'])

    if throughput_df is None:
        throughput_df = pd.DataFrame(columns=['arch', f'throughput {throughtput}', f'baseline_throughput {throughput_dict[array_dim_str]}', 'throughput_percent_dif'])



    
    area_df.loc[len(area_df)] = [
        'base',
        array_dim_str,
        area["value"],
        area_dict[array_dim_str],
        ((area["value"] - area_dict[array_dim_str]) / area_dict[array_dim_str] * 100)
    ]
    area_df.loc[len(area_df)] = [
        'with_overhead',
        array_dim_str,
        overhead_area["value"],
        area_dict[array_dim_str],
        ((overhead_area["value"] - area_dict[array_dim_str]) / area_dict[array_dim_str] * 100)
    ]
    power_df.loc[len(power_df)] = [
        array_dim_str,
        dynamic_power["value"],
        leakage_power["value"],
        power["value"],
        power_dict[array_dim_str],
        ((power["value"] - power_dict[array_dim_str]) / power_dict[array_dim_str] * 100)
    ]
    throughput_df.loc[len(throughput_df)] = [
        array_dim_str,
        throughtput,
        throughput_dict[array_dim_str],
        ((throughtput - throughput_dict[array_dim_str]) / throughput_dict[array_dim_str] * 100)
    ]
    power_breakdown_df = pd.DataFrame.from_dict(mod_power, orient='index')
    area_breakdown_df = pd.DataFrame.from_dict(mod_area, orient='index')

area_df.to_csv(area_path, index=False)
power_df.to_csv(power_path, index=False)
throughput_df.to_csv(throughtput_path, index=False)
power_breakdown_df.to_csv(power_breakdown, index=True)
area_breakdown_df.to_csv(area_breakdown, index=True)
