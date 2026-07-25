from zoo.llm.results.query.utils import query_area, query_performance_gemm_metrics, query_performance_nonlinear_metrics, load_yaml, compute_throughput_efficiancy, query_execution_time
import pandas as pd
from collections import OrderedDict
import os

def query(input_path, output_path):
    network_list = ['single_node', 'multi_node_4x4', 'multi_node_8x8']
    tensor_network_list = ['single_node', 'multi_node_2x1', 'multi_node_2x2']

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['64x8', '128x8', '256x8']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16', '64x64']
    baseline_subarch_list = ['mac', 'figna']

    throughput_module = OrderedDict({
        'mugi': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'magnitude_register'}),
        'carat': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'accumulator_vector'}),
        'systolic': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'accumulator_vector'}),
        'simd': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'accumulator_vector'}),
        'tensor': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'accumulator_vector'}),
    })

    model = 'llama_2_70b_GQA'
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'

    kv_heads = 'kv_heads_8'

    comprehensive_table_df = pd.DataFrame()

    for arch in (vlp_list + baseline_list) + ['tensor']:
        for network in network_list if arch != 'tensor' else tensor_network_list:
            for subarch in (baseline_subarch_list if arch in baseline_list else ['vlp'] if arch == 'mugi' else ['']):
                for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch == 'tensor' else ['']):
                    if arch_dim in ['64x64', '128x8'] and network != 'single_node':
                        continue
                    if arch_dim in ['8x8', '64x8'] and network == 'single_node':
                        continue
                    if arch_dim in ['256x8', '16x16'] and network == 'multi_node_8x8':
                        continue
                    if network == 'multi_node_8x8':
                        continue
                    if arch_dim in ['64x8', '8x8'] and network == 'multi_node_4x4':
                        continue
                    if network == 'multi_node_2x2':
                        continue

                    gemm_module = throughput_module[arch]['gemm']
                    nonlinear_module = throughput_module[arch]['nonlinear']
                    
                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/kv_heads_8/{termination_path}/')
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    runtime = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model)  # convert to seconds
                    tokens = int(batch_size.split('_')[-1]) * int(max_seq_len.split('_')[-1])
                    gemm_performance_metrics_dict = query_performance_gemm_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='gemm', module=gemm_module)
                    nonlinear_performance_metrics_dict = query_performance_nonlinear_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='nonlinear', module=nonlinear_module)

                    assert gemm_performance_metrics_dict['power'] == nonlinear_performance_metrics_dict['power'], "Power mismatch between gemm and nonlinear modules"

                    # performance_metrics_dict = OrderedDict({
                    #     'flops': gemm_performance_metrics_dict['flops'] + nonlinear_performance_metrics_dict['flops'],
                    #     'execution_time': gemm_performance_metrics_dict['execution_time'] + nonlinear_performance_metrics_dict['execution_time'],
                    #     'energy': gemm_performance_metrics_dict['energy'] + nonlinear_performance_metrics_dict['energy'],
                    #     'power': gemm_performance_metrics_dict['power']
                    # })

                    performance_metrics_dict = OrderedDict({
                        'flops': tokens,
                        'execution_time': runtime,
                        'energy': gemm_performance_metrics_dict['energy'] + nonlinear_performance_metrics_dict['energy'],
                        'power': gemm_performance_metrics_dict['power']
                    })

                    throughput_eff_dict = compute_throughput_efficiancy(performance_metrics_dict=performance_metrics_dict)
                    area = query_area(event_graph=event_graph, metric_dict=metric_dict, tag='onchip')

                    comprehensive_table_dict = OrderedDict({
                        'arch': arch,
                        'subarch': subarch,
                        'network': network,
                        'arch_dim': arch_dim,
                        'throughput': throughput_eff_dict['throughput'],
                        'energy_efficiency': throughput_eff_dict['energy_efficiency'] * (10**6), # GFLOPS/s/mJ
                        'power_efficiency': throughput_eff_dict['power_efficiency'],
                        'area': area,
                        'flops': performance_metrics_dict['flops'],
                        'execution_time': performance_metrics_dict['execution_time'],
                        'energy': performance_metrics_dict['energy'],
                        'power': performance_metrics_dict['power'],
                        'gemm_flops': gemm_performance_metrics_dict['flops'],
                        'gemm_execution_time': gemm_performance_metrics_dict['execution_time'],
                        'nonlinear_flops': nonlinear_performance_metrics_dict['flops'],
                        'nonlinear_execution_time': nonlinear_performance_metrics_dict['execution_time'],
                    })

                    comprehensive_table_df = pd.concat([comprehensive_table_df, pd.DataFrame(comprehensive_table_dict, index=[0])])
                    
    comprehensive_table_df.to_csv(output_path + 'comprehensive_table.csv', index=False)

def table(input_path: str, output_path: str):
    df = pd.read_csv(input_path + 'comprehensive_table.csv')
    
    vlp_single_node_list = []
    baseline_single_node_list = []
    baseline_sa_list = []
    vlp_multi_node_list = []
    sa_multi_node_list = []
    sa_f_multi_node_list = []
    sd_multi_node_list = []
    sd_f_multi_node_list = []
    tensor_single_node_list = []
    tensor_multi_node_list = []
    for index, row in df.iterrows():
        arch_label = row['arch'].capitalize()
        dim_label = row['arch_dim'].split('_')[0]
        subarch_label = row['subarch'] if pd.notna(row['subarch']) else ''
        network = row['network']

        if arch_label == 'Systolic':
            arch_label = 'SA'
        elif arch_label == 'Simd':
            arch_label = 'SD'
        if arch_label == 'Mugi':
            arch_label = '\\name'
        if arch_label == 'tensor':
            arch_label = 'Tensor'

        subarch_label = '-F' if subarch_label == 'figna' else ''

        design = arch_label + subarch_label + ' (' + dim_label.split('x')[0] + ')'
        throughput = row['throughput']
        area = row['area']
        energy_efficiency = row['energy_efficiency']
        power_efficiency = row['power_efficiency']
        if 'multi_node' in network:
            network_label = network.split('_')[2].split('x')
            network_label = ' & ' + network_label[0] + ' x ' + network_label[1]
        else:
            network_label = ''

        string = f"{network_label} & {design} & {throughput:.2f} & {area:.2f} & {energy_efficiency:.2f} & {power_efficiency:.2f} \\\\"
        if arch_label in ['\\name', 'Carat'] and network == 'single_node':
            vlp_single_node_list.append(string)
        elif arch_label in ['SA', 'SD'] and network == 'single_node' and dim_label in ['8x8', '16x16']:
            baseline_single_node_list.append(string)
        elif arch_label in ['SA', 'SD'] and network == 'single_node' and dim_label in ['64x64']:
            baseline_sa_list.append(string)
        elif arch_label in ['\\name', 'Carat'] and network != 'single_node':
            vlp_multi_node_list.append(string)
        elif arch_label in ['SA'] and network != 'single_node':
            if subarch_label == '-F':
                sa_f_multi_node_list.append(string)
            else:
                sa_multi_node_list.append(string)
        elif arch_label in ['SD'] and network != 'single_node':
            if subarch_label == '-F':
                sd_f_multi_node_list.append(string)
            else:
                sd_multi_node_list.append(string)
        elif arch_label == 'Tensor' and network == 'single_node':
            tensor_single_node_list.append(string)
        elif arch_label == 'Tensor' and network != 'single_node':
            tensor_multi_node_list.append(string)
        
    with open(output_path + 'comprehensive_table.txt', 'w') as f:
        f.write(' & Arch & Throughput & Area & Energy Efficiency & Power Efficiency \\\\\n')
        f.write(' &      & Tokens/s & mm^2 & Tokens/s/uJ & Tokens/s/W \\\\\n')
    with open(output_path + 'comprehensive_table.txt', 'a') as f:
        for string in vlp_single_node_list:
            f.write(string + '\n')
        for string in baseline_single_node_list:
            f.write(string + '\n')
        for string in baseline_sa_list:
            f.write(string + '\n')
        # for string in tensor_single_node_list:
        #     f.write(string + '\n')
        for string in vlp_multi_node_list:
            f.write(string + '\n')
        for string in sa_multi_node_list:
            f.write(string + '\n')
        for string in sa_f_multi_node_list:
            f.write(string + '\n')
        for string in sd_multi_node_list:
            f.write(string + '\n')
        for string in sd_f_multi_node_list:
            f.write(string + '\n')
        for string in tensor_multi_node_list:
            f.write(string + '\n')