from zoo.llm.results.query.utils import query_throughput_metrics, load_yaml, compute_throughput, geomean
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
import os

def query(input_path, output_path):
    
    network_list = ['single_node']

    vlp_list = ['mugi']
    vlp_arch_dim_list = ['128x8', '256x8']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16']
    baseline_subarch_list = ['mac']

    throughput_module = OrderedDict({
        'mugi': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'and_gate'}),
        'carat': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'register_vector'}),
        'systolic': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
        'simd': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'})
    })

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b']
    max_seq_len_list = ['max_seq_len_512', 'max_seq_len_2048', 'max_seq_len_4096']
    batch_size_list = ['batch_size_1', 'batch_size_2', 'batch_size_4', 'batch_size_8', 'batch_size_16', 'batch_size_32']

    workload = 'llama_2'

    batch_size_breakdown_df = pd.DataFrame()

    for network in network_list:
        for arch in (vlp_list + baseline_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else ['']):
                subarch_label = 'none' if subarch == '' else subarch
                for arch_dim in (vlp_arch_dim_list if arch == 'mugi' else baseline_arch_dim_list):
                    for max_seq_len in max_seq_len_list:
                        for batch_size in batch_size_list:
                            geomean_list = []
                            for model in model_list:

                                gemm_module = throughput_module[arch]['gemm']
                                nonlinear_module = throughput_module[arch]['nonlinear']

                                termination_path = 'full_termination' if arch == 'mugi' else ''
                                run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                                yaml_dict = load_yaml(run_path)

                                event_graph = yaml_dict['event_graph']
                                metric_dict = yaml_dict['metric_dict']

                                gemm_performance_metrics_dict = query_throughput_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event='gemm', module=gemm_module)
                                nonlinear_performance_metrics_dict = query_throughput_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event='nonlinear', module=nonlinear_module)

                                performance_metrics_dict = OrderedDict({
                                    'flops': gemm_performance_metrics_dict['flops'] + nonlinear_performance_metrics_dict['flops'],
                                    'execution_time': gemm_performance_metrics_dict['execution_time'] + nonlinear_performance_metrics_dict['execution_time']
                                })

                                throughput_eff_dict = compute_throughput(performance_metrics_dict=performance_metrics_dict)
                                geomean_list.append(throughput_eff_dict)

                            throughput_eff_geomean_dict = geomean(geomean_list)
                            batch_size_breakdown_dict = OrderedDict({
                                'arch': arch,
                                'subarch': subarch_label,
                                'arch_dim': arch_dim,
                                'max_seq_len': max_seq_len,
                                'batch_size': batch_size,
                                'throughput': throughput_eff_geomean_dict['throughput'],
                                # 'flops': throughput_eff_geomean_dict['flops'],
                                # 'execution_time': throughput_eff_geomean_dict['execution_time'],
                            })

                            batch_size_breakdown_df = pd.concat([batch_size_breakdown_df, pd.DataFrame(batch_size_breakdown_dict, index=[0])])

    batch_size_breakdown_df.to_csv(output_path + 'batch_size_breakdown.csv', index=False)

    baseline_df = batch_size_breakdown_df[
        (batch_size_breakdown_df['arch'] == 'systolic') &
        (batch_size_breakdown_df['subarch'] == 'mac') &
        (batch_size_breakdown_df['arch_dim'] == '8x8') &
        (batch_size_breakdown_df['batch_size'] == 'batch_size_1')
    ]

    merge_list = ['max_seq_len']
    numeric_columns = baseline_df.select_dtypes(include=['number']).columns
    columns_to_merge = merge_list + list(numeric_columns)

    merged_df = batch_size_breakdown_df.merge(
        baseline_df[columns_to_merge],
        on=merge_list,
        suffixes=('', '_baseline')
    )

    merged_df['throughput'] = merged_df['throughput'] / merged_df['throughput_baseline']
    normalized_df = merged_df.drop(
        columns=['throughput_baseline']
    )

    normalized_df.to_csv(output_path + 'batch_size_norm.csv', index=False)

def figure(input_path: str, output_path: str):
    df = pd.read_csv(input_path + 'batch_size_norm.csv')
    
    fig_width_pt = 516              
    fig_width = fig_width_pt / 72   
    fig_height = fig_width / 5.75
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    color_map = {
        'mugi': 'dodgerblue',
        'systolic': 'grey',
        'simd': 'orangered'
    }
    
    marker_map = {
        '128x8': 'v',
        '256x8': 's',
        '8x8': 'v',
        '16x16': 's'
    }
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_lens = sorted([int(seq.split('_')[-1]) for seq in df['max_seq_len'].unique()])
    segment_spacing = len(batch_sizes) + 2
    
    x_positions = []
    all_labels = []
    
    for j, seq_len in enumerate(df['max_seq_len'].unique()):
        seq_len_value = int(seq_len.split('_')[-1])
        seq_len_df = df[df['max_seq_len'] == seq_len]
        start_pos = j * segment_spacing
        x_positions_segment = np.arange(len(batch_sizes)) + start_pos
        x_positions.extend(x_positions_segment)
        
        for arch in seq_len_df['arch'].unique():
            arch_df = seq_len_df[seq_len_df['arch'] == arch]
            
            for arch_dim in arch_df['arch_dim'].unique():
                arch_dim_df = arch_df[arch_df['arch_dim'] == arch_dim]
                arch_label = 'Mugi/Carat' if arch == 'mugi' else 'SA/SA-F' if arch == 'systolic' else 'SD/SD-F'
                label = f'{arch_label} ({arch_dim.split('x')[0]})' if j == 0 else None
                if j == 0:
                    all_labels.append(label)
                color = color_map.get(arch, 'black')
                marker = marker_map.get(arch_dim, 'o')
                y_data = []

                for bs in batch_sizes:
                    batch_size = f'batch_size_{bs}'
                    value = arch_dim_df[arch_dim_df['batch_size'] == batch_size]['throughput'].values
                    y_data.append(value[0] if len(value) > 0 else np.nan)
                
                ax.plot(
                    x_positions_segment,
                    y_data,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle='-',
                    markersize=2.3,
                    linewidth=0.75,
                    alpha=0.8
                )
        
        mid_position = np.mean(x_positions_segment)
        ax.text(mid_position, ax.get_ylim()[1] + 1.75,
                f'Seq Len {seq_len_value}',
                ha='center', fontsize=5.5, fontweight='bold')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))
    ax.grid(axis='y', linestyle='--', color='grey', alpha=0.5, linewidth=0.5)  # Add gridline
    ax.set_xticks(x_positions)
    xtick_labels = [f'{bs}' for bs in batch_sizes] * len(seq_lens)
    ax.set_xticklabels(xtick_labels, fontsize=6)
    ax.set_title('Normalized Throughput', fontsize=6)
    
    ax.set_ylim(0, 40)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_yticks([10, 20, 30, 40])
    
    fig.legend(
        all_labels,
        frameon=True,
        loc='upper center',
        fontsize=6,
        ncol=6,
        bbox_to_anchor=(0.5, 1.2),
    )
    
    plt.tight_layout(pad=0.2)
    plt.savefig(output_path + 'batch_size_breakdown.pdf', dpi=1200, bbox_inches='tight')
    plt.show()