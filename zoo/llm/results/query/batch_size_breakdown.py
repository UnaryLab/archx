from zoo.llm.results.query.utils import query_throughput_energy_metrics, load_yaml, compute_throughput, geomean
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
import os

def query(input_path, output_path):

    
    network_list = ['single_node']

    mugi_subarch_list = ['vlp']

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['64x8', '256x8']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16']
    baseline_subarch_list = ['mac', 'figna']

    throughput_module = OrderedDict({
        'mugi': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'and_gate'}),
        'carat': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'register_vector'}),
        'systolic': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
        'simd': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
        'tensor': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
    })

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b']
    max_seq_len_list = ['max_seq_len_128', 'max_seq_len_256', 'max_seq_len_512', 'max_seq_len_1024', 'max_seq_len_2048', 'max_seq_len_4096']
    batch_size_list = ['batch_size_1', 'batch_size_2', 'batch_size_4', 'batch_size_8', 'batch_size_16', 'batch_size_32']

    workload = 'llama_2'

    batch_size_breakdown_df = pd.DataFrame()

    for network in network_list:
        for arch in (vlp_list + baseline_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else mugi_subarch_list if arch in ['mugi'] else ['']):
                subarch_label = 'none' if subarch == '' else subarch
                for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
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

                                gemm_performance_metrics_dict = query_throughput_energy_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event='gemm', module=gemm_module)
                                nonlinear_performance_metrics_dict = query_throughput_energy_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event='nonlinear', module=nonlinear_module)

                                performance_metrics_dict = OrderedDict({
                                    'flops': gemm_performance_metrics_dict['flops'] + nonlinear_performance_metrics_dict['flops'],
                                    'execution_time': gemm_performance_metrics_dict['execution_time'] + nonlinear_performance_metrics_dict['execution_time']
                                })

                                throughput_eff_dict = compute_throughput(performance_metrics_dict=performance_metrics_dict)
                                throughput_eff_dict['energy_per_token'] = (gemm_performance_metrics_dict['energy'] + nonlinear_performance_metrics_dict['energy']) / int(max_seq_len.split('_')[-1])
                                geomean_list.append(throughput_eff_dict)

                            throughput_eff_geomean_dict = geomean(geomean_list)
                            batch_size_breakdown_dict = OrderedDict({
                                'arch': arch,
                                'subarch': subarch_label,
                                'arch_dim': arch_dim,
                                'max_seq_len': max_seq_len,
                                'batch_size': batch_size,
                                'throughput': throughput_eff_geomean_dict['throughput'],
                                'energy_per_token': throughput_eff_geomean_dict['energy_per_token'],
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
    merged_df['energy_per_token'] = merged_df['energy_per_token'] / merged_df['energy_per_token_baseline']
    normalized_df = merged_df.drop(
        columns=['throughput_baseline', 'energy_per_token_baseline']
    )

    normalized_df.to_csv(output_path + 'batch_size_norm.csv', index=False)

def figure(input_path: str, output_path: str):

    df = pd.read_csv(input_path + 'batch_size_norm.csv')
    
    mugi_df = df[df['arch'] == 'mugi']
    other_df = df[df['arch'] != 'mugi']

    df = pd.concat([other_df, mugi_df])

    fig_width_pt = 500           
    fig_width = fig_width_pt / 72.27   
    fig_height = fig_width / 5  # Adjusted height for separate subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=False, 
                                   gridspec_kw={'hspace': 0.325})
    
    # Add titles above each subplot
    ax1.set_title('Norm Throughput', fontsize=7, pad=4)
    ax2.set_title('Norm Energy/Token', fontsize=7, pad=4)

    color_map = {
        'Mugi': "#7BCCEC",
        'Carat': "#E4DD86",
        'SA': "#63C78A",
        'SA-F': "#C27CF1",
        'SD': "#F3AB58",
        'SD-F': "#FF9B9B",
    }
    
    marker_map = {
        '64x8': 'v',
        '256x8': 's',
        '8x8': 'v',
        '16x16': 's'
    }
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_lens = sorted([int(seq.split('_')[-1]) for seq in df['max_seq_len'].unique()])
    segment_spacing = len(batch_sizes) + 0.25
    
    x_positions = []
    all_labels = []
    
    # Plot both throughput and energy_per_token
    seq_len_positions = []  # Store positions for sequence length labels
    seq_len_labels = []     # Store sequence length labels
    
    for ax, metric, title, y_lim, y_ticks in [(ax1, 'throughput', 'Normalized Throughput', (0, 35), [10, 20, 30]), 
                                               (ax2, 'energy_per_token', 'Normalized Energy per Token', (0, 27.5), [5, 15, 25])]:
        
        for j, seq_len in enumerate(df['max_seq_len'].unique()):
            seq_len_value = int(seq_len.split('_')[-1])
            seq_len_df = df[df['max_seq_len'] == seq_len]
            start_pos = j * segment_spacing
            x_positions_segment = np.arange(len(batch_sizes)) + start_pos
            if ax == ax1:  # Only populate x_positions and seq_len info once
                x_positions.extend(x_positions_segment)
                mid_position = np.mean(x_positions_segment)
                seq_len_positions.append(mid_position)
                seq_len_labels.append(f'{seq_len_value}')
            
            # Plot squares first, then triangles to ensure triangles appear on top
            for marker_type in ['s', 'v']:  # squares first, then triangles
                for arch in seq_len_df['arch'].unique():
                    arch_df = seq_len_df[seq_len_df['arch'] == arch]
                    
                    for subarch in arch_df['subarch'].unique():
                        subarch_df = arch_df[arch_df['subarch'] == subarch]
                        
                        # Filter arch_dims by current marker type
                        arch_dims = [dim for dim in subarch_df['arch_dim'].unique() 
                                   if marker_map.get(dim, 'o') == marker_type]
                        
                        for arch_dim in arch_dims:
                            arch_dim_df = subarch_df[subarch_df['arch_dim'] == arch_dim]
                            #arch_label = 'Mugi/Carat' if arch == 'mugi' else 'SA/SA-F' if arch == 'systolic' else 'SD/SD-F'
                            arch_label = 'Mugi' if arch == 'mugi' else 'Carat' if arch == 'carat' else 'SA' if arch == 'systolic' else 'SD'
                            
                            # Add subarch to label
                            if subarch == 'figna':
                                arch_label += '-F'
                            
                            #if arch_label == 'SA' and subarch
                            label = f"{arch_label} ({arch_dim.split('x')[0]})" if j == 0 and ax == ax1 else None
                            if j == 0 and ax == ax1:
                                all_labels.append(label)
                            color = color_map.get(arch_label, 'black')
                            marker = marker_map.get(arch_dim, 'o')
                            y_data = []

                            for bs in batch_sizes:
                                batch_size = f'batch_size_{bs}'
                                value = arch_dim_df[arch_dim_df['batch_size'] == batch_size][metric].values
                                y_data.append(value[0] if len(value) > 0 else np.nan)
                            if(ax == ax2):
                                ax.plot(
                                    x_positions_segment,
                                    y_data,
                                    label=label,
                                    color=color,
                                    marker=marker,
                                    linestyle='-',
                                    markersize=2,
                                    linewidth=0.25,
                                    alpha=0.85,
                                    markeredgecolor='black',
                                    markeredgewidth=0.15,
                                    markerfacecolor=color
                                )
                            else:
                                ax.plot(
                                    x_positions_segment,
                                    y_data,
                                    label=label,
                                    color=color,
                                    marker=marker,
                                    linestyle='-',
                                    markersize=2.25,
                                    linewidth=0.5,
                                    alpha=0.85,
                                    markeredgecolor='black',
                                    markeredgewidth=0.15,
                                    markerfacecolor=color
                                )
        
        # Format axis
        if metric == 'throughput':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))
        
        ax.grid(axis='y', linestyle='--', color='grey', alpha=0.5, linewidth=0.5)
        
        # Ensure grid doesn't extend beyond the plot area
        ax.set_axisbelow(True)
  
        ax.set_ylim(y_lim)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_yticks(y_ticks)
        
        # Add vertical lines at sequence length positions
        for pos in seq_len_positions:
            ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Set x-axis limits to reduce white space at beginning and end
    x_min = min(x_positions) - 1
    x_max = max(x_positions) + 1
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Create secondary x-axis on top subplot for sequence length labels
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(seq_len_positions)
    ax1_top.set_xticklabels(seq_len_labels, fontsize=6)
    ax1_top.tick_params(axis='x', length=3, pad=1)
    
    # Set x-axis labels only on bottom plot to avoid redundancy
    ax1.set_xticks(x_positions)
    ax2.set_xticks(x_positions)
    xtick_labels = [f'{bs}' for bs in batch_sizes] * len(seq_lens)
    ax1.set_xticklabels([])  # Hide labels on top subplot
    ax1.tick_params(axis='x', bottom=False)  # Hide bottom tick marks on top subplot
    ax2.set_xticklabels(xtick_labels, fontsize=6)
    ax2.tick_params(axis='x', pad=1)
    

    
    # Get handles and labels from the plot, then sort them together
    handles, labels = ax1.get_legend_handles_labels()
    
    # Create pairs of (handle, label) and sort them
    handle_label_pairs = [(handle, label) for handle, label in zip(handles, labels) if label is not None]
    
    # Sort pairs by architecture and then by array size (smaller first)
    def sort_pairs(pair):
        label = pair[1]
        
        # Extract array size from label (e.g., "Mugi (128)" -> 128)
        import re
        size_match = re.search(r'\((\d+)\)', label) if label else None
        array_size = int(size_match.group(1)) if size_match else 999
        
        if label and 'Mugi' in label:
            return (0, array_size, label)  # Mugi first, then by size
        elif label and 'Carat' in label:
            return (1, array_size, label)  # Carat second, then by size
        elif label and 'SA-F' in label:
            return (3, array_size, label)  # SA-F fourth, then by size
        elif label and 'SA' in label:
            return (2, array_size, label)  # SA third, then by size
        elif label and 'SD-F' in label:
            return (5, array_size, label)  # SD-F sixth, then by size
        elif label and 'SD' in label:
            return (4, array_size, label)  # SD fifth, then by size
        else:
            return (6, array_size, label)  # Others last
    
    sorted_pairs = sorted(handle_label_pairs, key=sort_pairs)
    sorted_handles = [pair[0] for pair in sorted_pairs]
    sorted_labels = [pair[1] for pair in sorted_pairs]
    
    fig.legend(
        sorted_handles,
        sorted_labels,
        frameon=True,
        loc='upper center',
        fontsize=6,
        ncol=6,
        bbox_to_anchor=(0.5115, 1.32),
        columnspacing=1.15
    )
    
    plt.tight_layout(pad=0.2)
    plt.savefig(output_path + 'batch_size_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'batch_size_breakdown.pdf', dpi=1200, bbox_inches='tight')