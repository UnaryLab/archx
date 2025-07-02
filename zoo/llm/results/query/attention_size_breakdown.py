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

    batch_size = 'batch_size_8'

    model_list = ['llama_2_70b_GQA']
    max_seq_len_list = ['max_seq_len_128', 'max_seq_len_256', 'max_seq_len_512', 'max_seq_len_1024', 'max_seq_len_2048', 'max_seq_len_4096']
    batch_size_list = ['batch_size_1', 'batch_size_2', 'batch_size_4', 'batch_size_8', 'batch_size_16', 'batch_size_32']
    gqa_size_list = ['kv_heads_64', 'kv_heads_32', 'kv_heads_16', 'kv_heads_8']

    workload = 'llama_2'

    attention_size_breakdown_df = pd.DataFrame()

    for network in network_list:
        for arch in (vlp_list + baseline_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else mugi_subarch_list if arch in ['mugi'] else ['']):
                subarch_label = 'none' if subarch == '' else subarch
                for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
                    for max_seq_len in max_seq_len_list:
                        for gqa_size in gqa_size_list:
                            for model in model_list:

                                gemm_module = throughput_module[arch]['gemm']

                                termination_path = 'full_termination' if arch == 'mugi' else ''
                                run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{gqa_size}/{termination_path}/')
                                yaml_dict = load_yaml(run_path)

                                event_graph = yaml_dict['event_graph']
                                metric_dict = yaml_dict['metric_dict']

                                gemm_performance_metrics_dict = query_throughput_energy_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event='attention', module=gemm_module)

                                performance_metrics_dict = OrderedDict({
                                    'flops': gemm_performance_metrics_dict['flops'],
                                    'execution_time': gemm_performance_metrics_dict['execution_time']
                                })

                                throughput_eff_dict = compute_throughput(performance_metrics_dict=performance_metrics_dict)
                                throughput_eff_dict['energy_per_token'] = gemm_performance_metrics_dict['energy'] / int(max_seq_len.split('_')[-1])

                            attention_size = 'attn_size_' + str(64 // int(gqa_size.split('_')[-1]))

                            attention_size_breakdown_dict = OrderedDict({
                                'arch': arch,
                                'subarch': subarch_label,
                                'arch_dim': arch_dim,
                                'max_seq_len': max_seq_len,
                                'attn_size': attention_size,
                                'throughput': throughput_eff_dict['throughput'],
                                'energy_per_token': throughput_eff_dict['energy_per_token'],
                                # 'flops': throughput_eff_geomean_dict['flops'],
                                # 'execution_time': throughput_eff_geomean_dict['execution_time'],
                            })

                            attention_size_breakdown_df = pd.concat([attention_size_breakdown_df, pd.DataFrame(attention_size_breakdown_dict, index=[0])])

    attention_size_breakdown_df.to_csv(output_path + 'attention_size_breakdown.csv', index=False)

    baseline_df = attention_size_breakdown_df[
        (attention_size_breakdown_df['arch'] == 'systolic') &
        (attention_size_breakdown_df['subarch'] == 'mac') &
        (attention_size_breakdown_df['arch_dim'] == '8x8') &
        (attention_size_breakdown_df['attn_size'] == 'attn_size_1')
    ]

    merge_list = ['max_seq_len']
    numeric_columns = baseline_df.select_dtypes(include=['number']).columns
    columns_to_merge = merge_list + list(numeric_columns)

    merged_df = attention_size_breakdown_df.merge(
        baseline_df[columns_to_merge],
        on=merge_list,
        suffixes=('', '_baseline')
    )

    merged_df['throughput'] = merged_df['throughput'] / merged_df['throughput_baseline']
    merged_df['energy_per_token'] = merged_df['energy_per_token'] / merged_df['energy_per_token_baseline']
    normalized_df = merged_df.drop(
        columns=['throughput_baseline', 'energy_per_token_baseline']
    )

    normalized_df.to_csv(output_path + 'attention_size_norm.csv', index=False)

def figure(input_path: str, output_path: str):
    df = pd.read_csv(input_path + 'attention_size_norm.csv')
    
    mugi_df = df[df['arch'] == 'mugi']
    other_df = df[df['arch'] != 'mugi']

    df = pd.concat([other_df, mugi_df])

    fig_width_pt = 500              
    fig_width = fig_width_pt / 72   
    fig_height = fig_width / 4.75  # Doubled height for two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, 
                                   gridspec_kw={'hspace': 0.6, 'height_ratios': [0.7, 1]})

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
    
    attn_sizes = [1, 2, 4, 8]
    seq_lens = sorted([int(seq.split('_')[-1]) for seq in df['max_seq_len'].unique()])
    segment_spacing = len(attn_sizes) + 2
    
    x_positions = []
    all_labels = []
    
    # Plot both throughput and energy_per_token
    seq_len_positions = []  # Store positions for sequence length labels
    seq_len_labels = []     # Store sequence length labels
    
    for ax, metric, title, y_lim, y_ticks in [(ax1, 'throughput', 'Normalized Throughput', (0, 25), [5, 15, 25]), 
                                               (ax2, 'energy_per_token', 'Normalized Energy per Token', (0.25, 1.3), [0.25, 0.75, 1.25])]:
        
        for j, seq_len in enumerate(df['max_seq_len'].unique()):
            seq_len_value = int(seq_len.split('_')[-1])
            seq_len_df = df[df['max_seq_len'] == seq_len]
            start_pos = j * segment_spacing
            x_positions_segment = np.arange(len(attn_sizes)) + start_pos
            if ax == ax1:  # Only populate x_positions and seq_len info once
                x_positions.extend(x_positions_segment)
                mid_position = np.mean(x_positions_segment)
                seq_len_positions.append(mid_position)
                seq_len_labels.append(f'{seq_len_value}')
            
            for arch in seq_len_df['arch'].unique():
                arch_df = seq_len_df[seq_len_df['arch'] == arch]
                
                for subarch in arch_df['subarch'].unique():
                    subarch_df = arch_df[arch_df['subarch'] == subarch]
                    
                    # Sort arch_dim so squares ('256x8', '16x16') are plotted first, then triangles ('128x8', '8x8')
                    arch_dims = sorted(subarch_df['arch_dim'].unique(), 
                                     key=lambda x: 0 if marker_map.get(x, 'o') == 's' else 1)
                    
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

                        for attns in attn_sizes:
                            attn_size = f'attn_size_{attns}'
                            value = arch_dim_df[arch_dim_df['attn_size'] == attn_size][metric].values
                            y_data.append(value[0] if len(value) > 0 else np.nan)
                        
                        ax.plot(
                            x_positions_segment,
                            y_data,
                            label=label,
                            color=color,
                            marker=marker,
                            linestyle='-',
                            markersize=2.5,
                            linewidth=0.6,
                            alpha=0.9,
                            markeredgecolor='black',
                            markeredgewidth=0.1,
                            markerfacecolor=color
                        )
        
        # Format axis
        if metric == 'throughput':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}x'))
        
        ax.grid(axis='y', linestyle='--', color='grey', alpha=0.5, linewidth=0.5)
        if(metric == 'throughput'):
            ax.set_title(title, fontsize=7, pad = 13)
        else:
            ax.set_title(title, fontsize=7)
        ax.set_ylim(y_lim)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_yticks(y_ticks)
        
        # Add vertical lines at sequence length positions
        for pos in seq_len_positions:
            ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Create secondary x-axis on top subplot for sequence length labels
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(seq_len_positions)
    ax1_top.set_xticklabels(seq_len_labels, fontsize=5.5)
    ax1_top.tick_params(axis='x', length=3, pad=1)  # Show tick marks on top
    
    # Create secondary x-axis on bottom subplot for sequence length tick marks only (no labels)
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(seq_len_positions)
    ax2_top.set_xticklabels([])  # No labels, just tick marks
    ax2_top.tick_params(axis='x', length=3, top=True, labeltop=False)  # Show tick marks but no labels
    
    # Only set x-axis labels on bottom plot
    ax2.set_xticks(x_positions)
    xtick_labels = [f'{attns}' for attns in attn_sizes] * len(seq_lens)
    ax2.set_xticklabels(xtick_labels, fontsize=5.5)
    
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
    
    # fig.legend(
    #     sorted_handles,
    #     sorted_labels,
    #     frameon=True,
    #     loc='upper center',
    #     fontsize=7,
    #     ncol=6,
    #     bbox_to_anchor=(0.5115, 1.35),
    #     columnspacing=1.15
    # )
    
    plt.tight_layout(pad=0.2)
    plt.savefig(output_path + 'attention_size_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'attention_size_breakdown.pdf', dpi=1200, bbox_inches='tight')
    plt.show()