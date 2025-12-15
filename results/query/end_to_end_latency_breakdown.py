from zoo.llm.results.query.utils import query_execution_time, compute_latency, load_yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

def query(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']
    gemm_vlp_latency_module = 'and_gate'

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['16x16']
    simd_subarch_list = ['mac', 'figna']
    systolic_subarch_list = ['mac', 'figna', 'taylor', 'pwl']
    gemm_baseline_latency_module = 'multiplier'

    mugi_nonlinear_module = 'magnitude_register'
    baseline_nonlinear_module = 'accumulator_vector'

    kv_paths = 'kv_heads_8'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network = 'single_node'

    end_to_end_breakdown = pd.DataFrame()

    for arch in vlp_list + baseline_list:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
            for subarch in (systolic_subarch_list if arch in ['systolic'] else simd_subarch_list if arch in ['simd'] else mugi_subarch_list if arch in ['mugi'] else ['']):
                for model in model_list:

                    module = gemm_vlp_latency_module if arch in vlp_list else gemm_baseline_latency_module
                    nonlinear_module = mugi_nonlinear_module if arch in ['mugi'] else baseline_nonlinear_module
                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    kv_path = kv_paths if model in ['llama_2_70b_GQA'] else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{kv_path}/{termination_path}/')
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    proj_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='projection')
                    proj_latency_eff_dict = {
                        'latency': proj_execution_time
                    }

                    attn_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='attention')
                    attn_latency_eff_dict = {
                        'latency': attn_execution_time
                    }

                    ffn_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='ffn')
                    ffn_latency_eff_dict = {
                        'latency': ffn_execution_time
                    }

                    nonlinear_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='nonlinear')
                    nonlinear_latency_eff_dict = {
                        'latency': nonlinear_execution_time
                    }

                    proj_metric_df = pd.DataFrame(proj_latency_eff_dict, index=[0])
                    proj_metric_df['layer'] = 'projection'
                    attn_metric_df = pd.DataFrame(attn_latency_eff_dict, index=[0])
                    attn_metric_df['layer'] = 'attention'
                    ffn_metric_df = pd.DataFrame(ffn_latency_eff_dict, index=[0])
                    ffn_metric_df['layer'] = 'ffn'
                    nonlinear_metric_df = pd.DataFrame(nonlinear_latency_eff_dict, index=[0])
                    nonlinear_metric_df['layer'] = 'nonlinear'


                    end_to_end_metric_df = pd.concat([proj_metric_df, attn_metric_df, ffn_metric_df, nonlinear_metric_df])
                    end_to_end_metric_df['arch'] = arch
                    end_to_end_metric_df['subarch'] = subarch
                    end_to_end_metric_df['arch_dim'] = arch_dim
                    end_to_end_metric_df['model'] = model

                    end_to_end_metric_df = end_to_end_metric_df.drop(columns=['flops', 'execution_time', 'power', 'energy'], errors='ignore')
                    end_to_end_breakdown = pd.concat([end_to_end_breakdown, end_to_end_metric_df], axis=0)

    end_to_end_breakdown.to_csv(output_path + 'end_to_end_latency_breakdown.csv', index=False)

    group_cols = [col for col in end_to_end_breakdown.columns if col not in ['latency', 'layer']]
    total_latency_df = end_to_end_breakdown.groupby(group_cols)['latency'].sum().reset_index()

    # Normalize each model separately against its own systolic mac 16x16 baseline
    end_to_end_breakdown['normalized_latency'] = 0.0  # Initialize column
    
    for model in end_to_end_breakdown['model'].unique():
        # Get baseline for this specific model
        baseline_df = total_latency_df[
            (total_latency_df['arch'] == 'mugi') &
            (total_latency_df['subarch'] == 'vlp') &
            (total_latency_df['arch_dim'] == '256x8') &
            (total_latency_df['model'] == model)
        ]
        
        if not baseline_df.empty:
            baseline_latency = baseline_df['latency'].values[0]
            # Normalize all entries for this model
            model_mask = end_to_end_breakdown['model'] == model
            end_to_end_breakdown.loc[model_mask, 'normalized_latency'] = (
                end_to_end_breakdown.loc[model_mask, 'latency'] / baseline_latency
            )

    end_to_end_breakdown.to_csv(output_path + 'end_to_end_latency_breakdown_norm.csv', index=False)

def figure(input_path: str, output_path: str):

    df = pd.read_csv(input_path + 'end_to_end_latency_breakdown_norm.csv')

    fig_width_pt = 250
    fig_width = fig_width_pt/72
    fig_height = fig_width/3.25

    font_size = 8

    fig, axes = plt.subplots(
        1, 4, figsize=(fig_width, fig_height), sharex=False, sharey=True
    )
    
    # Remove spacing between subplots to make them appear as one continuous graph
    plt.subplots_adjust(wspace=0)

    label_dict = {}

    for label in df.columns:
        if isinstance(df[label].iloc[0], str):
            label_dict[label] = list(df[label].unique())

    

    # Create extended architecture list for display (systolic subarchs as separate items)
    display_archs = []
    for arch in label_dict['arch']:
        if arch == 'simd':
            continue  # Skip simd architecture
        elif arch == 'systolic':
            display_archs.extend(['systolic', 'systolic_taylor', 'systolic_pwl'])
        else:
            display_archs.append(arch)
    
    bars_per_arch = {
        'mugi': 1,
        'carat': 1,
        'systolic': 1,
        'systolic_taylor': 1,
        'systolic_pwl': 1
    }

    bar_width = 0.6

    group_centers = np.arange(len(display_archs))


    colors_map = {
        'Mugi': "green",
        'Carat': "orange",
        'SA': "purple",
        'SA-F': "darkgray",
        'SD': "red",
        'SD-F': "dodgerblue"
    }

    # Colors for different layers in the stack
    layer_colors = {
        'projection': '#FF6B6B',  # Red
        'attention': "#53CC6D",   # Teal
        'ffn': '#45B7D1',         # Blue
        'nonlinear': '#FFA07A'    # Light Salmon
    }

    

    for idx, model in enumerate(label_dict['model']):
        ax = axes[idx]
        model_label = '7B' if model == 'llama_2_7b' else '13B' if model == 'llama_2_13b' else '70B' if model in 'llama_2_70b' else '70B GQA'
        ax.set_title(model_label, fontsize=font_size+1, pad = 2.5)

        

        x_pos = 0
        xticks = []
        xtick_labels = []

        for i, display_arch in enumerate(display_archs):
            # Determine actual arch and subarch for data filtering
            if display_arch.startswith('systolic'):
                arch = 'systolic'
                if display_arch == 'systolic':
                    subarch = 'mac'
                elif display_arch == 'systolic_taylor':
                    subarch = 'taylor'
                elif display_arch == 'systolic_pwl':
                    subarch = 'pwl'
            else:
                arch = display_arch
                subarch = ''
                
            n_bars = bars_per_arch.get(display_arch, 1)

            start_pos = group_centers[i] - (bar_width * n_bars) / 2
            bar_positions = start_pos + np.arange(n_bars) * bar_width

            np.empty((n_bars, len(label_dict['layer'])))
            
            if arch == 'mugi':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['subarch'] == 'vlp') &
                    (df['arch_dim'].isin(['256x8']))
                ]

            if arch == 'carat':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['arch_dim'].isin(['256x8']))
                ]
            
            if arch == 'systolic':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['subarch'] == subarch) &
                    (df['arch_dim'] == '16x16')
                ]

            if arch == 'tensor':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['arch_dim'] == '8x16x16')
                ]

            # Get data for each layer separately
            layer_data = {}
            for layer in ['projection', 'attention', 'ffn', 'nonlinear']:
                layer_filtered_df = filtered_df[filtered_df['layer'] == layer]
                layer_data[layer] = layer_filtered_df['normalized_latency'].tolist()

            # Create stacked bars for each architecture
            for j, pos in enumerate(bar_positions):
                bottom = 0
                for layer in ['projection', 'attention', 'ffn', 'nonlinear']:
                    if j < len(layer_data[layer]) and layer_data[layer]:
                        height = layer_data[layer][j] if j < len(layer_data[layer]) else 0
                        ax.bar(
                            pos,
                            height,
                            width=bar_width,
                            bottom=bottom,
                            label=layer if j == 0 and i == 0 else "",  # Only label once for legend
                            color=layer_colors[layer],
                            edgecolor='black',
                            linewidth=0.4
                        )
                        bottom += height

        # Calculate actual tick positions for each display architecture
        tick_positions = []
        display_arch_labels = []
        
        for i, display_arch in enumerate(display_archs):
            n_bars = bars_per_arch.get(display_arch, 1)
            start_pos = group_centers[i] - (bar_width * n_bars) / 2
            bar_positions = start_pos + np.arange(n_bars) * bar_width
            
            # For groups with multiple bars, position tick between bars
            # For single bars, position tick at the center of the bar
            if n_bars > 1:
                tick_pos = np.mean(bar_positions)
            else:
                tick_pos = bar_positions[0]
            tick_positions.append(tick_pos)
            
            # Create display labels
            if display_arch == 'systolic':
                display_arch_labels.append('S')
            elif display_arch == 'systolic_taylor':
                display_arch_labels.append('T')
            elif display_arch == 'systolic_pwl':
                display_arch_labels.append('P')
            elif display_arch == 'mugi':
                display_arch_labels.append('M')
            elif display_arch == 'carat':
                display_arch_labels.append('C')
            else:
                display_arch_labels.append(display_arch)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(display_arch_labels, fontsize=font_size)
        ax.set_xlim(-1, len(display_archs))
        
        # Add top x-axis with dimension labels
        ax2 = ax.twiny()
        
        # Calculate positions for merged dimension labels
        dim_tick_positions = []
        dim_tick_labels = []
        
        # Group architectures by dimension
        # VLP group: mugi, carat (256x8)
        vlp_indices = [i for i, arch in enumerate(display_archs) if arch in ['mugi', 'carat']]
        # Systolic group: systolic, systolic_taylor, systolic_pwl (16x16)
        systolic_indices = [i for i, arch in enumerate(display_archs) if arch.startswith('systolic')]
        
        # Calculate position for VLP group (256)
        if vlp_indices:
            vlp_positions = []
            for i in vlp_indices:
                n_bars = bars_per_arch.get(display_archs[i], 1)
                start_pos = group_centers[i] - (bar_width * n_bars) / 2
                bar_positions = start_pos + np.arange(n_bars) * bar_width
                if n_bars > 1:
                    vlp_positions.append(np.mean(bar_positions))
                else:
                    vlp_positions.append(bar_positions[0])
            
            # Center the label over the VLP group
            vlp_center = np.mean(vlp_positions)
            dim_tick_positions.append(vlp_center)
            dim_tick_labels.append('256')
        
        # Calculate position for Systolic group (16)
        if systolic_indices:
            systolic_positions = []
            for i in systolic_indices:
                n_bars = bars_per_arch.get(display_archs[i], 1)
                start_pos = group_centers[i] - (bar_width * n_bars) / 2
                bar_positions = start_pos + np.arange(n_bars) * bar_width
                if n_bars > 1:
                    systolic_positions.append(np.mean(bar_positions))
                else:
                    systolic_positions.append(bar_positions[0])
            
            # Center the label over the Systolic group
            systolic_center = np.mean(systolic_positions)
            dim_tick_positions.append(systolic_center)
            dim_tick_labels.append('16')
        
        ax2.set_xticks(dim_tick_positions)
        ax2.set_xticklabels(dim_tick_labels, fontsize=font_size)
        ax2.set_xlim(ax.get_xlim())
        
        # Set tick marks to normal thickness
        ax.tick_params(axis='both', width=1.0, length=4)
        ax2.tick_params(axis='x', width=1.0, length=4)
        
        # Make both xticks closer to the graph
        ax.tick_params(axis='x', pad=0.5)
        ax2.tick_params(axis='x', pad=0.25)
        
        # Set y-axis tick font size to match x-axis
        ax.tick_params(axis='y', labelsize=font_size)
        
        # Set y-axis to show integers 0, 1, 2 only
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['0', '1', '2'])
        ax.set_ylim(0, 2.2)  # Set y-limit to accommodate the 2.0 max
        
        # Remove all spines except outer ones to create unified appearance
        if idx > 0:
            # Remove y-axis labels and ticks for interior subplots
            ax.tick_params(axis='y', left=False, labelleft=False)
            # Add light dashed grid lines (same as leftmost subplot)
            ax.grid(True, axis='y', linestyle='--', linewidth=1.0, alpha=0.5, color='gray')
            # Remove left spine for interior subplots
            ax.spines['left'].set_visible(False)
        else:
            # Ensure y-axis is visible on the leftmost subplot with values
            ax.tick_params(axis='y', left=True, labelleft=True, labelsize=font_size)
            # Explicitly enable y-axis labels
            ax.yaxis.set_tick_params(labelleft=True)
            # Add light dashed grid lines
            ax.grid(True, axis='y', linestyle='--', linewidth=1.0, alpha=0.5, color='gray')
            # Set leftmost spine to normal thickness
            ax.spines['left'].set_linewidth(1.0)
        
        # Remove right spine for all but the rightmost subplot
        if idx < len(label_dict['model']) - 1:
            ax.spines['right'].set_visible(False)
        else:
            # Set rightmost spine to normal thickness
            ax.spines['right'].set_linewidth(1.0)
        
        # Set top and bottom spines to normal thickness
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
        # Remove top spine from twin axis except for outer ones
        if idx == 0:
            ax2.spines['top'].set_linewidth(1.0)
        else:
            ax2.spines['top'].set_visible(False)
            
        # Remove other spines from twin axis
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    # Remove xlabel from all subplots (was previously added)
    # for ax in axes:
    #     ax.set_xlabel("Architecture", fontsize=font_size)
    
    # Add legend for layers at the top in one row
    handles = [plt.Rectangle((0,0),1,1, color=layer_colors[layer]) for layer in ['projection', 'attention', 'ffn', 'nonlinear']]
    labels = ['Projection', 'Attention', 'FFN', 'Nonlinear']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.525, 1.37), ncol=4, fontsize=font_size,
               columnspacing=1, handlelength=.75, handleheight=0.5, handletextpad=0.4)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.25)

    plt.savefig(output_path + 'end_to_end_latency_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'end_to_end_latency_breakdown.pdf', dpi=1200, bbox_inches='tight')