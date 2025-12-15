from zoo.llm.results.query.utils import query_performance_metrics, compute_throughput_efficiancy, query_execution_time,  load_yaml, geomean
from archx.metric import query_module_metric, aggregate_event_metric
import pandas as pd
from collections import OrderedDict
import os
import sys
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np

def query(input_path, output_path):

    network_list = ['single_node',  'multi_node_4x4', 'multi_node_8x8']

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['64x8', '128x8', '256x8']
    vlp_node_stationary = ''

    mugi_subarch_list = ['vlp']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16', '64x64']
    baseline_subarch_list = ['mac', 'figna']
    baseline_node_stationary = 'node_stationary_ws'

    throughput_module = OrderedDict({
        'mugi': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'and_gate'}),
        'carat': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'register_vector'}),
        'systolic': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
        'simd': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
        'tensor': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'})
    })

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'

    noc_breakdown_df = pd.DataFrame()

    for network in network_list:
        for arch in (vlp_list + baseline_list + ['tensor']):
            for subarch in (baseline_subarch_list if arch in baseline_list else mugi_subarch_list if arch in ['mugi'] else ['']):
                for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):

                    if network == 'multi_node_4x4' and arch == 'tensor':
                        arch_dim = '8x16x16'  # Special case for tensor architecture
                        network_label = 'multi_node_2x1'
                    elif network == 'multi_node_8x8' and arch == 'tensor':
                        arch_dim = '8x16x16'  # Special case for tensor architecture
                        network_label = 'multi_node_2x2'
                    else:
                        network_label = network

                    if network == 'single_node' and (arch_dim != '64x64' and arch != 'tensor'):
                        continue
                    if network != 'single_node' and arch_dim == '64x64':
                        continue
                    noc_breakdown_model_list = []
                    for model in model_list:
                        gemm_module = throughput_module[arch]['gemm']
                        nonlinear_module = throughput_module[arch]['nonlinear']
                        
                        if model == 'llama_2_70b_GQA':
                            kv_path = 'kv_heads_8'
                        else:
                            kv_path = ''

                        


                        termination_path = 'full_termination' if arch == 'mugi' else ''
                        run_path = os.path.normpath(f'{input_path}{arch}/{network_label}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{kv_path}/{termination_path}/')
                        yaml_dict = load_yaml(run_path)

                        event_graph = yaml_dict['event_graph']
                        metric_dict = yaml_dict['metric_dict']

                        gemm_performance_metrics_dict = query_performance_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='gemm', module=gemm_module)
                        nonlinear_performance_metrics_dict = query_performance_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='nonlinear', module=nonlinear_module)

                        assert gemm_performance_metrics_dict['power'] == nonlinear_performance_metrics_dict['power'], "Power mismatch between gemm and nonlinear modules"

                        performance_metrics_dict = OrderedDict({
                            'flops': gemm_performance_metrics_dict['flops'] + nonlinear_performance_metrics_dict['flops'],
                            'execution_time': gemm_performance_metrics_dict['execution_time'] + nonlinear_performance_metrics_dict['execution_time'],
                            'energy': gemm_performance_metrics_dict['energy'] + nonlinear_performance_metrics_dict['energy'],
                            'power': gemm_performance_metrics_dict['power']
                        })

                        throughput_eff_dict = compute_throughput_efficiancy(performance_metrics_dict=performance_metrics_dict)

                        noc_breakdown_dict = OrderedDict({
                            'arch': arch,
                            'subarch': subarch,
                            'network': network_label,
                            'arch_dim': arch_dim,
                            'throughput': throughput_eff_dict['throughput'],
                            'energy_efficiency': throughput_eff_dict['energy_efficiency'],
                            'power_efficiency': throughput_eff_dict['power_efficiency'],
                            'flops': throughput_eff_dict['flops'],
                            'execution_time': throughput_eff_dict['execution_time']
                        })

                        noc_breakdown_model_list.append(noc_breakdown_dict)

                    noc_breakdown_dict = geomean(noc_breakdown_model_list)
                    noc_breakdown_df = pd.concat([noc_breakdown_df, pd.DataFrame(noc_breakdown_dict, index=[0])])
                        
    noc_breakdown_df.to_csv(output_path + 'noc_breakdown.csv', index=False)

    baseline_df = noc_breakdown_df[
        (noc_breakdown_df['arch'] == 'systolic') &
        (noc_breakdown_df['subarch'] == 'mac') &
        (noc_breakdown_df['arch_dim'] == '8x8') &
        (noc_breakdown_df['network'] == 'multi_node_4x4')
    ]

    baseline_row = baseline_df.iloc[0]

    numeric_columns = baseline_df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        noc_breakdown_df[col] = noc_breakdown_df[col] / baseline_row[col]

    noc_breakdown_df.to_csv(output_path + 'noc_breakdown_norm.csv', index=False)

def lighten_color(color, amount=0.0):
    """
    Lighten the given color by blending it with white.
    amount=0 returns the original color; amount=1 returns white.
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = np.array(mc.to_rgb(c))
    # Linearly interpolate between the color and white.
    new_color = (1 - amount) * c + amount * np.array([1, 1, 1])
    return new_color

def get_shaded_color_by_index(base_color, index, total):
    """
    For a given base_color, return a slightly lighter shade based on the
    index (0-based) of the design within its group. The maximum lightening is 40%.
    """
    if total > 1:
        # Compute a lightening factor that goes from 0 (first design) to 0.4 (last design)
        amount = (index / (total - 1)) * 0.4
    else:
        amount = 0
    return lighten_color(base_color, amount)

def figure(input_path: str, output_path: str):
    df = pd.read_csv(input_path + "noc_breakdown_norm.csv")
    
    # Figure settings
    fig_width_pt = 250          # ACM single-column width in points
    fig_width = fig_width_pt / 72  # Convert to inches
    fig_height = fig_width/1.75  # Adjusted height for readability
    
    font_title = 8
    font_tick = 7
    
    # Create figure with 3 rows, 1 column
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True, gridspec_kw={'top': 0.9, 'bottom': 0.2})
    
    # Color scheme for different architectures
    base_colors = {
        'mugi': "#04740B",
        'carat': "#680C80",
        'systolic': "#0E5EA0",
        'systolic_figna': "#FF6B6B",  # Red for SA-F
        'simd': "#A05E09",
        'simd_figna': "#4ECDC4",  # Teal for SD-F
        'tensor': "#AC0D0D"
    }
    
    # Prepare data for sorting and grouping
    data_rows = []
    for _, row in df.iterrows():
        arch = row['arch']
        subarch = row['subarch'] if pd.notna(row['subarch']) and row['subarch'] != '' else ''
        network = row['network']
        arch_dim = row['arch_dim']
        
        # Extract first number from arch_dim for array size
        array_size = arch_dim.split('x')[0]
        
        # Create label with simplified format
        if subarch == 'figna':
            label = f"{arch.capitalize()}-F ({array_size})"
        else:
            label = f"{arch.capitalize()} ({array_size})"
        
        # Add network info for grouping
        network_size = ''
        if 'single_node' in network:
            network_size = '1'
        elif '4x4' in network:
            network_size = '4x4'
        elif '8x8' in network:
            network_size = '8x8'
        
        if network_size:
            label += f" {network_size}"
        
        # Create color key based on arch and subarch
        if subarch == 'figna':
            color_key = f"{arch}_figna"
        else:
            color_key = arch
            
        base_color = base_colors.get(color_key, 'gray')
        
        # Apply shading based on network for all architectures
        if '8x8' in network:
            color = lighten_color(base_color, 0.5)
        else:
            color = base_color
        
        data_rows.append({
            'arch': arch,
            'subarch': subarch,
            'array_size': int(array_size),
            'network': network,
            'label': label,
            'color': color,
            'throughput': row['throughput'],
            'energy_efficiency': row['energy_efficiency'],
            'power_efficiency': row['power_efficiency']
        })
    
    # Sort data with custom grouping:
    # Group 1: mugi/carat 64, systolic/simd 8, tensor multi_node_2x1
    # Group 2: mugi/carat 128, systolic/simd 16, tensor multi_node_4x4  
    # Group 3: mugi/carat 256, systolic/simd 64 (single_node), tensor single_node
    def sort_key(item):
        arch = item['arch']
        array_size = item['array_size']
        network = item['network']
        subarch = item['subarch']
        
        # Define size groups - 8x8, then 16x16, then 64x64 scale up
        if ((arch in ['mugi', 'carat'] and array_size == 64) or 
            (arch in ['systolic', 'simd'] and array_size == 8) or
            (arch == 'tensor' and 'single_node' in network)):
            group = 0  # Group 1: 8x8 equivalent
        elif ((arch in ['mugi', 'carat'] and array_size == 128) or 
              (arch in ['systolic', 'simd'] and array_size == 16) or
              (arch == 'tensor' and 'multi_node_2x1' in network)):
            group = 1  # Group 2: 16x16 equivalent
        elif ((arch in ['mugi', 'carat'] and array_size == 256) or 
              (arch in ['systolic', 'simd'] and array_size == 64 and 'single_node' in network) or
              (arch == 'tensor' and 'multi_node_2x2' in network)):
            group = 2  # Group 3: 64x64 scale up
        else:
            group = 999  # Other configurations
        
        # Within each group, order by architecture type
        arch_priority = {'mugi': 0, 'carat': 1, 'systolic': 2, 'simd': 3, 'tensor': 4}
        
        # Then by subarch (mac before figna)
        subarch_priority = {'': 0, 'vlp': 1, 'mac': 2, 'figna': 3}
        
        # Finally by network size
        network_priority = {'single_node': 0, 'multi_node_2x1': 1, 'multi_node_4x4': 1, 'multi_node_8x8': 2}
        
        return (
            group,
            arch_priority.get(arch, 999),
            subarch_priority.get(subarch, 999),
            network_priority.get(network, 999)
        )
    
    sorted_data = sorted(data_rows, key=sort_key)
    
    # Extract sorted values
    labels = [item['label'] for item in sorted_data]
    colors = [item['color'] for item in sorted_data]
    throughput_values = [item['throughput'] for item in sorted_data]
    energy_values = [item['energy_efficiency'] for item in sorted_data]
    power_values = [item['power_efficiency'] for item in sorted_data]
    
    # Create x positions for bars with gaps between groups
    x_pos = []
    group_gap = 1  # Gap between groups
    current_x = 0
    
    # Find group boundaries and create positions
    group_boundaries = []
    group_centers = []
    current_group = None
    group_start_x = 0
    group_item_count = 0
    
    # Determine groups and create x positions
    for i, item in enumerate(sorted_data):
        arch = item['arch']
        array_size = item['array_size']
        network = item['network']
        
        # Determine which group this item belongs to
        if ((arch in ['mugi', 'carat'] and array_size == 64) or 
            (arch in ['systolic', 'simd'] and array_size == 8) or
            (arch == 'tensor' and 'single_node' in network)):
            item_group = 0  # Group 1: 8x8 equivalent
        elif ((arch in ['mugi', 'carat'] and array_size == 128) or 
              (arch in ['systolic', 'simd'] and array_size == 16) or
              (arch == 'tensor' and 'multi_node_2x1' in network)):
            item_group = 1  # Group 2: 16x16 equivalent
        elif ((arch in ['mugi', 'carat'] and array_size == 256) or 
              (arch in ['systolic', 'simd'] and array_size == 64 and 'single_node' in network) or
              (arch == 'tensor' and 'multi_node_2x2' in network)):
            item_group = 2  # Group 3: 64x64 scale up
        else:
            item_group = 999
        
        if current_group is None:
            current_group = item_group
            group_start_x = current_x
            group_item_count = 1
        elif current_group != item_group:
            # End of current group - add center and boundary
            group_centers.append(group_start_x + (group_item_count - 1) / 2)
            group_boundaries.append(current_x - group_gap / 2)
            
            # Start new group
            current_x += group_gap
            current_group = item_group
            group_start_x = current_x
            group_item_count = 1
        else:
            group_item_count += 1
        
        x_pos.append(current_x)
        current_x += 1
    
    # Add the last group center
    if len(sorted_data) > 0:
        group_centers.append(group_start_x + (group_item_count - 1) / 2)
    
    # Convert to numpy array for plotting
    x_pos = np.array(x_pos)
    bar_width = 0.8
    
    # Manual y-tick dictionaries for each subplot
    manual_yticks = {
        0: [6, 12, 18],  # ax1 - throughput
        1: [10, 20, 30],  # ax2 - energy efficiency  
        2: [1, 2]   # ax3 - power efficiency
    }
    
    # Plot throughput (top subplot)
    bars1 = ax1.bar(x_pos, throughput_values, bar_width, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.3,
                    label=labels)
    ax1.set_title('Norm Throughput', fontsize=font_title, pad = 3)
    ax1.tick_params(axis='y', labelsize=font_tick)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.3)
    ax1.set_yticks(manual_yticks[0])
    ax1.set_yticklabels([f'{int(tick)}x' for tick in manual_yticks[0]])
    ax1.set_xticks([])
    
    # Plot energy efficiency (middle subplot) - removed log scale
    bars2 = ax2.bar(x_pos, energy_values, bar_width,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.3)
    ax2.set_title('Norm Energy Efficiency', fontsize=font_title, pad = 3)
    ax2.tick_params(axis='y', labelsize=font_tick)
    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.3)
    ax2.set_yticks(manual_yticks[1])
    ax2.set_yticklabels([f'{int(tick)}x' for tick in manual_yticks[1]])
    ax2.set_xticks([])
    
    # Plot power efficiency (bottom subplot)
    bars3 = ax3.bar(x_pos, power_values, bar_width,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.3)
    ax3.set_title('Norm Power Efficiency', fontsize=font_title, pad = 3)
    ax3.tick_params(axis='y', labelsize=font_tick)
    ax3.grid(True, linestyle='--', alpha=0.7, linewidth=0.3)
    ax3.set_yticks(manual_yticks[2])
    ax3.set_yticklabels([f'{int(tick)}x' for tick in manual_yticks[2]])
    
    # Set group tick marks and labels on bottom subplot only
    group_labels = ['64/8/S', '128/16/2', '256/SU/4']
    if len(group_centers) >= 3:
        ax3.set_xticks(group_centers[:3])
        ax3.set_xticklabels(group_labels[:len(group_centers)], fontsize=font_tick)
    else:
        ax3.set_xticks(group_centers)
        ax3.set_xticklabels(group_labels[:len(group_centers)], fontsize=font_tick)
    
    ax1.tick_params(axis='x', length=2)  # Hide x-ticks for top subplot
    ax2.tick_params(axis='x', length=2)
    # Adjust x-tick parameters: reduce line size and move labels closer
    ax3.tick_params(axis='x', length=2, pad=1)
    
    # Create simplified legend - one entry per architecture type with network sizes
    legend_entries = {}
    legend_handles = {}
    
    # Process sorted data to create simplified legend entries
    for i, item in enumerate(sorted_data):
        arch = item['arch']
        subarch = item['subarch']
        network = item['network']
        
        # Create simplified legend label without network dimensions
        if arch == 'mugi':
            legend_key = 'Mugi'
        elif arch == 'carat':
            legend_key = 'Carat'
        elif arch == 'systolic':
            if subarch == 'figna':
                legend_key = 'SA-F'
            else:
                legend_key = 'SA'
        elif arch == 'simd':
            if subarch == 'figna':
                legend_key = 'SD-F'
            else:
                legend_key = 'SD'
        elif arch == 'tensor':
            legend_key = 'Tensor'
        else:
            legend_key = arch.capitalize()
        
        # Store the first occurrence of each legend key (no network dimensions)
        if legend_key not in legend_entries:
            legend_entries[legend_key] = item['color']
            legend_handles[legend_key] = bars1[i]
    
    # Define the desired order for legend (without network dimensions)
    legend_order = ['Mugi', 'Carat', 'SA', 'SA-F', 'SD', 'SD-F', 'Tensor']
    
    # Create ordered legend handles and labels
    ordered_handles = []
    ordered_labels = []
    for key in legend_order:
        if key in legend_handles:
            ordered_handles.append(legend_handles[key])
            ordered_labels.append(key)
    
    # Create legend
    fig.legend(ordered_handles, ordered_labels, ncol=7, fontsize=7, 
              loc='lower center', bbox_to_anchor=(0.5, .95), 
              frameon=True, columnspacing=.5, handlelength=.75, handleheight=0.5, handletextpad=0.3)
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(hspace=0.475, bottom=0.15)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path + 'noc_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'noc_breakdown.pdf', dpi=1200, bbox_inches='tight')
    plt.show()