from zoo.llm.results.query.utils import query_performance_nonlinear_metrics, compute_throughput_efficiancy, load_yaml, geomean
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def query(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']

    mugi_subarch_list = ['vlp']

    baseline_list = ['systolic']
    baseline_arch_dim_list = ['16x16']
    baseline_subarch_list = ['mac', 'pwl', 'taylor']

    mugi_throughput_module = 'magnitude_register'
    baseline_throughput_module = 'accumulator_vector'
    approximate_throughtput_module = 'adder_vector'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b']
    max_seq_len_list = ['max_seq_len_128', 'max_seq_len_256', 'max_seq_len_512', 'max_seq_len_1024', 'max_seq_len_2048', 'max_seq_len_4096']
    batch_size = 'batch_size_8'
    network = 'single_node'
    kv_paths = 'kv_heads_8'


    nonlinear_breakdown_df = pd.DataFrame()

    for arch in vlp_list + baseline_list + ['tensor']:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
            for subarch in (baseline_subarch_list if arch in baseline_list else mugi_subarch_list if arch in ['mugi'] else ['']):
                if arch == 'simd' and subarch in ['pwl', 'taylor']:
                    continue
                for max_seq_len in max_seq_len_list:
                    softmax_list = []
                    silu_list = []
                    for model in model_list:
                        module = mugi_throughput_module if arch == 'mugi' else approximate_throughtput_module if subarch in ['pwl', 'taylor'] else baseline_throughput_module
                        termination_path = 'full_termination' if arch == 'mugi' else ''
                        run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                        yaml_dict = load_yaml(run_path)

                        event_graph = yaml_dict['event_graph']
                        metric_dict = yaml_dict['metric_dict']
                        sm_metric_dict = query_performance_nonlinear_metrics(event_graph=event_graph, metric_dict=metric_dict, module=module, workload=model, event = 'softmax')
                        silu_metric_dict = query_performance_nonlinear_metrics(event_graph=event_graph, metric_dict=metric_dict, module=module, workload=model, event = 'silu')

                        if subarch == 'taylor':
                            sm_metric_dict['flops'] /= 9

                        silu_throughput_eff_dict = compute_throughput_efficiancy(silu_metric_dict)
                        sm_throughput_eff_dict = compute_throughput_efficiancy(sm_metric_dict)

                        softmax_list.append(sm_throughput_eff_dict)
                        silu_list.append(silu_throughput_eff_dict)

                    sm_throughput_eff_dict = geomean(softmax_list)
                    silu_throughput_eff_dict = geomean(silu_list)

                    sm_metric_df = pd.DataFrame(sm_throughput_eff_dict, index=[0])
                    sm_metric_df['function'] = 'softmax'
                    silu_metric_df = pd.DataFrame(silu_throughput_eff_dict, index=[0])
                    silu_metric_df['function'] = 'silu'

                    nonlinear_metric_df = pd.concat([sm_metric_df, silu_metric_df])
                    nonlinear_metric_df['arch'] = arch
                    nonlinear_metric_df['subarch'] = subarch
                    nonlinear_metric_df['arch_dim'] = arch_dim
                    nonlinear_metric_df['max_seq_len'] = max_seq_len

                    nonlinear_metric_df = nonlinear_metric_df.drop(columns=['power', 'energy'], errors='ignore')
                    nonlinear_breakdown_df = pd.concat([nonlinear_breakdown_df, nonlinear_metric_df], axis=0)

    nonlinear_breakdown_df.to_csv(output_path + 'nonlinear_breakdown.csv', index=False)

    baseline_df = nonlinear_breakdown_df[
        (nonlinear_breakdown_df['arch'] == 'systolic') &
        (nonlinear_breakdown_df['subarch'] == 'mac') &
        (nonlinear_breakdown_df['arch_dim'] == '16x16')
    ]

    numeric_columns = baseline_df.select_dtypes(include=['number']).columns
    columns_to_merge = ['max_seq_len', 'function'] + list(numeric_columns)

    merged_df = nonlinear_breakdown_df.merge(
        baseline_df[columns_to_merge],
        on=['max_seq_len', 'function'],
        suffixes=('', '_baseline')
    )

    merged_df['throughput'] = merged_df['throughput'] / merged_df['throughput_baseline']
    merged_df['energy_efficiency'] = merged_df['energy_efficiency'] / merged_df['energy_efficiency_baseline']
    merged_df['power_efficiency'] = merged_df['power_efficiency'] / merged_df['power_efficiency_baseline']

    normalized_df = merged_df.drop(
        columns=['throughput_baseline', 'energy_efficiency_baseline', 'power_efficiency_baseline']
    )

    normalized_df.to_csv(output_path + 'nonlinear_breakdown_norm.csv', index=False)

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.ColorConverter.to_rgb(c)
    return [(1 - amount) * x + amount for x in c]

def figure(input_path: str, output_path: str):
    """Generate nonlinear breakdown figure directly from CSV data."""
    try:
        # Read the normalized CSV data
        data_df = pd.read_csv(input_path + 'nonlinear_breakdown_norm.csv')
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {input_path}nonlinear_breakdown_norm.csv")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Extract unique sequence lengths and sort them
    seq_lens = sorted(data_df['max_seq_len'].str.replace('max_seq_len_', '').astype(int).unique())
    x_ticks = [str(seq_len) for seq_len in seq_lens]
    
    # Focus on specific sequence lengths for cleaner visualization
    target_seq_lens = ['128', '256', '512', '1024', '2048', '4096']
    data_dict = {'XTicks': target_seq_lens}
    
    # Architecture mapping
    arch_mapping = {
        'mugi': 'Mugi',
        'carat': 'Carat', 
        'systolic': 'VA',
        'tensor': 'Tensor'
    }
    
    # Create legend entries by extracting data from CSV
    for _, row in data_df.iterrows():
        arch = arch_mapping.get(row['arch'], row['arch'])
        function = row['function']
        arch_dim = row['arch_dim']
        subarch = row['subarch'] if pd.notna(row['subarch']) else ''
        seq_len = row['max_seq_len'].replace('max_seq_len_', '')
        
        # Skip entries not in our target sequence lengths
        if seq_len not in target_seq_lens:
            continue
            
        # Create key for this configuration
        function_key = 'SM ' if function == 'softmax' else 'SiLU '
        
        # Extract dimension info
        if arch in ['Mugi', 'Carat']:
            dim_size = arch_dim.split('x')[0]
        elif arch == 'Tensor':
            dim_size = '16'  # Use 16 for tensor
        else:  # VA/systolic
            dim_size = arch_dim.split('x')[0]
            
        dim_key = f' ({dim_size})'
        
        # Handle subarch
        subarch_key = ''
        if subarch and subarch != 'mac':
            if subarch == 'vlp':
                subarch_key = ''  # vlp is default for Mugi
            elif subarch == 'lut':
                subarch_key = ' LUT'
            elif subarch == 'pwl':
                subarch_key = ' PWL'
            elif subarch == 'taylor':
                subarch_key = ' Taylor'
                
        # Skip Taylor for SiLU (not commonly used)
        if subarch == 'taylor' and function == 'silu':
            continue
            
        # Skip PWL/Taylor for non-VA architectures (except Tensor which doesn't use them)
        if arch not in ['VA', 'Tensor'] and subarch in ['pwl', 'taylor']:
            continue
            
        key = function_key + arch + subarch_key + dim_key
        
        # Initialize data structure if not exists
        if key not in data_dict:
            data_dict[key] = {
                'NormThroughput': [0] * len(target_seq_lens),
                'NormEnergyEfficiency': [0] * len(target_seq_lens),
                'NormPowerEfficiency': [0] * len(target_seq_lens)
            }
            
        # Find index for this sequence length
        try:
            seq_idx = target_seq_lens.index(seq_len)
            data_dict[key]['NormThroughput'][seq_idx] = row['throughput']
            data_dict[key]['NormEnergyEfficiency'][seq_idx] = row['energy_efficiency'] 
            data_dict[key]['NormPowerEfficiency'][seq_idx] = row['power_efficiency']
        except ValueError:
            continue  # Skip if sequence length not in our targets
    
    # Remove any empty entries
    data_dict = {k: v for k, v in data_dict.items() if k != 'XTicks' and any(v['NormThroughput'])}
    
    # Add XTicks for sequence lengths
    data_dict['XTicks'] = target_seq_lens
    
    data = data_dict

    # -------------------------------
    # 2) FIGURE AND FONT SETTINGS
    # -------------------------------
    fig_width_pt = 240  # ACM single-column width in points
    fig_width = fig_width_pt / 72  # inches
    fig_height = fig_width / 1.8  # Adjusted height for readability

    font_title = 8
    font_tick = 7

    # -------------------------------
    # 3) COLOR SCHEME
    # -------------------------------
    base_colors = {
        'Mugi': "#0B752B",
        'Carat': "#602696", 
        'VA': "#0F6EA5",
        'PWL': "#C7A612",
        'Taylor': "#C21A1A",
        'Tensor': "#2A9B8E"  
    }

    # Generate colors for each data series
    colors = {}
    for key in data.keys():
        if 'Mugi' in key:
            if '128' in key:
                colors[key] = lighten_color(base_colors['Mugi'], 0.4)
            else:
                colors[key] = base_colors['Mugi']
        elif 'Carat' in key:
            if '128' in key:
                colors[key] = lighten_color(base_colors['Carat'], 0.4)
            else:
                colors[key] = base_colors['Carat']
        elif 'VA' in key:
            if 'PWL' in key:
                if 'SM' in key:
                    colors[key] = lighten_color(base_colors['PWL'], 0.4)
                else:
                    colors[key] = base_colors['PWL']
            elif 'Taylor' in key:
                colors[key] = base_colors['Taylor']
            else:
                if 'SM' in key:
                    colors[key] = lighten_color(base_colors['VA'], 0.4)
                else:
                    colors[key] = base_colors['VA']
        elif 'Tensor' in key:
            if 'SM' in key:
                colors[key] = lighten_color(base_colors['Tensor'], 0.4)
            else:
                colors[key] = base_colors['Tensor']


        else:
            colors[key] = 'black'  # fallback

    # -------------------------------
    # 4) CREATE SUBPLOTS AND PLOT AS BAR CHART
    # -------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True)

    x_labels = data['XTicks']
    x = np.arange(len(x_labels))
    
    # Get all categories (skip 'XTicks') and sort them using the same logic as legend
    categories = [key for key in data.keys() if key != 'XTicks']
    
    # Define the same sorting function for consistency
    def category_sort_key(label):
        # Extract components - reordered as requested
        if 'Mugi' in label:
            arch_priority = 0
        elif 'Carat' in label:
            arch_priority = 1  
        elif 'VA' in label and 'PWL' not in label and 'Taylor' not in label:
            arch_priority = 2
        elif 'Tensor' in label:
            arch_priority = 3
        elif 'PWL' in label:
            arch_priority = 4
        elif 'Taylor' in label:
            arch_priority = 5
        else:
            arch_priority = 6
            
        # Extract array size for secondary sort
        import re
        size_match = re.search(r'\((\d+)\)', label)
        array_size = int(size_match.group(1)) if size_match else 0
        
        # Function type (SM vs SiLU)
        func_priority = 0 if 'SM' in label else 1
        
        return (arch_priority, array_size, func_priority)
    
    # Sort categories to match legend order
    categories = sorted(categories, key=category_sort_key)
    num_categories = len(categories)
    bar_width = 0.93 / num_categories  # Slightly narrower bars for better spacing

    for i, key in enumerate(categories):
        throughput = data[key]['NormThroughput']
        energy_eff = data[key]['NormEnergyEfficiency']
        power_eff = data[key]['NormPowerEfficiency']
        
        color = colors.get(key, 'black')
        
        # Extract subarch for legend labels
        legend_label = key
        
        # Calculate bar positions
        bar_positions = x + (i - num_categories/2 + 0.5) * bar_width
        
        ax1.bar(bar_positions, throughput, width=bar_width, color=color, label=legend_label, 
                alpha=0.8, edgecolor='black', linewidth=0.15)
        ax2.bar(bar_positions, energy_eff, width=bar_width, color=color, label=legend_label,
                alpha=0.8, edgecolor='black', linewidth=0.15)
        ax3.bar(bar_positions, power_eff, width=bar_width, color=color, label=legend_label,
                alpha=0.8, edgecolor='black', linewidth=0.15)

    # -------------------------------
    # 6) FORMAT SUBPLOTS AND LEGEND
    # -------------------------------
    
    # Manual y-tick configuration - modify these values as needed
    manual_yticks = {
        'throughput':  [16, 32, 48],  # Set to None for auto, or list for manual
        'energy_efficiency':  [250, 500, 750],
        'power_efficiency': [6, 12, 18]
    }
    
    ytick_configs = [
        (ax1, 'Norm Throughput', manual_yticks['throughput']),
        (ax2, 'Norm Energy Efficiency', manual_yticks['energy_efficiency']), 
        (ax3, 'Norm Power Efficiency', manual_yticks['power_efficiency'])
    ]
    
    for ax, title, yticks in ytick_configs:
        ax.set_title(title, fontsize=font_title, pad=3)  # Make titles closer to figures
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=font_tick)
        ax.tick_params(axis='y', labelsize=font_tick)
        # Make x-axis tick marks smaller on all subplots
        ax.tick_params(axis='x', length=2, width=0.3)
        ax.minorticks_off()
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.3)
        
        # Set manual y-ticks if provided, otherwise use default formatting
        if yticks is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{val:.0f}x' for val in yticks])
        else:
            # Format y-axis ticks to show integers only
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}x'))
    
    # Move bottom subplot x-tick labels closer
    ax3.tick_params(axis='x', pad=1)

    # Get handles and labels from first subplot for legend
    handles, labels = ax1.get_legend_handles_labels()
    
    # Create virtual labels for architecture groups (first row)
    from matplotlib.patches import Rectangle
    virtual_handles = [
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=0),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=0),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=0),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=0)
    ]
    virtual_labels = ['  Mugi', ' Carat', ' VA-FP', '   VA-AP']
    
    # Custom sorting for legend: Mugi, Carat, VA, Tensor, PWL, Taylor
    def legend_sort_key(item):
        label = item[1]  # label is second element in (handle, label) tuple
        
        # Extract components - reordered as requested
        if 'Mugi' in label:
            arch_priority = 0
        elif 'Carat' in label:
            arch_priority = 1  
        elif 'VA' in label and 'PWL' not in label and 'Taylor' not in label:
            arch_priority = 2
        elif 'Tensor' in label:
            arch_priority = 3
        elif 'PWL' in label:
            arch_priority = 4
        elif 'Taylor' in label:
            arch_priority = 5
        else:
            arch_priority = 6
            
        # Extract array size for secondary sort
        import re
        size_match = re.search(r'\((\d+)\)', label)
        array_size = int(size_match.group(1)) if size_match else 0
        
        # Function type (SM vs SiLU)
        func_priority = 0 if 'SM' in label else 1
        
        return (arch_priority, array_size, func_priority)
    
    # Sort handles and labels together
    sorted_items = sorted(zip(handles, labels), key=legend_sort_key)
    sorted_handles, sorted_labels = zip(*sorted_items)
    
    # Create layout where virtual labels are at positions 0,4,8,12 (first row with ncol=4)
    # Calculate total grid size needed
    total_data_items = len(sorted_handles)
    
    # We need at least 4 positions for virtual labels at 0,4,8,12
    # Plus space for all data items
    min_positions_needed = max(13, 4 + total_data_items)  # At least position 12 + 1
    rows_needed = (min_positions_needed + 3) // 4  # Ceiling division for 4 columns
    grid_size = rows_needed * 4
    
    # Initialize grid
    grid_handles = [None] * grid_size
    grid_labels = [None] * grid_size
    
    # Place virtual labels at positions 0, 4, 8, 12 (top of each column)
    virtual_positions = [0, 5, 10, 15]
    for i, (handle, label) in enumerate(zip(virtual_handles, virtual_labels)):
        if i < len(virtual_positions):
            grid_handles[virtual_positions[i]] = handle
            grid_labels[virtual_positions[i]] = label
    
    # Place data items in remaining positions (skip positions 0,4,8,12)
    data_idx = 0
    for pos in range(grid_size):
        if pos not in virtual_positions and data_idx < len(sorted_handles):
            grid_handles[pos] = sorted_handles[data_idx]
            grid_labels[pos] = sorted_labels[data_idx]
            data_idx += 1
    
    # Remove None entries and create final lists
    combined_handles = [h for h in grid_handles if h is not None]
    combined_labels = [l for l in grid_labels if l is not None]
    
    # Custom reordering to move SM VA Taylor to top of last column
    # With ncol=4, we want to find SM VA Taylor and move it to position that puts it at top of column 4
    reordered_handles = list(sorted_handles)
    reordered_labels = list(sorted_labels)
    
    # Find SM VA Taylor entry
    sm_va_taylor_idx = None
    for i, label in enumerate(reordered_labels):
        if 'SM' in label and 'VA' in label and 'Taylor' in label:
            sm_va_taylor_idx = i
            break
    
    if sm_va_taylor_idx is not None:
        # Remove SM VA Taylor from its current position
        sm_va_taylor_handle = reordered_handles.pop(sm_va_taylor_idx)
        sm_va_taylor_label = reordered_labels.pop(sm_va_taylor_idx)
        
        # Calculate position for top of last column (column 4)
        # With ncol=4, positions 0,4,8,12... are tops of columns 1,2,3,4
        total_items = len(reordered_labels) + 1  # +1 because we removed one item
        rows_needed = (total_items + 3) // 4  # Ceiling division
        target_position = 3 * rows_needed  # Top of column 4 (0-indexed)
        
        # Make sure target position doesn't exceed list length
        target_position = min(target_position, len(reordered_labels))
        
        # Insert SM VA Taylor at the target position
        reordered_handles.insert(target_position, sm_va_taylor_handle)
        reordered_labels.insert(target_position, sm_va_taylor_label)
    
    # Apply the same reordering logic but to combined handles/labels
    reordered_handles = list(combined_handles)
    reordered_labels = list(combined_labels)
    
    # Find SM VA Taylor entry in the combined list (it will be offset by 4 due to virtual labels)
    sm_va_taylor_idx = None
    for i, label in enumerate(reordered_labels):
        if 'SM' in label and 'VA' in label and 'Taylor' in label:
            sm_va_taylor_idx = i
            break
    
    if sm_va_taylor_idx is not None:
        # Remove SM VA Taylor from its current position
        sm_va_taylor_handle = reordered_handles.pop(sm_va_taylor_idx)
        sm_va_taylor_label = reordered_labels.pop(sm_va_taylor_idx)
        
        # Calculate position for top of last column (column 4)
        # With ncol=4, positions 0,4,8,12... are tops of columns 1,2,3,4
        total_items = len(reordered_labels) + 1  # +1 because we removed one item
        rows_needed = (total_items + 3) // 4  # Ceiling division
        target_position = 3 * rows_needed  # Top of column 4 (0-indexed)
        
        # Make sure target position doesn't exceed list length
        target_position = min(target_position, len(reordered_labels))
        
        # Insert SM VA Taylor at the target position
        reordered_handles.insert(target_position, sm_va_taylor_handle)
        reordered_labels.insert(target_position, sm_va_taylor_label)

    # Swap indices 14 and 15 in final handles and labels
    if len(reordered_handles) > 16 and len(reordered_labels) > 16:
        reordered_handles[15], reordered_handles[16] = reordered_handles[16], reordered_handles[15]
        reordered_labels[15], reordered_labels[16] = reordered_labels[16], reordered_labels[15]
    
    # Remove architecture names from labels since they're in the header row
    cleaned_labels = []
    for i, label in enumerate(reordered_labels):
        if i not in [0, 5, 10, 15]:  # Virtual label positions
            # Remove Mugi, Carat, VA, and Tensor from labels
            cleaned_label = label.replace('Mugi', '').replace('Carat', '').replace('VA ', '').replace('Tensor', 'T')
            # Clean up any double spaces that might result
            cleaned_label = ' '.join(cleaned_label.split())
            cleaned_labels.append(cleaned_label)
        else:
            cleaned_labels.append(label)  # Keep virtual labels as is
    
    final_handles = tuple(reordered_handles)
    final_labels = tuple(cleaned_labels)

    # Create legend
    fig.legend(final_handles, final_labels, ncol=4, fontsize=7, 
              loc='upper center', bbox_to_anchor=(0.51, 1.43), 
              frameon=True, columnspacing=0.6, handlelength=.55, handletextpad=0.4, handleheight=0.5)

    plt.subplots_adjust(hspace=0.4)  # Increase spacing between subplots
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path + 'nonlinear_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'nonlinear_breakdown.pdf', dpi=1200, bbox_inches='tight')