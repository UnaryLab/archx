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
    print(f"Available sequence lengths: {x_ticks}")
    
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
    
    print(f"Generated {len(data_dict) - 1} data series:")
    for key in data_dict:
        if key != 'XTicks':
            print(f"  {key}")
    
    data = data_dict

    # -------------------------------
    # 2) FIGURE AND FONT SETTINGS
    # -------------------------------
    fig_width_pt = 240  # ACM single-column width in points
    fig_width = fig_width_pt / 72  # inches
    fig_height = fig_width / 1.8  # Adjusted height for readability

    font_title = 7
    font_tick = 6

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
    
    # Get all categories (skip 'XTicks')
    categories = [key for key in data.keys() if key != 'XTicks']
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
    for ax, title in zip([ax1, ax2, ax3], ['Norm Throughput', 'Norm Energy Efficiency', 'Norm Power Efficiency']):
        ax.set_title(title, fontsize=font_title, pad=3)  # Make titles closer to figures
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=font_tick)
        ax.tick_params(axis='y', labelsize=font_tick)
        # Make x-axis tick marks smaller on all subplots
        ax.tick_params(axis='x', length=2, width=0.3)
        ax.minorticks_off()
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.3)
        
        # Format y-axis ticks to show integers only
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}x'))
    
    # Move bottom subplot x-tick labels closer
    ax3.tick_params(axis='x', pad=1)

    # Get handles and labels from first subplot for legend
    handles, labels = ax1.get_legend_handles_labels()
    
    # Custom sorting for legend: Mugi first, then by array size, then Carat, then VA
    def legend_sort_key(item):
        label = item[1]  # label is second element in (handle, label) tuple
        
        # Extract components
        if 'Mugi' in label:
            arch_priority = 0
        elif 'Carat' in label:
            arch_priority = 1  
        elif 'VA' in label:
            arch_priority = 2
        elif 'Tensor' in label:
            arch_priority = 3
        else:
            arch_priority = 4
            
        # Extract array size for secondary sort
        import re
        size_match = re.search(r'\((\d+)\)', label)
        array_size = int(size_match.group(1)) if size_match else 0
        
        # Subarch priority (PWL, Taylor, then base)
        if 'PWL' in label:
            subarch_priority = 0
        elif 'Taylor' in label:
            subarch_priority = 1
        else:
            subarch_priority = 2
        
        # Function type (SM vs SiLU) - but for VA PWL, we want SM then SiLU
        if 'VA' in label and 'PWL' in label:
            func_priority = 0 if 'SM' in label else 1
        else:
            func_priority = 0 if 'SM' in label else 1
        
        return (arch_priority, array_size, subarch_priority, func_priority)
    
    # Sort handles and labels together
    sorted_items = sorted(zip(handles, labels), key=legend_sort_key)
    sorted_handles, sorted_labels = zip(*sorted_items)
    
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
    
    final_handles = tuple(reordered_handles)
    final_labels = tuple(reordered_labels)

    # Create legend
    fig.legend(final_handles, final_labels, ncol=4, fontsize=6, 
              loc='upper center', bbox_to_anchor=(0.55, 1.31), 
              frameon=True, columnspacing=.5, handlelength=.75, handletextpad=0.3, handleheight=0.5)

    plt.subplots_adjust(hspace=0.4)  # Increase spacing between subplots
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path + 'nonlinear_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'nonlinear_breakdown.pdf', dpi=1200, bbox_inches='tight')
    plt.show()