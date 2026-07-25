from archx.metric import aggregate_event_metric, aggregate_tag_metric, query_module_metric
from zoo.llm.results.query.utils import query_area, query_tag_power, load_yaml, query_dynamic_energy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import math

def figure(input_path, output_path):
    area_df = pd.read_csv(input_path + 'array_area_breakdown_bar.csv')
    power_df = pd.read_csv(input_path + 'array_power_breakdown_bar.csv')
    area_scaled_df = pd.read_csv(input_path + 'array_area_breakdown_scaled.csv')
    power_scaled_df = pd.read_csv(input_path + 'array_power_breakdown_scaled.csv')

    fig_width_pt = 270  # ACM single-column width in points
    fig_width = 270 / 72  # Convert to inches
    fig_height = fig_width * 1.3
    drop_columns = ['arch', 'subarch', 'arch_dim', 'total']
    scaled_drop_columns = ['arch', 'subarch', 'arch_dim', 'network', 'total']

    legend_font_size = 7
    tick_font_size = 6
    bar_font_size = 6

    area_value_df = area_df.drop(columns=drop_columns)
    power_value_df = power_df.drop(columns=drop_columns)
    
    # Reorder columns to match desired legend order
    desired_order = ['accumulator', 'fifo', 'pe', 'nonlinear', 'vector', 'tc', 'value_reuse']
    area_value_df = area_value_df[desired_order]
    power_value_df = power_value_df[desired_order]

    area_scaled_df = area_scaled_df[area_scaled_df['arch'] != 'tensor']
    power_scaled_df = power_scaled_df[power_scaled_df['arch'] != 'tensor']

    area_value_scaled_df = area_scaled_df.drop(columns=scaled_drop_columns)
    power_value_scaled_df = power_scaled_df.drop(columns=scaled_drop_columns)

    arch_labels = ['Mugi', 'Mugi-L', 'Carat', 'S-F', 'SD-F']

    # Extract the first dimension from arch_dim (e.g., "128" from "128x8")
    x_1_labels = [dim.split('x')[0] for dim in area_df['arch_dim'].to_list()]

    raw_labels = area_value_df.columns.to_list()
    labels = [label.replace('_', ' ').capitalize() for label in raw_labels]
    labels = ['Acc', 'Fifo', 'PE', 'Nonlinear', 'Vector', 'TC', 'VR']

    scaled_raw_labels = area_value_scaled_df.columns.to_list()
    scaled_labels = [scaled_label.replace('_', ' ').capitalize() for scaled_label in scaled_raw_labels]
    scaled_labels = ['Array', 'SRAM', 'NoC']
    catagories = area_df['arch'].to_list()

    legend_list = ['Acc', 'Array', 'Fifo', 'SRAM', 'PE', 'NoC', 'Nonlinear', 'Vector', 'TC', 'Value Reuse']

    color_dict = {
        'fifo': "#85E68F",
        'pe': "#D3B9EA",
        'tc': "#F06AFC",
        'value_reuse': "#42A658",
        'accumulator': "#9EDDE8",
        'nonlinear': "#40EE9D",
        'vector': "#8AA5F7",
        'array': "#E58080",
        'node_memory': "#F49C59",
        'router': "#CAC561"
    }

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Swap dimensions for horizontal layout

    bar_width = 0.22  # Reduced bar width to fit 4 columns
    y_offset = 0.7 # Offset to shift bars down
    y = np.arange(len(catagories)) + y_offset  # Now using y positions with offset
    
    # Define positions for 4 columns side by side
    y_area = y - 3*bar_width/2       # Area (original)
    y_power = y - bar_width/2        # Power (original)
    y_area_scaled = y + bar_width/2  # Area (scaled)
    y_power_scaled = y + 3*bar_width/2  # Power (scaled)

    area_list_pre = None
    power_list_pre = None
    area_scaled_list_pre = None
    power_scaled_list_pre = None

    
    # Plot original area and power bars
    for i, (area_col, power_col) in enumerate(zip(area_value_df.columns, power_value_df.columns)):
        area_list = area_value_df[area_col].to_numpy()
        power_list = power_value_df[power_col].to_numpy()

        area_label_list = area_value_df[area_col].to_list()
        power_label_list = power_value_df[power_col].to_list()

        area_label_list = ['' if value <= 20 else f'{value:.1f}%' for value in area_label_list]
        power_label_list = ['' if value <= 20 else f'{value:.1f}%' for value in power_label_list]

        if i > 0:
            area_bar = plt.barh(y_area, area_list, height=bar_width,
                    left=area_list_pre, color=color_dict[raw_labels[i]],
                    edgecolor='black', linewidth=0.3)
            power_bar = plt.barh(y_power, power_list, height=bar_width,
                    left=power_list_pre, color=color_dict[raw_labels[i]],
                    edgecolor='black', linewidth=0.3)

            area_list_pre = np.add(area_list_pre, area_list)
            power_list_pre = np.add(power_list_pre, power_list)
        else:
            area_bar = plt.barh(y_area, area_list, height=bar_width,
                    color=color_dict[raw_labels[i]],
                    edgecolor='black', linewidth=0.3)
            power_bar = plt.barh(y_power, power_list, height=bar_width,
                    color=color_dict[raw_labels[i]],
                    edgecolor='black', linewidth=0.3)

            area_list_pre = np.array(area_list)
            power_list_pre = np.array(power_list)

    # Plot scaled area and power bars
    for i, (area_scaled_col, power_scaled_col) in enumerate(zip(area_value_scaled_df.columns, power_value_scaled_df.columns)):
        area_scaled_list = area_scaled_df[area_scaled_col].to_numpy()
        power_scaled_list = power_scaled_df[power_scaled_col].to_numpy()

        area_label_list = area_scaled_df[area_scaled_col].to_list()
        power_label_list = power_scaled_df[power_scaled_col].to_list()

        area_label_list = ['' if value <= 20 else f'{value:.1f}%' for value in area_label_list]
        power_label_list = ['' if value <= 20 else f'{value:.1f}%' for value in power_label_list]

        if i > 0:
            area_scaled_bar = plt.barh(y_area_scaled, area_scaled_list, height=bar_width,
                     left=area_scaled_list_pre, color=color_dict[scaled_raw_labels[i]],
                     edgecolor='black', linewidth=0.3)
            power_scaled_bar = plt.barh(y_power_scaled, power_scaled_list, height=bar_width,
                     left=power_scaled_list_pre, color=color_dict[scaled_raw_labels[i]],
                     edgecolor='black', linewidth=0.3)
            area_scaled_list_pre = np.add(area_scaled_list_pre, area_scaled_list)
            power_scaled_list_pre = np.add(power_scaled_list_pre, power_scaled_list)

        else:
            area_scaled_bar = plt.barh(y_area_scaled, area_scaled_list, height=bar_width,
                     color=color_dict[scaled_raw_labels[i]],
                     edgecolor='black', linewidth=0.3)
            power_scaled_bar = plt.barh(y_power_scaled, power_scaled_list, height=bar_width,
                     color=color_dict[scaled_raw_labels[i]],
                     edgecolor='black', linewidth=0.3)
            area_scaled_list_pre = np.array(area_scaled_list)
            power_scaled_list_pre = np.array(power_scaled_list)

    # Add total value annotations inside the first bar of each group
    area_totals = area_df['total'].to_numpy()
    power_totals = power_df['total'].to_numpy()

    area_scaled_totals = area_scaled_df['total'].to_numpy()
    power_scaled_totals = power_scaled_df['total'].to_numpy()
    
    for i, (y_pos, area_total, power_total, area_scaled_total, power_scaled_total) in enumerate(zip(y, area_totals, power_totals, area_scaled_totals, power_scaled_totals)):
        ax.text(1, y_area[i], f'{area_total:.1f} mm²', 
                va='center', ha='left', fontsize=bar_font_size, color='black', weight='bold')
        ax.text(1, y_power[i], f'{power_total:.1f} mW', 
                va='center', ha='left', fontsize=bar_font_size, color='black', weight='bold')
        
        ax.text(1, y_area_scaled[i], f'{area_scaled_total:.1f} mm²', 
                va='center', ha='left', fontsize=bar_font_size, color='black', weight='bold')
        ax.text(1, y_power_scaled[i], f'{power_scaled_total:.1f} W', 
                va='center', ha='left', fontsize=bar_font_size, color='black', weight='bold')

    # Create custom legend - organize so scaled values appear in last row
    legend_handles = []
    legend_scaled_handles = []
    
    # Add Node: label as first item in the legend
    legend_handles.append(plt.Rectangle((0,0),1,1, 
                         color='white', alpha=0, 
                         label='Node:'))
    
    # Add original labels first (these will fill the first row completely)
    for i, raw_label in enumerate(raw_labels):
        # Add virtual blank label at index 5 (before the 5th original label)
        append_label = labels[i] if labels[i] != 'Value Reuse' else 'VR'
        legend_handles.append(plt.Rectangle((0,0),1,1, 
                             color=color_dict[raw_label], 
                             label=append_label))
    
    # Add NoC (4x4): label as first item in the scaled legend
    legend_scaled_handles.append(plt.Rectangle((0,0),1,1, 
                                color='white', alpha=0, 
                                label='NoC (4x4):'))
    
    # Add scaled labels (these will be in the second row)
    for i, scaled_raw_label in enumerate(scaled_raw_labels):
        legend_scaled_handles.append(plt.Rectangle((0,0),1,1, 
                             color=color_dict[scaled_raw_label], 
                             label=scaled_labels[i]))
    
    # Add legend with ncol set to the number of original labels to ensure scaled values go to second row
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.48, 1.014), 
               frameon=False, fontsize=legend_font_size, ncol=math.ceil(len(legend_handles)/2), columnspacing=0.75, handlelength=1, handletextpad=0.5)
    
    # Add scaled legend below the original legend
    
    
    # Add labels and formatting (architecture labels on left y-axis)
    # Set left y-axis ticks at architecture group centers to align with architecture labels
    # Calculate group centers based on the 4 columns
    group_centers = [pos for pos in y]  # Center of each group (between the 4 bars)
    
    # Calculate architecture group centers for y-ticks
    from collections import defaultdict
    arch_subarch_positions = defaultdict(list)
    
    # Group bar positions by architecture and subarch combination
    for i, (arch, subarch) in enumerate(zip(area_df['arch'], area_df['subarch'])):
        if str(subarch) != 'nan':
            key = f"{arch}_{subarch}"
        else:
            key = f"{arch}_"
        arch_subarch_positions[key].append(y[i])
    
    # Calculate architecture center positions for ticks
    expected_combinations = [
        ('mugi', 'vlp', 'Mugi'),
        ('mugi', 'lut', 'Mugi-L'), 
        ('carat', '', 'Carat'),
        ('systolic', 'figna', 'S-F'),
        ('simd', 'figna', 'SD-F')
    ]
    
    arch_tick_positions = []
    for arch, subarch, display_name in expected_combinations:
        key = f"{arch}_{subarch}"
        if key in arch_subarch_positions:
            positions = arch_subarch_positions[key]
            center_pos = sum(positions) / len(positions)
            arch_tick_positions.append(center_pos)
    
    # Set y-ticks at architecture centers
    ax.set_yticks(arch_tick_positions)
    # Create empty labels matching the number of architecture centers
    empty_labels = [''] * len(arch_tick_positions)
    ax.set_yticklabels(empty_labels, fontsize=tick_font_size)  # Hide tick labels, just show ticks
    ax.tick_params(axis='y', width=0.5, length=2, pad=10)
    
    # Add right y-axis with dimension labels
    ax3 = ax.twinx()

    ax3.legend(handles=legend_scaled_handles, loc='upper center', bbox_to_anchor=(0.48, 0.0478), 
               frameon=False, fontsize=legend_font_size, ncol=len(legend_scaled_handles), columnspacing=0.75,
               handlelength=1)

    # Position the right axis labels at group centers for dimensions  
    ax3.set_yticks(group_centers[::-1])  # Use reversed group centers
    # Make sure x_1_labels matches the length of group_centers
    x_1_labels_matched = x_1_labels[:len(group_centers)] if len(x_1_labels) > len(group_centers) else x_1_labels
    ax3.set_yticklabels(x_1_labels_matched, fontsize=tick_font_size)  # Show dimension labels on right
    ax3.tick_params(axis='y', width=0.5, length=2, pad=1)
    
    # Add architecture group labels on the left y-axis aligned with group centers
    # Position labels to span across all dimensions of the same architecture+subarch combination
    # First, group the positions by architecture and subarch
    
    # Position the architecture labels
    for arch, subarch, display_name in expected_combinations:
        key = f"{arch}_{subarch}"
        if key in arch_subarch_positions:
            positions = arch_subarch_positions[key]
            # Calculate the center of all positions for this architecture+subarch
            center_pos = sum(positions) / len(positions)
            ax.text(-1.5, center_pos, display_name, va='center', ha='right', fontsize=tick_font_size)
    ax3.set_ylim(ax.get_ylim())  # Match the main axis limits
    
    # Format x-axis with percentage signs and match y-axis line width
    x_tick_positions = [0, 50, 100]
    x_tick_labels = [f'{int(tick)}%' for tick in x_tick_positions]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=tick_font_size)
    ax.tick_params(axis='x', labelsize=8, width=0.5, length=2, pad=1)
    
    # Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.grid(False)
    ax3.grid(False)

    # Set thin line widths for all axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    for spine in ax3.spines.values():
        spine.set_linewidth(0.5)
    
    # Add grid for better readability - only at 0% and 100%, not 50%
    ax.set_xticks([0, 50, 100])  # Keep all tick marks
    ax.grid(axis='x', alpha=0.3, linestyle='--')  # Grid on x-axis
    # Turn off grid line at 50% while keeping the tick
    ax.axvline(x=0, color='gray', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axvline(x=100, color='gray', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(False)  # Turn off automatic grid
    ax.set_xlim(0, 100)  # Set x-axis limits for percentages
    
    # Set y-axis limits to accommodate offset bars with minimal bottom space
    ax.set_ylim(y_offset - 1.1, len(catagories) + y_offset - 0.1)
    
    # Invert y-axis so mugi appears at top and sd-f at bottom
    ax.invert_yaxis()

    # Minimize whitespace around the plot and create more space at top
    plt.subplots_adjust(left=0.1, right=0.95, top=0.75, bottom=0.25)
    plt.tight_layout(pad=0.5)  # Reduce padding
    
    plt.savefig(output_path + 'array_breakdown_bar.png', dpi=1200, 
                bbox_inches='tight', pad_inches=0.025)  # Minimize padding around saved figure
    plt.savefig(output_path + 'array_breakdown_bar.pdf', dpi=1200, 
                bbox_inches='tight', pad_inches=0.025)
    plt.close()

def query(input_path, output_path):
    query_node(input_path=input_path, output_path=output_path)
    query_noc(input_path=input_path, output_path=output_path)

def query_noc(input_path, output_path):

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16']
    baseline_subarch_list = ['figna']

    
    model = 'llama_2_7b'
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network_list = ['multi_node_4x4']
    kv_heads = 8

    #tag_list = ['fifo', 'pe', 'tc', 'value_reuse', 'accumulator', 'nonlinear', 'vector', 'memory', 'node_memory']
    tag_list = ['array', 'node_memory', 'router']
    df_labels = ['arch', 'subarch', 'arch_dim', 'total'] + tag_list
    area_df = pd.DataFrame(columns=df_labels)
    power_df = pd.DataFrame(columns=df_labels)

    for arch in vlp_list + baseline_list + ['tensor']:
        for subarch in (baseline_subarch_list if arch in baseline_list else mugi_subarch_list if arch in ['mugi'] else ['']):
            for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch == 'tensor' else ['']):
            
                for network in network_list:
                    if network == 'multi_node_4x4' and arch == 'tensor':
                        continue
                    if network != 'multi_node_4x4' and arch != 'tensor':
                        continue

                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    if arch == 'carat':
                        subarch_label = 'vlp'
                    elif arch == 'tensor':
                        subarch_label = 'core'
                    else:
                        subarch_label = subarch

                    area_row = {
                        'arch': arch,
                        'subarch': subarch_label,
                        'arch_dim': arch_dim,
                        'network': network
                    }

                    power_row = {
                        'arch': arch,
                        'subarch': subarch_label,
                        'arch_dim': arch_dim,
                        'network': network
                    }

                    total_area = 0
                    total_power = 0
                    for tag in tag_list:
                        tag_area = query_area(tag=tag, event_graph=event_graph, metric_dict=metric_dict, workload=model)
                        tag_power = query_tag_power(tag=tag, event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model) / 1000

                        total_area += tag_area
                        total_power += tag_power
                        area_row[tag] = tag_area
                        power_row[tag] = tag_power

                    # tag_array_area = query_area(tag='array', event_graph=event_graph, metric_dict=metric_dict, workload=model)
                    # tag_array_power = query_tag_power(tag='array', event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model)

                    area_row['total'] = total_area
                    power_row['total'] = total_power
                    # area_row['array'] = tag_array_area
                    # power_row['array'] = tag_array_power

                    area_df = pd.concat([area_df, pd.DataFrame([area_row])], ignore_index=True) if not area_df.empty else pd.DataFrame([area_row])
                    power_df = pd.concat([power_df, pd.DataFrame([power_row])], ignore_index=True) if not power_df.empty else pd.DataFrame([power_row])
        
    for (area_idx, area_row), (power_idx, power_row) in zip(area_df.iterrows(), power_df.iterrows()):
        area_total = area_row['total']
        power_total = power_row['total']
        
        for tag in tag_list:
            if area_total > 0:
                area_df.loc[area_idx, tag] = (area_row[tag] / area_total) * 100
            if power_total > 0:
                power_df.loc[power_idx, tag] = (power_row[tag] / power_total) * 100

    area_df.to_csv(output_path + 'array_area_breakdown_scaled.csv', index=False)
    power_df.to_csv(output_path + 'array_power_breakdown_scaled.csv', index=False)

def query_node(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16']
    baseline_subarch_list = ['figna']

    network = 'single_node'
    model = 'llama_2_7b'
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    
    tag_list = ['accumulator', 'fifo', 'pe', 'nonlinear', 'vector', 'tc', 'value_reuse']
    df_labels = ['arch', 'subarch', 'arch_dim', 'total'] + tag_list
    area_df = pd.DataFrame(columns=df_labels)
    power_df = pd.DataFrame(columns=df_labels)

    for arch in vlp_list + baseline_list:
        for subarch in (baseline_subarch_list if arch in baseline_list else mugi_subarch_list if arch in ['mugi'] else ['']):
            for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch == 'tensor' else ['']):
            
                termination_path = 'full_termination' if arch == 'mugi' else ''
                run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                yaml_dict = load_yaml(run_path)

                event_graph = yaml_dict['event_graph']
                metric_dict = yaml_dict['metric_dict']

                if arch == 'mugi' and subarch == 'vlp' and arch_dim == '128x8':
                    fifo_area = query_area(event_graph=event_graph, metric_dict=metric_dict, workload=model, module='pe_fifo') 

                area_row = {
                    'arch': arch,
                    'subarch': subarch,
                    'arch_dim': arch_dim,
                }

                power_row = {
                    'arch': arch,
                    'subarch': subarch,
                    'arch_dim': arch_dim,
                }

                total_area = 0
                total_power = 0
                for tag in tag_list:
                    try:
                        tag_area = query_area(tag=tag, event_graph=event_graph, metric_dict=metric_dict, workload=model)
                        tag_power = query_tag_power(tag=tag, event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model)
                    except:
                        tag_area = 0
                        tag_power = 0
                    total_area += tag_area
                    total_power += tag_power
                    area_row[tag] = tag_area
                    power_row[tag] = tag_power

                # tag_array_area = query_area(tag='array', event_graph=event_graph, metric_dict=metric_dict, workload=model)
                # tag_array_power = query_tag_power(tag='array', event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model)

                area_row['total'] = total_area
                power_row['total'] = total_power
                # area_row['array'] = tag_array_area
                # power_row['array'] = tag_array_power

                area_df = pd.concat([area_df, pd.DataFrame([area_row])], ignore_index=True) if not area_df.empty else pd.DataFrame([area_row])
                power_df = pd.concat([power_df, pd.DataFrame([power_row])], ignore_index=True) if not power_df.empty else pd.DataFrame([power_row])
    
    area_df.to_csv(output_path + 'array_area_breakdown_bar_values.csv', index=False)
    power_df.to_csv(output_path + 'array_area_breakdown_bar_values.csv', index=False)

    # Convert values to percentages
    for (area_idx, area_row), (power_idx, power_row) in zip(area_df.iterrows(), power_df.iterrows()):
        area_total = area_row['total']
        power_total = power_row['total']
        
        for tag in tag_list:
            if area_total > 0:
                area_df.loc[area_idx, tag] = (area_row[tag] / area_total) * 100
            if power_total > 0:
                power_df.loc[power_idx, tag] = (power_row[tag] / power_total) * 100

    area_df.to_csv(output_path + 'array_area_breakdown_bar.csv', index=False)
    power_df.to_csv(output_path + 'array_power_breakdown_bar.csv', index=False)