from zoo.llm.results.query.utils import query_execution_time, load_yaml, query_dynamic_energy, query_leakage_power, query_tag_power, query_operational_carbon, query_area
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

def query(input_path, output_path):
    query_model(input_path, output_path)
    query_hardware(input_path, output_path)

def query_hardware(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']
    gemm_vlp_carbon_module = 'and_gate'

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['16x16']
    simd_subarch_list = ['mac', 'figna']
    systolic_subarch_list = ['mac', 'figna', 'taylor', 'pwl']
    gemm_baseline_carbon_module = 'multiplier'

    mugi_nonlinear_module = 'magnitude_register'
    baseline_nonlinear_module = 'accumulator_vector'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network = 'single_node'

    kv_paths = 'kv_heads_8'

    carbon_intensity = 301  # gCO2eq/kWh
    dram_cpg = 460

    end_to_end_breakdown = pd.DataFrame()

    for arch in vlp_list + baseline_list:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
            for subarch in (systolic_subarch_list if arch in ['systolic'] else simd_subarch_list if arch in ['simd'] else mugi_subarch_list if arch in ['mugi'] else ['']):
                for model in model_list:

                    module = gemm_vlp_carbon_module if arch in vlp_list else gemm_baseline_carbon_module
                    nonlinear_module = mugi_nonlinear_module if arch in ['mugi'] else baseline_nonlinear_module
                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    kv_path = kv_paths if model in ['llama_2_70b_GQA'] else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{kv_path}/{termination_path}/')
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    onchip_carbon = query_operational_carbon(tag='onchip', event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model, CI=carbon_intensity)
                    offchip_carbon = query_operational_carbon(tag='dram', event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model, CI=carbon_intensity)

                    onchip_carbon_dict = {'carbon': onchip_carbon, 'type': 'onchip_op'}
                    offchip_carbon_dict = {'carbon': offchip_carbon, 'type': 'offchip_op'}

                    onchip_carbon_df = pd.DataFrame(onchip_carbon_dict, index=[0])
                    offchip_carbon_df = pd.DataFrame(offchip_carbon_dict, index=[0])

                    onchip_area = query_area(event_graph=event_graph, metric_dict=metric_dict, tag='onchip')
                    #offchip_area = query_area(event_graph=event_graph, metric_dict=metric_dict, tag='dram')

                    total_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model)

                    onchip_embodied_carbon = (0.55 * onchip_area) * (carbon_intensity / 3.6)
                    offchip_embodied_carbon = dram_cpg * 1

                    onchip_embodied_carbon /= 1000 # convert to kgCO2eq
                    offchip_embodied_carbon /= 1000 # convert to kgCO2eq

                    onchip_embodied_carbon *= (total_execution_time / (5 * 365 * 24 * 3600))
                    offchip_embodied_carbon *= (total_execution_time / (5 * 365 * 24 * 3600))

                    onchip_carbon_embodied_dict = {'carbon': onchip_embodied_carbon, 'type': 'onchip_em'}
                    offchip_carbon_embodied_dict = {'carbon': offchip_embodied_carbon, 'type': 'offchip_em'}

                    onchip_carbon_embodied_df = pd.DataFrame(onchip_carbon_embodied_dict, index=[0])
                    offchip_carbon_embodied_df = pd.DataFrame(offchip_carbon_embodied_dict, index=[0])

                    end_to_end_metric_df = pd.concat([onchip_carbon_df, offchip_carbon_df, onchip_carbon_embodied_df, offchip_carbon_embodied_df])
                    end_to_end_metric_df['arch'] = arch
                    end_to_end_metric_df['subarch'] = subarch
                    end_to_end_metric_df['arch_dim'] = arch_dim
                    end_to_end_metric_df['model'] = model

                    end_to_end_metric_df = end_to_end_metric_df.drop(columns=['flops', 'execution_time', 'power', 'energy'], errors='ignore')
                    end_to_end_breakdown = pd.concat([end_to_end_breakdown, end_to_end_metric_df], axis=0)

    end_to_end_breakdown.to_csv(output_path + 'hardware_carbon.csv', index=False)

    group_cols = [col for col in end_to_end_breakdown.columns if col not in ['carbon', 'type']]
    total_carbon_df = end_to_end_breakdown.groupby(group_cols)['carbon'].sum().reset_index()

    # Normalize each model separately against its own systolic mac 16x16 baseline
    end_to_end_breakdown['normalized_carbon'] = 0.0  # Initialize column
    
    for model in end_to_end_breakdown['model'].unique():
        # Get baseline for this specific model
        baseline_df = total_carbon_df[
            (total_carbon_df['arch'] == 'mugi') &
            (total_carbon_df['subarch'] == 'vlp') &
            (total_carbon_df['arch_dim'] == '256x8') &
            (total_carbon_df['model'] == model)
        ]
        
        if not baseline_df.empty:
            baseline_carbon = baseline_df['carbon'].values[0]
            # Normalize all entries for this model
            model_mask = end_to_end_breakdown['model'] == model
            end_to_end_breakdown.loc[model_mask, 'normalized_carbon'] = (
                end_to_end_breakdown.loc[model_mask, 'carbon'] / baseline_carbon
            )

    end_to_end_breakdown.to_csv(output_path + 'hardware_carbon_norm.csv', index=False)


def query_model(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']
    gemm_vlp_carbon_module = 'and_gate'

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['16x16']
    simd_subarch_list = ['mac', 'figna']
    systolic_subarch_list = ['mac', 'figna', 'taylor', 'pwl']
    gemm_baseline_carbon_module = 'multiplier'

    mugi_nonlinear_module = 'magnitude_register'
    baseline_nonlinear_module = 'accumulator_vector'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network = 'single_node'

    kv_paths = 'kv_heads_8'

    carbon_intensity = 301  # gCO2eq/kWh

    end_to_end_breakdown = pd.DataFrame()
    dram_cpg = 460

    for arch in vlp_list + baseline_list:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
            for subarch in (systolic_subarch_list if arch in ['systolic'] else simd_subarch_list if arch in ['simd'] else mugi_subarch_list if arch in ['mugi'] else ['']):
                for model in model_list:

                    module = gemm_vlp_carbon_module if arch in vlp_list else gemm_baseline_carbon_module
                    nonlinear_module = mugi_nonlinear_module if arch in ['mugi'] else baseline_nonlinear_module
                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    kv_path = kv_paths if model in ['llama_2_70b_GQA'] else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{kv_path}/{termination_path}/')
                    print(run_path)
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    proj_op_carbon = query_operational_carbon(tag=None, event_graph=event_graph, metric_dict=metric_dict, workload=model, event='projection', CI=carbon_intensity)
                    proj_op_carbon_dict = {'carbon': proj_op_carbon}

                    attn_op_carbon = query_operational_carbon(tag=None, event_graph=event_graph, metric_dict=metric_dict, workload=model, event='attention', CI=carbon_intensity)
                    attn_op_carbon_dict = {'carbon': attn_op_carbon}

                    ffn_op_carbon = query_operational_carbon(tag=None, event_graph=event_graph, metric_dict=metric_dict, workload=model, event='ffn', CI=carbon_intensity)
                    ffn_op_carbon_dict = {'carbon': ffn_op_carbon}

                    nonlinear_op_carbon = query_operational_carbon(tag=None, event_graph=event_graph, metric_dict=metric_dict, workload=model, event='nonlinear', CI=carbon_intensity)
                    nonlinear_op_carbon_dict = {'carbon': nonlinear_op_carbon}

                    total_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model)

                    # total_op_carbon = query_operational_carbon(tag=None, event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model, CI=carbon_intensity)
                    # total_op_carbon_dict = {'carbon': total_op_carbon}

                    onchip_area = query_area(event_graph=event_graph, metric_dict=metric_dict, tag='onchip')
                    #offchip_area = query_area(event_graph=event_graph, metric_dict=metric_dict, tag='dram')

                    onchip_carbon = (0.55 * onchip_area) * (carbon_intensity / 3.6)
                    offchip_carbon = dram_cpg * 1

                    onchip_carbon /= 1000 # convert to kgCO2eq
                    offchip_carbon /= 1000 # convert to kgCO2eq

                    onchip_carbon *= ((total_execution_time) / (5* 365 * 24 * 3600))
                    offchip_carbon *= ((total_execution_time) / (5* 365 * 24 * 3600))

                    onchip_carbon_dict = {'carbon': onchip_carbon, 'layer': 'onchip_em'}
                    offchip_carbon_dict = {'carbon': offchip_carbon, 'layer': 'offchip_em'}

                    onchip_carbon_df = pd.DataFrame(onchip_carbon_dict, index=[0])
                    offchip_carbon_df = pd.DataFrame(offchip_carbon_dict, index=[0])

                    proj_metric_df = pd.DataFrame(proj_op_carbon_dict, index=[0])
                    proj_metric_df['layer'] = 'projection'
                    attn_metric_df = pd.DataFrame(attn_op_carbon_dict, index=[0])
                    attn_metric_df['layer'] = 'attention'
                    ffn_metric_df = pd.DataFrame(ffn_op_carbon_dict, index=[0])
                    ffn_metric_df['layer'] = 'ffn'
                    # total_op_carbon_df = pd.DataFrame(total_op_carbon_dict, index=[0])
                    # total_op_carbon_df['layer'] = 'total'
                    nonlinear_metric_df = pd.DataFrame(nonlinear_op_carbon_dict, index=[0])
                    nonlinear_metric_df['layer'] = 'nonlinear'


                    end_to_end_metric_df = pd.concat([proj_metric_df, attn_metric_df, ffn_metric_df, nonlinear_metric_df, onchip_carbon_df, offchip_carbon_df])
                    end_to_end_metric_df['arch'] = arch
                    end_to_end_metric_df['subarch'] = subarch
                    end_to_end_metric_df['arch_dim'] = arch_dim
                    end_to_end_metric_df['model'] = model

                    end_to_end_metric_df = end_to_end_metric_df.drop(columns=['flops', 'execution_time', 'power', 'energy'], errors='ignore')
                    end_to_end_breakdown = pd.concat([end_to_end_breakdown, end_to_end_metric_df], axis=0)

    end_to_end_breakdown.to_csv(output_path + 'model_carbon.csv', index=False)

    group_cols = [col for col in end_to_end_breakdown.columns if col not in ['carbon', 'layer']]
    total_carbon_df = end_to_end_breakdown.groupby(group_cols)['carbon'].sum().reset_index()

    # Normalize each model separately against its own systolic mac 16x16 baseline
    end_to_end_breakdown['normalized_carbon'] = 0.0  # Initialize column
    
    for model in end_to_end_breakdown['model'].unique():
        # Get baseline for this specific model
        baseline_df = total_carbon_df[
            (total_carbon_df['arch'] == 'mugi') &
            (total_carbon_df['subarch'] == 'vlp') &
            (total_carbon_df['arch_dim'] == '256x8') &
            (total_carbon_df['model'] == model)
        ]
        
        if not baseline_df.empty:
            baseline_carbon = baseline_df['carbon'].values[0]
            # Normalize all entries for this model
            model_mask = end_to_end_breakdown['model'] == model
            end_to_end_breakdown.loc[model_mask, 'normalized_carbon'] = (
                end_to_end_breakdown.loc[model_mask, 'carbon'] / baseline_carbon
            )

    end_to_end_breakdown.to_csv(output_path + 'model_carbon_norm.csv', index=False)

def figure(input_path: str, output_path: str):


    df = pd.read_csv(input_path + 'model_carbon_norm.csv')
    df_hardware = pd.read_csv(input_path + 'hardware_carbon_norm.csv')

    fig_width_pt = 250
    fig_width = fig_width_pt/72
    fig_height = fig_width/4.24

    font_size = 7

    fig, axes = plt.subplots(
        1, 4, figsize=(fig_width, fig_height), sharex=False, sharey=True
    )
    plt.subplots_adjust(wspace=0)

    label_dict = {}
    for label in df.columns:
        if isinstance(df[label].iloc[0], str):
            label_dict[label] = list(df[label].unique())

    display_archs = ['mugi', 'carat', 'systolic', 'simd', 'systolic_taylor', 'systolic_pwl']
    bars_per_arch = {arch: 2 for arch in display_archs}  # 2 bars: model, hardware
    bar_width = 0.4
    group_centers = np.arange(len(display_archs))

    layer_colors = {
        'projection': '#FF6B6B',
        'attention': "#53CC6D",
        'ffn': '#45B7D1',
        'nonlinear': '#FFA07A',
        'onchip_op': "#45B7D1",
        'offchip_op': "#FFA07A",  
        'onchip_em': "#AB54C0",
        'offchip_em': "#B3AB64"
    }

    bar_type_colors = {
        'model': 'white',
        'hardware': '#E0E0E0'
    }

    for idx, model in enumerate(label_dict['model']):
        ax = axes[idx]
        model_label = '7B' if model == 'llama_2_7b' else '13B' if model == 'llama_2_13b' else '70B' if model == 'llama_2_70b' else '70B GQA'
        ax.set_title(model_label, fontsize=font_size, pad=2.5)
        if idx == len(label_dict['model']) - 1:
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('Norm kgCO2eq', fontsize=font_size, rotation=270, labelpad=8.5)

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

            # Bar positions: left (model), right (hardware)
            start_pos = group_centers[i] - bar_width
            model_pos = start_pos
            hardware_pos = start_pos + bar_width

            # --- Model carbon ---
            if arch == 'mugi':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['subarch'] == 'vlp') &
                    (df['arch_dim'].isin(['256x8']))
                ]
                filtered_hw = df_hardware[
                    (df_hardware['model'] == model) &
                    (df_hardware['arch'] == arch) &
                    (df_hardware['subarch'] == 'vlp') &
                    (df_hardware['arch_dim'].isin(['256x8']))
                ]
            elif arch == 'carat':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['arch_dim'].isin(['256x8']))
                ]
                filtered_hw = df_hardware[
                    (df_hardware['model'] == model) &
                    (df_hardware['arch'] == arch) &
                    (df_hardware['arch_dim'].isin(['256x8']))
                ]
            elif arch == 'simd':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['subarch'] == 'mac') &
                    (df['arch_dim'] == '16x16')
                ]
                filtered_hw = df_hardware[
                    (df_hardware['model'] == model) &
                    (df_hardware['arch'] == arch) &
                    (df_hardware['subarch'] == 'mac') &
                    (df_hardware['arch_dim'] == '16x16')
                ]
            elif arch == 'systolic':
                filtered_df = df[
                    (df['model'] == model) &
                    (df['arch'] == arch) &
                    (df['subarch'] == subarch) &
                    (df['arch_dim'] == '16x16')
                ]
                filtered_hw = df_hardware[
                    (df_hardware['model'] == model) &
                    (df_hardware['arch'] == arch) &
                    (df_hardware['subarch'] == subarch) &
                    (df_hardware['arch_dim'] == '16x16')
                ]
            else:
                filtered_df = pd.DataFrame()
                filtered_hw = pd.DataFrame()

            # --- Plot model carbon bar (stacked by layer) ---
            bottom = 0
            for layer in ['projection', 'attention', 'ffn', 'nonlinear', 'offchip_em', 'onchip_em']:
                layer_filtered_df = filtered_df[filtered_df['layer'] == layer]
                heights = layer_filtered_df['normalized_carbon'].tolist()
                if heights:
                    ax.bar(
                        model_pos,
                        heights[0],
                        width=bar_width,
                        bottom=bottom,
                        color=layer_colors[layer],
                        edgecolor='black',
                        linewidth=0.3,
                        label=layer if i == 0 and idx == 0 else None
                    )
                    bottom += heights[0]

            # --- Plot hardware carbon bar (stacked by type) ---
            bottom_hw = 0
            for hw_type in ['offchip_op', 'onchip_op', 'offchip_em', 'onchip_em']:
                hw_filtered_df = filtered_hw[filtered_hw['type'] == hw_type]
                heights_hw = hw_filtered_df['normalized_carbon'].tolist()
                if heights_hw:
                    ax.bar(
                        hardware_pos,
                        heights_hw[0],
                        width=bar_width,
                        bottom=bottom_hw,
                        color=layer_colors[hw_type],
                        edgecolor='black',
                        linewidth=0.3,
                        label=hw_type if i == 0 and idx == 0 else None
                    )
                    bottom_hw += heights_hw[0]

        # Tick positions and labels
        tick_positions = [group_centers[i] - bar_width/2 for i in range(len(display_archs))]
        display_arch_labels = []
        for display_arch in display_archs:
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
            elif display_arch == 'simd':
                display_arch_labels.append('SD')
            else:
                display_arch_labels.append(display_arch)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(display_arch_labels, fontsize=font_size-1)
        ax.set_xlim(-1, len(display_archs))

        # Top x-axis for dimension labels
        ax2 = ax.twiny()
        dim_tick_positions = []
        dim_tick_labels = []
        vlp_indices = [i for i, arch in enumerate(display_archs) if arch in ['mugi', 'carat']]
        systolic_indices = [i for i, arch in enumerate(display_archs) if arch.startswith('systolic')]
        if vlp_indices:
            vlp_positions = [group_centers[i] - bar_width/2 for i in vlp_indices]
            vlp_center = np.mean(vlp_positions)
            dim_tick_positions.append(vlp_center)
            dim_tick_labels.append('256')
        if systolic_indices:
            systolic_positions = [group_centers[i] - bar_width/2 for i in systolic_indices]
            systolic_center = np.mean(systolic_positions)
            dim_tick_positions.append(systolic_center)
            dim_tick_labels.append('16')
        ax2.set_xticks(dim_tick_positions)
        ax2.set_xticklabels(dim_tick_labels, fontsize=font_size-1)
        ax2.set_xlim(ax.get_xlim())

        ax.tick_params(axis='both', width=0.3, length=2)
        ax2.tick_params(axis='x', width=0.3, length=2)
        ax.tick_params(axis='x', pad=0.5)
        ax2.tick_params(axis='x', pad=0.25)
        ax.tick_params(axis='y', labelsize=font_size, pad=1)
        ax.set_yticks([0, 0.5, 1, 1.5])
        ax.set_ylim(0, 1.5)

        if idx > 0:
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5, color='gray')
            ax.spines['left'].set_visible(False)
        else:
            ax.tick_params(axis='y', left=True, labelleft=True, labelsize=font_size)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5, color='gray')
            ax.spines['left'].set_linewidth(0.3)
        if idx < len(label_dict['model']) - 1:
            ax.spines['right'].set_visible(False)
        else:
            ax.spines['right'].set_linewidth(0.3)
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        if idx == 0:
            ax2.spines['top'].set_linewidth(0.3)
        else:
            ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    # Legend for layers
    handles = [plt.Rectangle((0,0),1,1, color=layer_colors[layer]) for layer in ['projection', 'attention', 'ffn', 'nonlinear', 'onchip_op', 'offchip_op', 'onchip_em', 'offchip_em']]
    labels = ['Projection', 'Attention', 'FFN', 'Nonlinear', 'Onchip Op', 'Offchip Op', 'Onchip Em', 'Offchip Em']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 1.575), ncol=4, fontsize=font_size,
               columnspacing=1, handlelength=.75, handleheight=0.5, handletextpad=1)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.25)

    plt.savefig(output_path + 'carbon.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_path + 'carbon.pdf', dpi=1200, bbox_inches='tight')