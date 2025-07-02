from zoo.llm.results.query.utils import query_performance_metrics, compute_throughput_efficiancy, load_yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

def query(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']
    vlp_throughput_module = 'and_gate'

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['16x16']
    baseline_subarch_list = ['mac', 'figna']
    baseline_throughput_module = 'multiplier'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network = 'single_node'

    gemm_breakdown_df = pd.DataFrame()

    for arch in vlp_list + baseline_list:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else ['']):
                for model in model_list:

                    module = vlp_throughput_module if arch in vlp_list else baseline_throughput_module
                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    proj_metric_dict = query_performance_metrics(event_graph=event_graph, metric_dict=metric_dict, module=module, workload=model, event = 'projection')
                    proj_throughput_eff_dict = compute_throughput_efficiancy(proj_metric_dict)

                    attn_metric_dict = query_performance_metrics(event_graph=event_graph, metric_dict=metric_dict, module=module, workload=model, event = 'attention')
                    attn_throughput_eff_dict = compute_throughput_efficiancy(attn_metric_dict)
                    
                    ffn_metric_dict = query_performance_metrics(event_graph=event_graph, metric_dict=metric_dict, module=module, workload=model, event = 'ffn')
                    ffn_throughput_eff_dict = compute_throughput_efficiancy(ffn_metric_dict)

                    proj_metric_df = pd.DataFrame(proj_throughput_eff_dict, index=[0])
                    proj_metric_df['layer'] = 'projection'
                    attn_metric_df = pd.DataFrame(attn_throughput_eff_dict, index=[0])
                    attn_metric_df['layer'] = 'attention'
                    ffn_metric_df = pd.DataFrame(ffn_throughput_eff_dict, index=[0])
                    ffn_metric_df['layer'] = 'ffn'


                    gemm_metric_df = pd.concat([proj_metric_df, attn_metric_df, ffn_metric_df])
                    gemm_metric_df['arch'] = arch
                    gemm_metric_df['subarch'] = subarch
                    gemm_metric_df['arch_dim'] = arch_dim
                    gemm_metric_df['model'] = model

                    gemm_metric_df = gemm_metric_df.drop(columns=['flops', 'execution_time', 'power', 'energy'], errors='ignore')
                    gemm_breakdown_df = pd.concat([gemm_breakdown_df, gemm_metric_df], axis=0)

    gemm_breakdown_df.to_csv(output_path + 'gemm_breakdown.csv', index=False)

    baseline_df = gemm_breakdown_df[
        (gemm_breakdown_df['arch'] == 'systolic') &
        (gemm_breakdown_df['subarch'] == 'mac') &
        (gemm_breakdown_df['arch_dim'] == '16x16') &
        (gemm_breakdown_df['model'] == 'llama_2_7b')
    ]

    matching_columns = ['layer']

    numeric_columns = baseline_df.select_dtypes(include=['number']).columns
    columns_to_merge = matching_columns + list(numeric_columns)

    merged_df = gemm_breakdown_df.merge(
        baseline_df[columns_to_merge],
        on=matching_columns,
        suffixes=('', '_baseline')
    )

    merged_df['throughput'] = merged_df['throughput'] / merged_df['throughput_baseline']
    merged_df['energy_efficiency'] = merged_df['energy_efficiency'] / merged_df['energy_efficiency_baseline']
    merged_df['power_efficiency'] = merged_df['power_efficiency'] / merged_df['power_efficiency_baseline']

    normalized_df = merged_df.drop(
        columns=['throughput_baseline', 'energy_efficiency_baseline', 'power_efficiency_baseline']
    )

    normalized_df.to_csv(output_path + 'gemm_breakdown_norm.csv', index=False)

def lighten_color(color, amount=0.5):
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = mcolors.ColorConverter.to_rgb(c)
        return [(1 - amount) * x + amount for x in c]

def figure(input_path: str, output_path: str):
    categories = ['proj', 'attn']
    x_labels = ['7B', '13B', '70B', '70B GQA']

    df = pd.read_csv(input_path + 'gemm_breakdown_norm.csv')

    key = 'Throughput'
    layer = 'proj'
    model = 'Mugi'
    size = '128'

    data_df = {'Throughput': {}, 'Energy Efficiency': {}, 'Power Efficiency': {}}
    for key in data_df.keys():
        key_label = 'throughput' if key == 'Throughput' else 'energy_efficiency' if key == 'Energy Efficiency' else 'power_efficiency'
        for layer in categories:
            layer_label = 'projection' if layer == 'proj' else 'attention' if layer == 'attn' else 'ffn'
            data_df[key][layer] = {}
            for model in ['Mugi', 'Carat', 'SA', 'SA-F', 'SD', 'SD-F']:
                model_label = 'mugi' if model == 'Mugi' else 'carat' if model == 'Carat' else 'systolic' if model == 'SA' else 'simd' if model == 'SD' else 'systolic' if model == 'SA-F' else 'simd'
                subarch_label = np.nan if model == 'Mugi' or model == 'Carat' else 'mac' if model == 'SA' or model == 'SD' else 'figna'
                for size_def in [1, 2]:
                    if size_def == 1:
                        if model == 'Mugi' or model == 'Carat':
                            size = '128'
                            arch_dim = '128x8'
                        else:
                            size = '16'
                            arch_dim = '16x16'
                    else:
                        if model == 'Mugi' or model == 'Carat':
                            size = '256'
                            arch_dim = '256x8'
                        else:
                            size = '16'
                            arch_dim = '16x16'

                    if layer == 'attn': 
                        values = df[
                            (df['layer'] == layer_label) &
                            (df['arch'] == model_label) &
                            (df['arch_dim'] == arch_dim) &
                            (pd.isna(df['subarch']) if pd.isna(subarch_label) else df['subarch'] == subarch_label)
                        ]

                    else:
                        values = df[
                            (df['layer'] == layer_label) &
                            (df['arch'] == model_label) &
                            (df['arch_dim'] == arch_dim) &
                            (pd.isna(df['subarch']) if pd.isna(subarch_label) else df['subarch'] == subarch_label) &
                            (df['model'] != 'llama_2_70b_GQA')
                        ]
                    values = values[key_label].values.flatten().tolist()

                    data_df[key][layer][model + ' (' + size + ')'] = values

    data = data_df
    # Define figure dimensions and font sizes
    fig_width_pt = 250  # ACM single-column width in points
    fig_width = fig_width_pt / 72  # Convert to inches
    fig_height = fig_width * .9 # Adjusted height for readability

    font_title = 5.5
    font_label = 5
    font_tick = 5

    # Define base colors (added a color for Carat)
    base_colors = {
        'Mugi': "forestgreen",
        'Carat': "rebeccapurple",
        'SA': "dodgerblue",
        'SA-F': "red",
        'SD': "grey",
        'SD-F': "orange"
    }

    # Function to lighten a color
    def lighten_color(color, amount=0.5):
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = mcolors.ColorConverter.to_rgb(c)
        return [(1 - amount) * x + amount for x in c]

    # Create specific colors for each model and size, including Carat
    colors = {
        'Mugi (128)': base_colors['Mugi'],
        'Mugi (256)': lighten_color(base_colors['Mugi'], 0.3),
        #'Carat (64)': base_colors['Carat'],
        'Carat (128)': base_colors['Carat'],  # you can adjust by lightening if desired
        'Carat (256)': lighten_color(base_colors['Carat'], 0.3),
        #'SA (8)': base_colors['SA'],
        'SA (16)': lighten_color(base_colors['SA'], 0.3),
        #'SA Figna (8)': "crimson",
        'SA-F (16)': lighten_color("crimson", 0.3),
        #'SIMD (8)': base_colors['SIMD'],
        'SD (16)': lighten_color(base_colors['SD'], 0.3),
        #'SIMD Figna (8)': "darkorange",
        'SD-F (16)': lighten_color("darkorange", 0.3),
    }

    # Define figure size and bar width
    fig, axes = plt.subplots(3, 2, figsize=(fig_width, fig_height))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    bar_width = 0.107

    # Plotting loop
    for i, (metric, categories_data) in enumerate(data.items()):
        for j, (category, model_values) in enumerate(categories_data.items()):
            if j == 1:
                x = np.arange(len(x_labels))
                x_lab = x_labels
            else:
                x = np.arange(len(x_labels) - 1)
                x_lab = x_labels[:-1]
            ax = axes[i, j]
            for k, (model, values) in enumerate(model_values.items()):
                ax.bar(x + k * bar_width, values, bar_width, label=model, color=colors.get(model, 'black'))
            if i == 0:  # Only set titles for the top row
                category = 'proj/ffn' if category == 'proj' else category
                ax.set_title(f"{category}", fontsize=font_title)
            if i < 2:  # Keep ticks but hide labels for top and middle rows
                ax.set_xticks(x + bar_width * (len(model_values) // 2))
                ax.set_xticklabels([])
            else:
                ax.set_xticks(x + bar_width * (len(model_values) // 2))
                ax.set_xticklabels(x_lab, fontsize=font_tick)
            ax.tick_params(axis='y', labelsize=font_title, width=0.5)  # Thinner ticks
            ax.tick_params(axis='x', width=0.5)  # Thinner ticks

            # Add "x" after each y-axis value as an integer
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))

            ax.spines['top'].set_linewidth(0.5)  # Thinner top border
            ax.spines['right'].set_linewidth(0.5)  # Thinner right border
            ax.spines['left'].set_linewidth(0.5)  # Thinner left border
            ax.spines['bottom'].set_linewidth(0.5)  # Thinner bottom border
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            if i == 0 and j == 0:
                ax.legend(fontsize=5, ncol=4, loc='lower center', bbox_to_anchor=(1.115, 1.25), frameon=True)
            if j == 0:
                ax.set_ylabel('Norm ' + metric, fontsize=font_label)

    plt.savefig(output_path + "gemm_breakdown.pdf", dpi=1200, bbox_inches="tight")
    plt.show()