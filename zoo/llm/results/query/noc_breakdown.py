from zoo.llm.results.query.utils import query_performance_metrics, compute_throughput_efficiancy, load_yaml, geomean
import pandas as pd
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np

def query(input_path, output_path):

    network_list = ['single_node', 'multi_node_4x4', 'multi_node_8x8']

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['64x8', '128x8', '256x8']
    vlp_node_stationary = ''

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16', '64x64']
    baseline_subarch_list = ['mac', 'figna']
    baseline_node_stationary = 'node_stationary_ws'

    throughput_module = OrderedDict({
        'mugi': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'and_gate'}),
        'carat': OrderedDict({'gemm': 'and_gate', 'nonlinear': 'register_vector'}),
        'systolic': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'}),
        'simd': OrderedDict({'gemm': 'multiplier', 'nonlinear': 'register_vector'})
    })

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'

    noc_breakdown_df = pd.DataFrame()

    for network in network_list:
        for arch in (vlp_list + baseline_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else ['']):
                for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list):
                    if network == 'single_node' and arch_dim != '64x64':
                        continue
                    if network != 'single_node' and arch_dim == '64x64':
                        continue
                    noc_breakdown_model_list = []
                    for model in model_list:
                        gemm_module = throughput_module[arch]['gemm']
                        nonlinear_module = throughput_module[arch]['nonlinear']
                        
                        termination_path = 'full_termination' if arch == 'mugi' else ''
                        run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
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
                            'network': network,
                            'arch_dim': arch_dim,
                            'throughput': throughput_eff_dict['throughput'],
                            'energy_efficiency': throughput_eff_dict['energy_efficiency'],
                            'power_efficiency': throughput_eff_dict['power_efficiency'],
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
    data_dict = {'XTicks':['64/8/8', '128/16/16', '256/S-U/S-U'],
                'Designs':{}}

    for arch in ['Mugi', 'Carat', 'SA', 'SA-F', 'SD', 'SD-F']:
        arch_label = arch.lower() if arch in ['Mugi', 'Carat'] else 'systolic' if arch in ['SA', 'SA-F'] else 'simd'
        subarch_label = np.nan if arch in ['Mugi', 'Carat'] else 'mac' if arch in ['SA', 'SD'] else 'figna'

        for noc_dim in ['4x4', '8x8']:
            noc_dim_label = 'multi_node_4x4' if noc_dim == '4x4' else 'multi_node_8x8'
            data_dict['Designs'][arch + ' ' + noc_dim] = {
                'NormThroughput': [],
                'NormEnergyEfficiency': [],
                'NormPowerEfficiency': []
            }

            for node_dim in data_dict['XTicks']:
                xtick_split = node_dim.split('/')
                node_dim_value = xtick_split[0] if arch_label in ['mugi', 'carat'] else xtick_split[1]
                node_dim_label = node_dim_value + 'x8' if arch_label in ['mugi', 'carat'] else '64x64' if node_dim_value == 'S-U' else node_dim_value + 'x' + node_dim_value
                noc_dim_label = 'single_node' if node_dim_label == '64x64' else noc_dim_label
                filter_df = df[
                    (df['arch'] == arch_label)
                    & (pd.isna(df['subarch']) if pd.isna(subarch_label) else df['subarch'] == subarch_label)
                    & (df['network'] == noc_dim_label)
                    & (df['arch_dim'] == node_dim_label)
                ]

                data_dict['Designs'][arch + ' ' + noc_dim]['NormThroughput'] += filter_df['throughput'].tolist()
                data_dict['Designs'][arch + ' ' + noc_dim]['NormEnergyEfficiency'] += filter_df['energy_efficiency'].tolist()
                data_dict['Designs'][arch + ' ' + noc_dim]['NormPowerEfficiency'] += filter_df['power_efficiency'].tolist()
        
    data = data_dict

    fig_width_pt = 240          # ACM single-column width in points
    fig_width = fig_width_pt / 72  # Convert to inches
    fig_height = fig_width/1.2  # Adjusted height for readability

    font_title = 6
    font_tick  = 5

    colors_map = {
        'Mugi': "green",
        'Carat': "orange",
        'SA': "purple",
        'SA-F': "darkgray",
        'SD': "red",
        'SD-F': "dodgerblue"
    }

    XTicks = data['XTicks']             # X-axis category labels
    x = np.arange(len(XTicks))           # Evenly spaced positions for the categories
    designs = list(data['Designs'].keys())
    n_designs = len(designs)
    total_width = 0.8                   # Total width allocated for each x-category
    bar_width = total_width / n_designs  # Individual bar width

    design_to_group = {}
    group_to_designs = {}
    for design in designs:
        if "Carat" in design:
            group = "Carat"
        elif '-F' in design:
            if 'SA' in design:
                group = 'SA-F'
            else:
                group = 'SD-F'
        elif 'SA' in design:
            group = 'SA'
        elif 'SD' in design:
            group = 'SD'
        else:
            group = 'Mugi'
        design_to_group[design] = group
        group_to_designs.setdefault(group, []).append(design)

    # Compute an index for each design within its group (to assign different shades).
    design_index_in_group = {}
    for group, ds in group_to_designs.items():
        # For consistency, sort the designs (e.g. "Mugi 4x4" comes before "Mugi 8x8")
        sorted_ds = sorted(ds)
        for idx, d in enumerate(sorted_ds):
            design_index_in_group[d] = idx

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(fig_width, fig_height), sharex=True, gridspec_kw={'hspace': 0.45, 'top': 0.93, 'bottom': 0.2}
    )  # Reduced hspace to 0.2

    # Plot each design as a set of bars (one per x-category) in each subplot.
    for i, design in enumerate(designs):
        group = design_to_group[design]
        total_in_group = len(group_to_designs[group])
        index_in_group = design_index_in_group[design]
        base_color = colors_map[group]
        shaded_color = get_shaded_color_by_index(base_color, index_in_group, total_in_group)

        # Compute x-offset for this design's bars
        offset = (i - (n_designs - 1) / 2) * bar_width

        # Plot Norm Throughput
        ax1.bar(x + offset, data['Designs'][design]['NormThroughput'],
                width=bar_width, label=design, color=shaded_color, alpha=0.8)
        # Plot Norm Energy Efficiency
        ax2.bar(x + offset, data['Designs'][design]['NormEnergyEfficiency'],
                width=bar_width, label=design, color=shaded_color, alpha=0.8)
        # Plot Norm Power Efficiency
        ax3.bar(x + offset, data['Designs'][design]['NormPowerEfficiency'],
                width=bar_width, label=design, color=shaded_color, alpha=0.8)


    for ax, title, ylabel in zip(
        [ax1, ax2, ax3],
        ['Norm Throughput', 'Norm Energy Efficiency', 'Norm Power Efficiency'],
        ['Norm Thr', 'Norm Energy Eff', 'Norm Pwr Eff']
    ):
        ax.set_title(title, fontsize=font_title, pad=4)  # Added pad=4 to move titles down
        # ax.set_ylabel(ylabel, fontsize=font_label)  # Removed y-axis label
        ax.set_xticks(x)
        ax.set_xticklabels(XTicks, fontsize=font_tick)
        ax.tick_params(axis='y', labelsize=font_tick)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Manually set y-axis label position
        # ax.yaxis.set_label_coords(-0.1, 0.5)  # Removed manual y-axis label positioning

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))

    handles, labels = ax1.get_legend_handles_labels()
    num_entries = len(labels)
    max_rows = 6  # Maximum number of rows for the legend
    ncol = int(np.ceil(num_entries / max_rows))
    ax.yaxis.set_label_coords(-0.1, 0.5)

    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=6,
        #nrow=2,
        bbox_to_anchor=(0.5, 1.1),
        fontsize=4
    )

    plt.savefig(output_path + "noc_breakdown.pdf", dpi=1200, bbox_inches='tight')
    plt.show()
