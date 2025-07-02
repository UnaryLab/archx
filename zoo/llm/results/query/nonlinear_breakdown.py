from zoo.llm.results.query.utils import query_performance_nonlinear_metrics, compute_throughput_efficiancy, load_yaml, geomean
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def query(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']

    baseline_list = ['systolic']
    baseline_arch_dim_list = ['16x16']
    baseline_subarch_list = ['mac', 'pwl', 'taylor']

    mugi_throughput_module = 'magnitude_register'
    baseline_throughput_module = 'accumulator_vector'
    approximate_throughtput_module = 'adder_vector'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b']
    max_seq_len_list = ['max_seq_len_512', 'max_seq_len_2048', 'max_seq_len_4096']
    batch_size = 'batch_size_8'
    network = 'single_node'

    nonlinear_breakdown_df = pd.DataFrame()

    for arch in vlp_list + baseline_list:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else ['']):
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
    data_df = pd.read_csv(input_path + 'nonlinear_breakdown_norm.csv')
    data_dict = {'XTicks': ['512', '2048', '4096']}

    for arch in ['Mugi', 'Carat', 'VA']:
        arch_key = arch
        for arch_dim in ['16', '128', '256']:
            if arch == 'VA' and arch_dim != '16':
                continue
            if (arch == 'Mugi' or arch == 'Carat') and arch_dim == '16':
                continue

            arch_label = 'mugi' if arch == 'Mugi' else 'carat' if arch == 'Carat' else 'systolic'

            dim_label = arch_dim + 'x8' if arch in ['Mugi', 'Carat'] else arch_dim + 'x' + arch_dim
            dim_key = ' (' + arch_dim + ')'
            for subarch in ['', 'PWL', 'Taylor']:
                for function in ['softmax', 'silu']:

                    function_key = 'SM ' if function == 'softmax' else 'SiLU '

                    subarch = np.nan if subarch == '' and arch != 'VA' else 'mac' if subarch == '' and arch == 'VA' else subarch
                    if subarch == 'Taylor' and function != 'softmax':
                        continue
                    if arch != 'VA' and (subarch == 'PWL' or subarch == 'Taylor'):
                        continue
                    subarch_label = subarch.lower() if subarch is not np.nan else subarch
                    subarch_key = ' ' + subarch if subarch is not np.nan else ''
                    subarch_key = '' if subarch_key == ' mac' else subarch_key

                    values = data_df[
                        (data_df['arch'] == arch_label) &
                        (data_df['function'] == function) &
                        (data_df['arch_dim'] == dim_label) &
                        (pd.isna(data_df['subarch']) if pd.isna(subarch_label) else data_df['subarch'] == subarch_label)
                    ]

                    key = function_key + arch_key + subarch_key + dim_key

                    data_dict[key] = {
                        'NormThroughput': values['throughput'].values,
                        'NormEnergyEfficiency': values['energy_efficiency'].values,
                        'NormPowerEfficiency': values['power_efficiency'].values
                    }

    data = {
        'XTicks': ['512', '2048', '4096'],

        # Mugi 64
        #'SM Mugi (64)': {
        #    'NormThroughput':       [49.43110781, 46.33105959, 45.6795269],
        #    'NormEnergyEfficiency': [34865.88722, 1951.589918, 475.0551014],
        #    'NormPowerEfficiency':  [705.3430271, 663.4403085, 654.5595454]
        #},
        #'SiLU Mugi (64)': {
        #    'NormThroughput':       [42.1875, 44.296875, 44.6484375],
        #    'NormEnergyEfficiency': [27855.24928, 7310.389714, 3684.068873],
        #    'NormPowerEfficiency':  [619.0055396, 649.812419, 654.9455774]
        #},

        # Mugi 128
        'SM Mugi (128)': {
            'NormThroughput':       [109.4615295, 111.5448376, 112.0151202],
            'NormEnergyEfficiency': [147127.2667, 7138.332043, 1686.472441],
            'NormPowerEfficiency':  [1344.100227, 1177.107382, 1144.093518]
        },
        'SiLU Mugi (128)': {
            'NormThroughput':       [87.01304833, 87.890625, 88.9453125],
            'NormEnergyEfficiency': [92412.38141, 24881.23102, 12588.69388],
            'NormPowerEfficiency':  [1026.804238, 1105.83249, 1118.995011]
        },

        # Mugi 256
        'SM Mugi (256)': {
            'NormThroughput':       [206.2848781, 200.2963601, 197.1853818],
            'NormEnergyEfficiency': [389385.9496, 23392.62003, 5313.260388],
            'NormPowerEfficiency':  [1887.612671, 1839.46416, 1752.956044]
        },
        'SiLU Mugi (256)': {
            'NormThroughput':       [160.3125, 175.078125, 177.5390625],
            'NormEnergyEfficiency': [269747.2475, 73574.28717, 37298.00195],
            'NormPowerEfficiency':  [1498.59582, 1634.984159, 1657.688975]
        },

        # Carat 64
        #'SM Carat (64)': {
        #    'NormThroughput':       [8, 8, 8],
        #    'NormEnergyEfficiency': [820.6805356, 52.1061051, 13.0390913],
        #    'NormPowerEfficiency':  [102.5850669, 102.5850667, 102.5850586]
        #},
        #'SiLU Carat (64)': {
        #    'NormThroughput':       [8, 8, 8],
        #    'NormEnergyEfficiency': [842.8780316, 210.7195079, 105.359754],
        #    'NormPowerEfficiency':  [105.359754, 105.359754, 105.359754]
        #},

        # Carat 128
        'SM Carat (128)': {
            'NormThroughput':       [15.99820854, 15.9994962, 15.99974378],
            'NormEnergyEfficiency': [2504.162156, 159.0181563, 39.79411298],
            'NormPowerEfficiency':  [156.5276606, 156.5402154, 156.5426107]
        },
        'SiLU Carat (128)': {
            'NormThroughput':       [16, 16, 16],
            'NormEnergyEfficiency': [2571.503541, 642.8758853, 321.4379426],
            'NormPowerEfficiency':  [160.7189713, 160.7189713, 160.7189713]
        },

        # Carat 256
        'SM Carat (256)': {
            'NormThroughput':       [31.98925608, 31.99697761, 31.99846281],
            'NormEnergyEfficiency': [6302.282943, 400.3331523, 100.1890986],
            'NormPowerEfficiency':  [197.0124884, 197.0598373, 197.068915]
        },
        'SiLU Carat (256)': {
            'NormThroughput':       [32, 32, 32],
            'NormEnergyEfficiency': [6472.828794, 1618.207199, 809.1035993],
            'NormPowerEfficiency':  [202.2758998, 202.2758998, 202.2758998]
        },

        # Vector Array (non-PWL)
        #'SM VA (8)': {
        #    'NormThroughput':       [1, 1, 1],
        #    'NormEnergyEfficiency': [1, 0.063491332, 0.015888144],
        #    'NormPowerEfficiency':  [1, 0.9999999, 0.99999988]
        #},
        #'SiLU VA (8)': {
        #    'NormThroughput':       [1, 1, 1],
        #    'NormEnergyEfficiency': [1, 0.25, 0.125],
        #    'NormPowerEfficiency':  [1, 1, 1]
        #},
        'SM VA (16)': {
            'NormThroughput':       [2, 2, 2],
            'NormEnergyEfficiency': [3.783924177, 0.240246353, 0.060119524],
            'NormPowerEfficiency':  [1.891962088, 1.891961647, 1.891961557]
        },
        'SiLU VA (16)': {
            'NormThroughput':       [2, 2, 2],
            'NormEnergyEfficiency': [3.788803663, 0.947200916, 0.473600458],
            'NormPowerEfficiency':  [1.894401831, 1.894401831, 1.894401831]
        },

        # Vector Array PWL
        #'SM VA (8) PWL': {
        #    'NormThroughput':       [11.25, 11.25, 11.25],
        #    'NormEnergyEfficiency': [126.0801257, 8.004986878, 2.00317704],
        #    'NormPowerEfficiency':  [11.20712229, 11.2071097, 11.20710867]
        #},
        #'SiLU VA (8) PWL': {
        #    'NormThroughput':       [11.25, 11.25, 11.25],
        #    'NormEnergyEfficiency': [129.5883266, 32.39705797, 16.19852701],
        #    'NormPowerEfficiency':  [11.51896237, 11.51895395, 11.51895254]
        #},
        'SM VA (16) PWL': {
            'NormThroughput':       [22.5, 22.5, 22.5],
            'NormEnergyEfficiency': [477.0790058, 30.29031938, 7.579883414],
            'NormPowerEfficiency':  [21.20351137, 21.20346587, 21.20346215]
        },
        'SiLU VA (16) PWL': {
            'NormThroughput':       [22.5, 22.5, 22.5],
            'NormEnergyEfficiency': [490.372891, 122.5930516, 61.29651152],
            'NormPowerEfficiency':  [21.79435071, 21.79432028, 21.79431521]
        },
        # 'SM VA (128) PWL': {
        #     'NormThroughput':       [179.9798461, 179.9943323, 179.9971176],
        #     'NormEnergyEfficiency': [27088.49926, 1720.131475, 430.4602815],
        #     'NormPowerEfficiency':  [150.5085144, 150.5179634, 150.5200336]
        # },
        # 'SiLU VA (128) PWL': {
        #     'NormThroughput':       [180, 180, 180],
        #     'NormEnergyEfficiency': [27863.34358, 6965.766821, 3482.877655],
        #     'NormPowerEfficiency':  [154.7963532, 154.7948182, 154.7945624]
        # },
        # 'SM VA (256) PWL': {
        #     'NormThroughput':       [359.8791309, 359.9659981, 359.9827066],
        #     'NormEnergyEfficiency': [86704.4477, 5507.434816, 1378.308712],
        #     'NormPowerEfficiency':  [240.9265786, 240.9757868, 240.9859012]
        # },
        # 'SiLU VA (256) PWL': {
        #     'NormThroughput':       [360, 360, 360],
        #     'NormEnergyEfficiency': [89254.33642, 22313.22974, 11156.58534],
        #     'NormPowerEfficiency':  [247.9287123, 247.9247749, 247.9241187]
        # },

        # Vector Array Taylor
        #'SM VA (8) Taylor': {
        #    'NormThroughput':       [5, 5, 5],
        #    'NormEnergyEfficiency': [25.00271223, 1.587454856, 0.397246533],
        #    'NormPowerEfficiency':  [5.000542446, 5.000539939, 5.000539735]
        #},
        #'SiLU VA (8) Taylor': {
        #    'NormThroughput':       [2, 2, 2],
        #    'NormEnergyEfficiency': [2.002680809, 0.500670202, 0.250335101],
        #    'NormPowerEfficiency':  [2.002680809, 2.002680808, 2.002680808]
        #},
        'SM VA (16) Taylor': {
            'NormThroughput':       [10, 10, 10],
            'NormEnergyEfficiency': [94.94724174, 6.028321619, 1.508534093],
            'NormPowerEfficiency':  [9.494724174, 9.494715051, 9.494714305]
        },
        # 'SiLU VA (16) Taylor': {
        #     'NormThroughput':       [4, 4, 4],
        #     'NormEnergyEfficiency': [7.622730916, 1.905682727, 0.952841364],
        #     'NormPowerEfficiency':  [3.811365458, 3.811365455, 3.811365454]
        # },
        # 'SM VA (128) Taylor': {
        #     'NormThroughput':       [79.99104272, 79.99748102, 79.99871892],
        #     'NormEnergyEfficiency': [5830.742165, 370.2576335, 92.65647137],
        #     'NormPowerEfficiency':  [72.89243853, 72.89759995, 72.89865177]
        # },
        # 'SiLU VA (128) Taylor': {
        #     'NormThroughput':       [23.27272727, 29.25714286, 30.56716418],
        #     'NormEnergyEfficiency': [225.3988034, 86.0309229, 46.60732192],
        #     'NormPowerEfficiency':  [19.37020967, 23.52408048, 24.39602007]
        # },
        # 'SM VA (256) Taylor': {
        #     'NormThroughput':       [159.9462804, 159.984888, 159.992314],
        #     'NormEnergyEfficiency': [21680.79417, 1377.161605, 344.6528404],
        #     'NormPowerEfficiency':  [135.5504743, 135.5787, 135.5843345]
        # },
        # 'SiLU VA (256) Taylor': {
        #     'NormThroughput':       [34.13333333, 52.51282051, 57.69014085],
        #     'NormEnergyEfficiency': [449.2771774, 241.6653924, 142.1904314],
        #     'NormPowerEfficiency':  [26.32483461, 36.81621212, 39.43562747]
        # }
    }

    data = data_dict

    # -------------------------------
    # 2) FIGURE AND FONT SETTINGS
    # -------------------------------
    fig_width_pt = 240  # ACM single-column width in points
    fig_width = fig_width_pt / 72  # inches
    fig_height = fig_width * 0.7  # Adjusted height for readability

    font_title = 6.5
    font_tick = 5

    # -------------------------------
    # 3) UNIQUE COLOR MAP FOR EACH INPUT SOURCE
    # -------------------------------

    base_colors = {
        'Mugi': 'forestgreen',
        'Carat': 'rebeccapurple',
        'VA': 'dodgerblue',
        'PWL': 'orange',
        'Taylor': 'red',
    }

    # Function to lighten a color
    # Create specific colors for each model and size, including Carat

    softmax_color = {
        'SM Mugi (128)': base_colors['Mugi'],
        'SiLU Mugi (128)': lighten_color(base_colors['Mugi'], 0.2),
        'SM Carat (128)': base_colors['Carat'],
        'SiLU Carat (128)': lighten_color(base_colors['Carat'], 0.2),
        'SM VA (16)': base_colors['VA'],
        'SM VA PWL (16)': base_colors['PWL'],
        'SM VA Taylor (16)': base_colors['Taylor'],
        }

    silu_color = {
        'SM Mugi (256)': lighten_color(base_colors['Mugi'], 0.4),
        'SiLU Mugi (256)': lighten_color(base_colors['Mugi'], 0.6),
        'SM Carat (256)': lighten_color(base_colors['Carat'], 0.4),
        'SiLU Carat (256)': lighten_color(base_colors['Carat'], 0.6),
        'SiLU VA (16)': lighten_color(base_colors['VA'], 0.4),
        'SiLU VA PWL (16)': lighten_color(base_colors['PWL'], 0.4),
        'SiLU VA Taylor (16)': base_colors['Taylor']
    }

    colors = {**softmax_color, **silu_color}

    class StringObjectHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            patch = plt.Rectangle([0, 0], 1, 1, facecolor=orig_handle, edgecolor='none')
            handlebox.add_artist(patch)
            return patch

    # -------------------------------
    # 4) CREATE SUBPLOTS AND PLOT AS GROUPED BARS
    # -------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True)

    x_labels = data['XTicks']
    N = len(x_labels)
    # Get all categories (skip 'XTicks')
    categories = [key for key in data.keys() if key != 'XTicks']
    num_categories = len(categories)
    x = np.arange(N)
    bar_width = 0.9 / num_categories

    for i, category in enumerate(categories):
        throughput = data[category]['NormThroughput']
        energy_eff = data[category]['NormEnergyEfficiency']
        power_eff  = data[category]['NormPowerEfficiency']
        color = colors.get(category, 'black')

        ax1.bar(x + i*bar_width, throughput, width=bar_width, color=color, label=category)
        ax2.bar(x + i*bar_width, energy_eff, width=bar_width, color=color, label=category)
        ax3.bar(x + i*bar_width, power_eff,  width=bar_width, color=color, label=category)

    # -------------------------------
    # 5) FORMAT SUBPLOTS, SET LOG SCALE, AND ADD LEGEND
    # -------------------------------
    for ax, title, ylabel in zip(
        [ax1, ax2, ax3],
        ['Norm Throughput', 'Norm Energy Efficiency', 'Norm Power Efficiency'],
        ['Norm Thr', 'Norm Energy Eff', 'Norm Pwr Eff']
    ):
        ax.set_title(title, fontsize=font_title)
        # ax.set_ylabel(ylabel, fontsize=font_tick)  # Removed y-axis label
        ax.set_xticks(x + (num_categories - 1) * bar_width / 2)
        ax.set_xticklabels(x_labels, fontsize=font_tick)
        ax.tick_params(axis='y', labelsize=font_tick)
        ax.minorticks_off()
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}x'))
        # Manually set y-axis label position
        # ax.yaxis.set_label_coords(-0.1, 0.5)  # Removed y-axis label positioning

    # Add the legend to the figure instead of individual axes
    handles, labels = ax1.get_legend_handles_labels()  # Get from first axes
    rows = -(-len(labels) // 4)
    new_order = [i + j * rows for i in range(rows) for j in range(4) if i + j * rows < len(labels)]

    # Create the legend for the entire figure
    fig.legend([handles[i] for i in new_order], 
            [labels[i] for i in new_order], 
            ncol=4, 
            fontsize=4.5, 
            loc='upper center', 
            bbox_to_anchor=(0.535, 1.2), 
            frameon=True, 
            columnspacing=1.25)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path + 'nonlinear_breakdown.pdf', dpi=1200, bbox_inches='tight')
    plt.show()