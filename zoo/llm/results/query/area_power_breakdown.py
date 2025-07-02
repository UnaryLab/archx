from archx.metric import aggregate_event_metric, aggregate_tag_metric, query_module_metric
from zoo.llm.results.query.utils import query_area, query_tag_power, load_yaml, query_dynamic_energy
import matplotlib.pyplot as plt
import pandas as pd
import os

def breakdown(df, array_colors, labels, output_path, value_unit):
    fig_width_pt = 240  # ACM single-column width in points
    fig_width = fig_width_pt / 72  # Convert to inches
    fig_height = fig_width * 0.65  # Given height

    n_columns = len(df)
    n_unique_columns = (df['arch'].nunique())
    n_rows = n_columns // n_unique_columns

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_unique_columns, figsize=(fig_width, fig_height),
                             gridspec_kw={"wspace": -0.3, "hspace": 0.2, 'left': -0.65, 'right': 0.65})
    
    for index, row in df.iterrows():
        ax_row = index % n_rows
        ax_column = index // n_rows
        
        arch = row['arch']
        subarch = row['subarch']
        arch_dim = row['arch_dim']
        total = row['total']
        values = row.drop(labels=['arch', 'subarch', 'arch_dim', 'total']).to_list()

        arch = arch.capitalize()
        arch = 'SA' if arch == 'Systolic' else 'SD' if arch == 'Simd' else arch
        subarch = '' if subarch == 'mac' else '-F' if subarch == 'figna' else ''
        arch_dim = '(' + arch_dim.split('x')[0] + ')'

        ax = axes[ax_row, ax_column]

        ax.pie(
            values,
            labels=None,
            autopct=lambda p: f'{p:.0f}%' if p > 7 else '',
            startangle=90,
            colors=array_colors,
            textprops={'fontsize': 5}
        )
        ax.set_title(f"{arch}{subarch} {arch_dim}\n({total:.2f} {value_unit})", fontsize=5, pad=-2)
        ax.axis('off')

    plt.legend(
        labels, loc='lower center', ncol=7, fontsize=5, bbox_to_anchor=(-1.45, -0.2)
    )
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()

def figure(input_path: str, output_path: str):

    area_df = pd.read_csv(input_path + 'array_area_breakdown.csv')
    power_df = pd.read_csv(input_path + 'array_power_breakdown.csv')

    labels = ['Fifo', 'PE', 'TC', 'Value Reuse', 'Acc', 'Nonlinear', 'Vector']
    array_colors = ['#D5AAFF', '#A8D5BA', '#FFDDC1', '#85E3FF', '#f09494', '#FFC3A0', "#a7f672"]

    breakdown(df=area_df, array_colors=array_colors, labels=labels, output_path=output_path + 'area_breakdown.pdf', value_unit='mm²')
    breakdown(df=power_df, array_colors=array_colors, labels=labels, output_path=output_path + 'power_breakdown.pdf', value_unit='mW')

def query(input_path, output_path):

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16']
    baseline_subarch_list = ['figna']

    network = 'single_node'
    model = 'llama_2_7b'
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    
    tag_list = ['fifo', 'pe', 'tc', 'value_reuse', 'accumulator', 'nonlinear', 'vector']
    df_labels = ['arch', 'subarch', 'arch_dim', 'total'] + tag_list
    area_df = pd.DataFrame(columns=df_labels)
    power_df = pd.DataFrame(columns=df_labels)

    for arch in vlp_list + baseline_list:
        for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list):
            for subarch in (baseline_subarch_list if arch in baseline_list else ['']):

                termination_path = 'full_termination' if arch == 'mugi' else ''
                run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                yaml_dict = load_yaml(run_path)

                event_graph = yaml_dict['event_graph']
                metric_dict = yaml_dict['metric_dict']

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
    
    area_df.to_csv(output_path + 'array_area_breakdown.csv', index=False)
    power_df.to_csv(output_path + 'array_power_breakdown.csv', index=False)