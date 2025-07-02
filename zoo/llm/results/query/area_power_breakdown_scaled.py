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
    n_unique_columns = pd.unique(df[['arch', 'subarch', 'network']].values.ravel()).size
    n_unique_arch = pd.unique(df[['arch', 'subarch']].values.ravel()).size
    n_rows = n_columns // n_unique_columns

    n_rows = 2
    n_columns = 6
    total = 12

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(fig_width, fig_height),
                             gridspec_kw={"wspace": -0.3, "hspace": 0.2, 'left': -0.65, 'right': 0.65})
    
    for index, row in df.iterrows():
        ax_row = index % n_rows
        ax_column = index // n_rows

        
        arch = row['arch']
        subarch = row['subarch']
        arch_dim = row['arch_dim']
        network_dim = row['network']
        total = row['total']
        values = row.drop(labels=['arch', 'subarch', 'arch_dim', 'network', 'total']).to_list()
    

        arch = arch.capitalize()
        arch = 'SA' if arch == 'Systolic' else 'SD' if arch == 'Simd' else arch
        subarch = '' if subarch == 'mac' else '-F' if subarch == 'figna' else '-L' if subarch == 'lut' else ''
        network_dim = network_dim.split('_')[-1] if network_dim != 'single_node' else '1'
        arch_dim = '(' + arch_dim.split('x')[0] + ',' + network_dim + ')' 
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
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.legend(
        labels, loc='lower center', ncol=7, fontsize=5, bbox_to_anchor=(-1.74, -0.15)
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()

def figure(input_path: str, output_path: str):

    area_df = pd.read_csv(input_path + 'array_area_breakdown_scaled.csv')
    power_df = pd.read_csv(input_path + 'array_power_breakdown_scaled.csv')

    labels = ['Array', 'SRAM', 'NoC']
    array_colors = ['#A8D5BA','#85E3FF', '#f09494']

    breakdown(df=area_df, array_colors=array_colors, labels=labels, output_path=output_path + 'area_breakdown_scaled.pdf', value_unit='mm²')
    breakdown(df=power_df, array_colors=array_colors, labels=labels, output_path=output_path + 'power_breakdown_scaled.pdf', value_unit='W')

def query(input_path, output_path):

    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['8x8', '16x16']
    baseline_subarch_list = ['figna']

    
    model = 'llama_2_7b'
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network_list = ['multi_node_4x4', 'single_node', 'multi_node_2x1']
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
                        try:
                            tag_area = query_area(tag=tag, event_graph=event_graph, metric_dict=metric_dict, workload=model)
                            tag_power = query_tag_power(tag=tag, event_graph=event_graph, metric_dict=metric_dict, workload=model, event=model) / 1000
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
        
        area_df.to_csv(output_path + 'array_area_breakdown_scaled.csv', index=False)
        power_df.to_csv(output_path + 'array_power_breakdown_scaled.csv', index=False)