import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chiplet4ai.llama.query.utils import query_performance_metrics
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict

import pandas as pd

runs_path = 'chiplet4ai/llama/description/configurations.csv'
output_path = 'chiplet4ai/llama/results/'

array_query_df = pd.DataFrame()

with open(runs_path, 'r') as f:
    runs_df = pd.read_csv(f)
    array_dim_set = set()
    for index, row in runs_df.iterrows():
        run_path = row['run_path']
        run_arch_path = run_path + '/architecture.yaml'
        run_workload_path = run_path + '/workload.yaml'
        run_event_graph_path = run_path + '/checkpoint.gt'
        run_metric_path = run_path + '/metric.yaml'

        architecture_dict = load_architecture_dict(run_arch_path)
        workload_dict = load_workload_dict(run_workload_path)
        event_graph = load_event_graph(run_event_graph_path)
        metric_dict = load_metric_dict(run_metric_path)

        array_dim = architecture_dict['multiplier']['instance']
        
        depth = architecture_dict['ififo']['query']['depth']

        for key, value in workload_dict.items():

            batch_size = value['configuration']['batch_size']

            if array_dim[0] == array_dim[1] and depth == 16:
                array_dim_set.add(tuple(array_dim))
                metrics = query_performance_metrics(
                    event_graph = event_graph,
                    metric_dict = metric_dict,
                    module = 'multiplier',
                    event = key,
                    tag='onchip'
                )

                array_query_row = {
                    'array_dim': f'{array_dim[0]}x{array_dim[1]}',
                    'workload': key,
                    'batch_size': batch_size,
                    'flops': metrics['flops'],
                    'execution_time_s': metrics['execution_time'],
                    'energy_j': metrics['energy'],
                    'power_w': metrics['power'],
                    'cycle_count': metrics['cycle_count'],
                    'throughput': metrics['throughput']
                }

                array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    array_query_df.to_csv(output_path + 'array_performance_metrics.csv', index=False)
    
    # Save a second CSV with scientific notation
    if not array_query_df.empty:
        array_query_df_sci = array_query_df.copy()
        for col in ['flops', 'execution_time_s', 'energy_j', 'power_w', 'cycle_count', 'throughput']:
            array_query_df_sci[col] = array_query_df_sci[col].apply(lambda x: f'{x:.3e}')
        array_query_df_sci.to_csv(output_path + 'array_performance_metrics_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. Scientific notation CSV not saved.")