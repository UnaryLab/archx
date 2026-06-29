import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.llama.query.utils import query_performance_metrics
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict

import pandas as pd

# logger.remove()

# configs = ['8b', '70b']
configs = ['8b']


output_path = 'chiplet4ai/llama/results/'

array_query_8b_df = pd.DataFrame()
array_query_70b_df = pd.DataFrame()

test_set = set()

for config in configs:
    runs_path = f'chiplet4ai/llama/description_{config}/configurations.csv'
    with open(runs_path, 'r') as f:
        runs_df = pd.read_csv(f)
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

            for key, value in workload_dict.items():

                batch_size = value['configuration']['batch_size']
                technology = architecture_dict['multiplier']['query']['technology']
                depth = architecture_dict['ififo']['query']['depth']

                # if (batch_size, array_dim[0]) not in test_set:
                #     test_set.add((batch_size, array_dim[0]))

                metrics = query_performance_metrics(
                    event_graph = event_graph,
                    metric_dict = metric_dict,
                    module = 'multiplier',
                    event = 'av_decode',
                    tag='onchip'
                )


                array_query_row = {
                    'array_dim': f'{array_dim[0]}x{array_dim[1]}',
                    'batch_size': batch_size,
                    'technology_nm': technology,
                    'chiplet_size': f'{depth}x{depth}',
                    'flops': metrics['flops'],
                    'execution_time_s': metrics['execution_time'],
                    'cycle_count': metrics['cycle_count'],
                    'throughput': metrics['throughput']
                }


                if config == '8b':
                    array_query_8b_df = pd.concat([array_query_8b_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_8b_df.empty else pd.DataFrame([array_query_row])
                else:
                    array_query_70b_df = pd.concat([array_query_70b_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_70b_df.empty else pd.DataFrame([array_query_row])

        if config == '8b':
            # Sort by array_dim and batch_size before saving
            array_query_8b_df = array_query_8b_df.sort_values(by=['array_dim', 'batch_size'])
            array_query_8b_df.to_csv(output_path + f'gemm_performance_metrics_{config}.csv', index=False)

            if not array_query_8b_df.empty:
                array_query_8b_df_sci = array_query_8b_df.copy()
                for col in ['flops', 'execution_time_s', 'cycle_count', 'throughput']:
                    array_query_8b_df_sci[col] = array_query_8b_df_sci[col].apply(lambda x: f'{x:.3e}')
                array_query_8b_df_sci = array_query_8b_df_sci.sort_values(by=['array_dim', 'batch_size'])
                array_query_8b_df_sci.to_csv(output_path + f'gemm_performance_metrics_{config}_scientific.csv', index=False)
            else:
                print("Warning: No matching configurations found. Scientific notation CSV not saved.")
        else:
            array_query_70b_df = array_query_70b_df.sort_values(by=['array_dim', 'batch_size'])
            array_query_70b_df.to_csv(output_path + f'gemm_performance_metrics_{config}.csv', index=False)
        
            # Save a second CSV with scientific notation
            if not array_query_70b_df.empty:
                array_query_70b_df_sci = array_query_70b_df.copy()
                for col in ['flops', 'execution_time_s', 'cycle_count', 'throughput']:
                    array_query_70b_df_sci[col] = array_query_70b_df_sci[col].apply(lambda x: f'{x:.3e}')
                array_query_70b_df_sci = array_query_70b_df_sci.sort_values(by=['array_dim', 'batch_size'])
                array_query_70b_df_sci.to_csv(output_path + f'gemm_performance_metrics_{config}_scientific.csv', index=False)
            else:
                print("Warning: No matching configurations found. Scientific notation CSV not saved.")