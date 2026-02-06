import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.llama.query.utils import query_performance_metrics, query_cycle_count
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict

import pandas as pd

logger.remove()

#configs = ['8b', '70b']
configs = ['8b']

output_path = 'chiplet4ai/llama/results/'

array_query_8b_df = pd.DataFrame()
array_query_70b_df = pd.DataFrame()

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
            chiplet_size = architecture_dict['ififo']['query']['depth']
            fifo_dim = architecture_dict['ififo']['instance']
            technology = architecture_dict['multiplier']['query']['technology']
            depth = architecture_dict['ififo']['query']['depth']

            isram_bank = architecture_dict['isram']['query']['bank']
            isram_width = architecture_dict['isram']['query']['width']
            isram_depth = architecture_dict['isram']['query']['depth']

            for key, value in workload_dict.items():
                
                batch_size = value['configuration']['batch_size']
                memory_size = value['configuration']['memory_size_mb']
                bitwidth = value['configuration']['activation_bitwidth']
                if (index % 100) == 0:
                    print(index)

                if array_dim == [32, 32] and technology == 7 and isram_bank == 32:
                
                    compute_cycle_count = query_cycle_count(
                        event_graph = event_graph,
                        metric_dict = metric_dict,
                        workload = key,
                        event = 'array_mapping'
                    )

                    input_memory_cycle_count = query_cycle_count(
                        event_graph = event_graph,
                        metric_dict = metric_dict,
                        workload = key,
                        event = 'input_offchip_writes'
                    )

                    #print(compute_cycle_count, input_memory_cycle_count, memory_size)
                    compare = compute_cycle_count > input_memory_cycle_count
                    print(f'Workload: {key}, Compute Cycles: {compute_cycle_count}, Input Memory Cycles: {input_memory_cycle_count}, Memory Size (MB): {memory_size}, Compute Bound: {compare}')
                    if compare:
                        print('compute bound')
                    else:
                        print('memory bound')

                    array_query_row = {
                        'technology_nm': technology,
                        'array_dim': f'{array_dim[0]}x{array_dim[1]}',
                        'chiplet_size': f'{depth}x{depth}',
                        'workload': key,
                        'batch_size': batch_size,
                        # 'flops': metrics['flops'],
                        # 'execution_time_s': metrics['execution_time'],
                        # 'energy_j': metrics['energy'],
                        # 'power_w': metrics['power'],
                        # 'cycle_count': metrics['cycle_count'],
                        # 'throughput': metrics['throughput']
                    }


                    if config == '8b':
                        array_query_8b_df = pd.concat([array_query_8b_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_8b_df.empty else pd.DataFrame([array_query_row])
                    else:
                        array_query_70b_df = pd.concat([array_query_70b_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_70b_df.empty else pd.DataFrame([array_query_row])

        if config == '8b':
            array_query_8b_df.to_csv(output_path + f'mem_performance_metrics_{config}.csv', index=False)

            if not array_query_8b_df.empty:
                array_query_8b_df_sci = array_query_8b_df.copy()
                for col in ['flops', 'execution_time_s', 'energy_j', 'power_w', 'cycle_count', 'throughput']:
                    array_query_8b_df_sci[col] = array_query_8b_df_sci[col].apply(lambda x: f'{x:.3e}')
                array_query_8b_df_sci.to_csv(output_path + f'mem_performance_metrics_{config}_scientific.csv', index=False)
            else:
                print("Warning: No matching configurations found. Scientific notation CSV not saved.")
        else:
            array_query_70b_df.to_csv(output_path + f'mem_performance_metrics_{config}.csv', index=False)
        
            # Save a second CSV with scientific notation
            if not array_query_70b_df.empty:
                array_query_70b_df_sci = array_query_70b_df.copy()
                for col in ['flops', 'execution_time_s', 'energy_j', 'power_w', 'cycle_count', 'throughput']:
                    array_query_70b_df_sci[col] = array_query_70b_df_sci[col].apply(lambda x: f'{x:.3e}')
                array_query_70b_df_sci.to_csv(output_path + f'mem_performance_metrics_{config}_scientific.csv', index=False)
            else:
                print("Warning: No matching configurations found. Scientific notation CSV not saved.")