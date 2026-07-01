import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.llama.query.utils import query_performance_metrics, query_cycle_count
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict, query_module_metric
import pandas as pd
from tqdm import tqdm
import os

logger.remove()

def memory_sizes(architecture_dict):
    srams = []
    for sram in ['isram', 'wsram', 'osram']:
        sram_query = architecture_dict[sram]['query']
        sram_size = sram_query['width'] * sram_query['depth'] * sram_query['bank']
        srams.append(sram_size)
    return srams

output_path = 'chiplet4ai/llama/results/'
runs_path = f'chiplet4ai/llama/description/configurations.csv'
array_query_df = pd.DataFrame()

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(runs_path, 'r') as f:
    runs_df = pd.read_csv(f)
    for index, row in tqdm(runs_df.iterrows(), total=len(runs_df)):
        run_path = row['run_path']
        run_arch_path = run_path + '/architecture.yaml'
        run_workload_path = run_path + '/workload.yaml'
        run_event_graph_path = run_path + '/checkpoint.json'
        run_metric_path = run_path + '/metric.yaml'

        architecture_dict = load_architecture_dict(run_arch_path)
        workload_dict = load_workload_dict(run_workload_path)
        event_graph = load_event_graph(run_event_graph_path)
        metric_dict = load_metric_dict(run_metric_path)

        array_dim = architecture_dict['pe']['instance']

        if array_dim[0] != array_dim[1]:
            continue

        srams = memory_sizes(architecture_dict)

        flag = False
        for sram in srams:
            if sram != 10 * 2**23:
                flag = True
        if flag:
            continue

        workload_name = workload_dict['name']
        batch_size = workload_dict['configuration']['batch_size']

        metrics = query_performance_metrics(
            event_graph = event_graph,
            metric_dict = metric_dict,
            workload = workload_name,
            event = 'llama_array',
            module = 'pe',
            tag='onchip',
            metrics=['cycle_count', 'execution_time', 'flops', 'throughput']
        )

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'flops': metrics['flops'],
            'execution_time_s': metrics['execution_time'],
            'cycle_count': metrics['cycle_count'],
            'throughput': metrics['throughput']
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

        array_query_df = array_query_df.sort_values(by=['model', 'array_dim', 'batch_size'])
        array_query_df.to_csv(output_path + f'array_performance_metrics.csv', index=False)

    if not array_query_df.empty:
        array_query_df_sci = array_query_df.copy()
        for col in ['flops', 'execution_time_s', 'cycle_count', 'throughput']:
            array_query_df_sci[col] = array_query_df_sci[col].apply(lambda x: f'{x:.3e}')
        array_query_df_sci = array_query_df_sci.sort_values(by=['model', 'array_dim', 'batch_size'])
        array_query_df_sci.to_csv(output_path + f'array_performance_metrics_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. Scientific notation CSV not saved.")