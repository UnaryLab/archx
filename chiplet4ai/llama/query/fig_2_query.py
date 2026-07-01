import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.llama.query.utils import query_bandwidth, aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from archx.utils import read_yaml
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

tag='onchip'
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
        run_event_path = run_path + '/event.yaml'
        run_metric_path = run_path + '/metric.yaml'

        architecture_dict = load_architecture_dict(run_arch_path)
        workload_dict = load_workload_dict(run_workload_path)
        event_graph = load_event_graph(run_event_graph_path)
        metric_dict = load_metric_dict(run_metric_path)
        event_dict = read_yaml(run_event_path)

        array_dim = architecture_dict['pe']['instance']
        technology = architecture_dict['pe']['query']['technology']
        depth = architecture_dict['ififo']['query']['depth']

        isram_bank = architecture_dict['isram']['query']['bank']
        isram_depth = architecture_dict['isram']['query']['depth']
        isram_width = architecture_dict['isram']['query']['width']

        wsram_bank = architecture_dict['wsram']['query']['bank']
        wsram_depth = architecture_dict['wsram']['query']['depth']
        wsram_width = architecture_dict['wsram']['query']['width']

        if array_dim != [512, 512]:
            continue

        
        batch_size = workload_dict['configuration']['batch_size']
        workload_name = workload_dict['name']

        if batch_size != 512:
            continue

        isram_bandwidth, isram_execution_time, isram_data_moved = query_bandwidth(event_graph, metric_dict, workload_name, 'sram_input_write_mapping', 'dram_input_read', 64)
        wsram_bandwidth, wsram_execution_time, wsram_data_moved = query_bandwidth(event_graph, metric_dict, workload_name, 'sram_weight_write_mapping', 'dram_weight_read', 64)

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'asram_size': isram_bank * isram_depth * isram_width,
            'wsram_size': wsram_bank * wsram_depth * wsram_width,
            'isram_execution_time': isram_execution_time,
            'wsram_execution_time': wsram_execution_time,
            'isram_data_moved': isram_data_moved,
            'wsram_data_moved': wsram_data_moved,
            'asram_bandwidth': isram_bandwidth,
            'wsram_bandwidth': wsram_bandwidth
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    array_query_df = array_query_df.sort_values(by=['model', 'array_dim', 'batch_size', 'asram_size', 'wsram_size'])
    array_query_df.to_csv(output_path + f'bandwidth_performance_metrics.csv', index=False)

    if not array_query_df.empty:
        array_query_df_sci = array_query_df.copy()
        for col in ['asram_size', 'wsram_size', 'isram_execution_time', 'wsram_execution_time', 'isram_data_moved', 'wsram_data_moved', 'asram_bandwidth', 'wsram_bandwidth']:
            array_query_df_sci[col] = array_query_df_sci[col].apply(lambda x: f'{x:.3e}')
        array_query_df_sci = array_query_df_sci.sort_values(by=['model', 'array_dim', 'batch_size', 'asram_size', 'wsram_size'])
        array_query_df_sci.to_csv(output_path + f'bandwidth_performance_metrics_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. Scientific notation CSV not saved.")