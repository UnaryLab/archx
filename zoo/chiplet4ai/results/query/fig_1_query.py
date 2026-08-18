import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import query_cycle_count
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
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

output_path = 'zoo/chiplet4ai/results/csv/'
runs_path = f'zoo/chiplet4ai/designs/llama/description/configurations.csv'
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

        # The design space sweeps frequency (1000/2000 MHz); stall cycles depend on
        # DRAM bytes-per-cycle and therefore on frequency, so the two halves are distinct
        # design points. This figure shows the 1000 MHz reference slice.

        if architecture_dict['pe']['query']['frequency'] != 1000:
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
        max_seq_len = workload_dict['configuration']['max_seq_len']

        cycle_count = query_cycle_count(
            event_graph=event_graph,
            metric_dict=metric_dict,
            workload=workload_name,
            event='llama_array'
        )

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'max_seq_len': max_seq_len,
            'cycle_count': cycle_count
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    if not array_query_df.empty:
        array_query_df = array_query_df.sort_values(by=['model', 'array_dim', 'batch_size', 'max_seq_len'])

        # Three max_seq_len slices: all models at 4096, all models at 131072,
        # and a mixed slice (llama at its 131072 point, deepseek at its 1048576
        # point) for comparing them at their respective long-context settings.
        llama_models = ['llama_3_1_8b', 'llama_3_1_70b', 'llama_3_1_405b']
        slices = {
            'array_performance_metrics_seqlen_4096': array_query_df['max_seq_len'] == 4096,
            'array_performance_metrics_seqlen_131072': array_query_df['max_seq_len'] == 131072,
            'array_performance_metrics_seqlen_mixed': (
                (array_query_df['model'].isin(llama_models) & (array_query_df['max_seq_len'] == 131072))
                | ((array_query_df['model'] == 'deepseek_v4') & (array_query_df['max_seq_len'] == 1048576))
            ),
        }

        for out_name, mask in slices.items():
            df_slice = array_query_df[mask].drop(columns=['max_seq_len'])
            if df_slice.empty:
                print(f"Warning: No matching configurations found for '{out_name}'. CSV not saved.")
                continue

            df_slice.to_csv(output_path + f'{out_name}.csv', index=False)

            df_slice_sci = df_slice.copy()
            df_slice_sci['cycle_count'] = df_slice_sci['cycle_count'].apply(lambda x: f'{x:.3e}')
            df_slice_sci.to_csv(output_path + f'{out_name}_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. CSVs not saved.")
