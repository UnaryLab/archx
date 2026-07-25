import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.llama.query.utils import query_execution_time
from chiplet4ai.llama import llama_model
from archx.metric import aggregate_event_count
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

def transfer_window_cycles(data_moved, bandwidth_gib_per_second, frequency_mhz):
    if data_moved <= 0 or bandwidth_gib_per_second <= 0:
        return 0
    return data_moved / (bandwidth_gib_per_second * 2**30) * float(frequency_mhz) * 1e6

def bandwidth_from_window_cycles(data_moved, window_cycles, frequency_mhz):
    if data_moved <= 0 or window_cycles <= 0:
        return 0
    seconds = window_cycles / (float(frequency_mhz) * 1e6)
    return (data_moved / seconds) / 2**30

def summarize_bandwidth(samples, frequency_mhz):
    data_moved = sum(sample['data_moved'] for sample in samples)
    transfer_window = sum(sample['transfer_window_cycles'] for sample in samples)
    active_avg = bandwidth_from_window_cycles(data_moved, transfer_window, frequency_mhz)
    peak = max((sample['bandwidth'] for sample in samples if sample['data_moved'] > 0), default=0)
    return {
        'data_moved': data_moved,
        'transfer_window_cycles': transfer_window,
        'active_avg_bandwidth': active_avg,
        'peak_required_bandwidth': peak,
    }

def collect_mapping_samples(event_graph, architecture_dict, workload_dict, workload_name, event_suffix, movement_specs, frequency_mhz):
    samples = {name: [] for name in movement_specs}

    for event in sorted(event_graph.get_all_node_names()):
        if not event.endswith(event_suffix) or not hasattr(llama_model, event):
            continue

        event_count = aggregate_event_count(
            event_graph=event_graph,
            workload=workload_name,
            event=event
        )
        if event_count <= 0:
            continue

        performance = getattr(llama_model, event)(architecture_dict, workload_dict)
        subevents = performance.get('subevent', {})

        for name, spec in movement_specs.items():
            subevent = subevents.get(spec['event'])
            if not subevent:
                continue

            count = float(subevent.get('count', 0)) * event_count
            bandwidth = float(subevent.get('factor', {}).get('bandwidth', 0))
            data_moved = count * spec.get('count_to_bytes', 1)
            transfer_window = transfer_window_cycles(data_moved, bandwidth, frequency_mhz)

            samples[name].append({
                'event': event,
                'data_moved': data_moved,
                'transfer_window_cycles': transfer_window,
                'bandwidth': bandwidth,
            })

    return {
        name: summarize_bandwidth(event_samples, frequency_mhz)
        for name, event_samples in samples.items()
    }

tag='onchip'
output_path = 'zoo/chiplet4ai/llama/results/'
runs_path = f'zoo/chiplet4ai/llama/description/configurations.csv'
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

        array_execution_time = query_execution_time(
            event_graph=event_graph,
            metric_dict=metric_dict,
            workload=workload_name,
            event='llama_array'
        )
        frequency_mhz = architecture_dict['dram'].get('query', {}).get('frequency', 1000)
        dram_summary = collect_mapping_samples(
            event_graph=event_graph,
            architecture_dict=architecture_dict,
            workload_dict=workload_dict,
            workload_name=workload_name,
            event_suffix='_dram',
            movement_specs={
                'input': {'event': 'dram_input_read'},
                'weight': {'event': 'dram_weight_read'},
            },
            frequency_mhz=frequency_mhz,
        )
        sram_summary = collect_mapping_samples(
            event_graph=event_graph,
            architecture_dict=architecture_dict,
            workload_dict=workload_dict,
            workload_name=workload_name,
            event_suffix='_sram',
            movement_specs={
                'input': {'event': 'sram_input_write_mapping', 'count_to_bytes': isram_width / 8},
                'weight': {'event': 'sram_weight_write_mapping', 'count_to_bytes': wsram_width / 8},
            },
            frequency_mhz=frequency_mhz,
        )

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'asram_size': isram_bank * isram_depth * isram_width,
            'wsram_size': wsram_bank * wsram_depth * wsram_width,
            'array_execution_time': array_execution_time,
            'input_data_moved': dram_summary['input']['data_moved'],
            'weight_data_moved': dram_summary['weight']['data_moved'],
            'input_transfer_window_cycles': dram_summary['input']['transfer_window_cycles'],
            'weight_transfer_window_cycles': dram_summary['weight']['transfer_window_cycles'],
            'input_sram_window_cycles': sram_summary['input']['transfer_window_cycles'],
            'weight_sram_window_cycles': sram_summary['weight']['transfer_window_cycles'],
            'input_dram_bandwidth': dram_summary['input']['active_avg_bandwidth'],
            'weight_dram_bandwidth': dram_summary['weight']['active_avg_bandwidth'],
            'input_sram_bandwidth': sram_summary['input']['active_avg_bandwidth'],
            'weight_sram_bandwidth': sram_summary['weight']['active_avg_bandwidth'],
            'input_active_avg_required_bandwidth': dram_summary['input']['active_avg_bandwidth'],
            'weight_active_avg_required_bandwidth': dram_summary['weight']['active_avg_bandwidth'],
            'input_peak_required_bandwidth': dram_summary['input']['peak_required_bandwidth'],
            'weight_peak_required_bandwidth': dram_summary['weight']['peak_required_bandwidth'],
            'input_required_bandwidth': dram_summary['input']['active_avg_bandwidth'],
            'weight_required_bandwidth': dram_summary['weight']['active_avg_bandwidth']
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    array_query_df = array_query_df.sort_values(by=['model', 'array_dim', 'batch_size', 'asram_size', 'wsram_size'])
    array_query_df.to_csv(output_path + f'bandwidth_performance_metrics.csv', index=False)

    if not array_query_df.empty:
        array_query_df_sci = array_query_df.copy()
        for col in ['asram_size', 'wsram_size', 'array_execution_time', 'input_data_moved', 'weight_data_moved', 'input_transfer_window_cycles', 'weight_transfer_window_cycles', 'input_sram_window_cycles', 'weight_sram_window_cycles', 'input_dram_bandwidth', 'weight_dram_bandwidth', 'input_sram_bandwidth', 'weight_sram_bandwidth', 'input_active_avg_required_bandwidth', 'weight_active_avg_required_bandwidth', 'input_peak_required_bandwidth', 'weight_peak_required_bandwidth', 'input_required_bandwidth', 'weight_required_bandwidth']:
            array_query_df_sci[col] = array_query_df_sci[col].apply(lambda x: f'{x:.3e}')
        array_query_df_sci = array_query_df_sci.sort_values(by=['model', 'array_dim', 'batch_size', 'asram_size', 'wsram_size'])
        array_query_df_sci.to_csv(output_path + f'bandwidth_performance_metrics_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. Scientific notation CSV not saved.")
