"""Array-SHAPE query for fig_6: the aspect-ratio sweep at each model's maximum context.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/array_shape_performance_metrics.csv.

WHAT THIS ADDS TO fig_1. fig_1 walks the DIAGONAL of the array design space -- 32x32,
64x64, ... 512x512 -- so every point doubles the reduction depth and the output width
together, and the figure can only say how much array helps, never which array. This query
keeps the full 5x5 cross product of `pe.instance`, so the two sides can be read apart:
rows are the weight-stationary reduction depth (`array_m`, the K dimension and the cycles
a weight load costs) and columns are the output width (`array_n`, the N dimension). Shapes
on a shared anti-diagonal hold the SAME number of PEs, which is what makes an iso-area
comparison possible at all.

SCOPE, and why it is one slice rather than three. description.py generates the off-diagonal
shapes only at the 10 MiB reference SRAM and only at each model's MAXIMUM context --
131072 for the Llamas, 1048576 for DeepSeek. That is exactly fig_1's and fig_3's 'mixed'
slice, so fig_6's diagonal reproduces the third panel of both, and the short-context slices
simply do not exist off the diagonal. The filters below therefore mirror fig_1_query's
except that the square-array test is dropped and the max_seq_len slice is picked per model
instead of being emitted three ways.

BOTH FREQUENCIES ARE KEPT, unlike every other array query, which slices to 1000 MHz. That
needs care, because the three metrics below do not respond to frequency alike:

  cycle_count        `llama_array`, the COMPUTE-ONLY view. Exactly frequency-INVARIANT:
                     frequency enters the model only through the DRAM lane's
                     bytes-per-cycle (mapping.py), which this view excludes. The 2000 MHz
                     rows reproduce the 1000 MHz ones digit for digit -- that is correct,
                     not a bug, and it is why fig_6.py plots the 1000 MHz slice alone.
  llama_cycle_count  `llama`, the full view. A 2 GHz part gets half the DRAM bytes per
                     cycle, so its stall cycles grow; this is NOT comparable across
                     frequencies as a raw number.
  runtime_ms         `llama` in wall-clock. The only column that compares two nodes
                     honestly.
  average_bandwidth  required DRAM bandwidth in GB/s: total bytes over the total on-chip
                     window they must hide under. Frequency-proportional, since the cycle
                     counts behind it do not move. NOTE this is a DEMAND, not a speed --
                     a shape can score high by being fast OR by having worse reuse and
                     moving more bytes, so its maximum is the most memory-hungry design,
                     not the best one.
  peak_bandwidth     the same, pooled per phase and the hungrier phase taken. See
                     fig_4_query for why the peak is per phase and not per GEMM.
  throughput_tokens_per_s
                     generated tokens per second: batch_size * (max_seq_len -
                     prefill_seq_len) / runtime. The work-normalised speed metric -- it
                     credits a bigger batch for the extra work it does rather than
                     penalising it for the extra time -- and the one fig_4's `throughput`
                     variant ranks on.
  runtime_ms_per_sequence
                     runtime_ms / batch_size. The same ranking as throughput, inverted.
                     Every other metric here scales with batch -- prefill runs
                     M = batch_size * prefill_seq_len rows and decode runs M = batch_size
                     (model.py) -- so a batch-32 point does a sixteenth of the work of a
                     batch-512 one and wins any raw comparison by default. Dividing by the
                     batch turns the column into time per sequence, which is the same
                     quantity at every batch and is what 'this configuration is faster'
                     has to mean when batch is one of the axes being chosen.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import (query_cycle_count, query_execution_time,
                                            gemm_demand, bandwidth_summary)
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
import pandas as pd
from tqdm import tqdm
import os

logger.remove()

BASE_SRAM_BITS = 10 * 2**23  # the reference capacity every non-sweep configuration carries

# Each model's own maximum context, the point this figure reports. Same mapping fig_1's
# and fig_3's 'mixed' slice uses.
MAX_SEQ_LEN = {
    'llama_3_1_8b': 131072,
    'llama_3_1_70b': 131072,
    'llama_3_1_405b': 131072,
    'deepseek_v4': 1048576,
}

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

        # NO SQUARE FILTER -- the off-diagonal shapes are the point of this query.
        # NO FREQUENCY FILTER either: frequency is a column here, not a slice. See the
        # module docstring for which metric may be compared across it.
        frequency = architecture_dict['pe']['query']['frequency']

        # Capacity is fig_2's axis, not this one: hold every SRAM at the reference size so
        # a shape's cycles are not confounded by how much memory it was given.
        if any(sram != BASE_SRAM_BITS for sram in memory_sizes(architecture_dict)):
            continue

        workload_name = workload_dict['name']
        batch_size = workload_dict['configuration']['batch_size']
        max_seq_len = workload_dict['configuration']['max_seq_len']
        prefill_seq_len = workload_dict['configuration']['prefill_seq_len']

        if max_seq_len != MAX_SEQ_LEN.get(workload_name):
            continue

        cycle_count = query_cycle_count(
            event_graph=event_graph, metric_dict=metric_dict,
            workload=workload_name, event='llama_array')
        llama_cycle_count = query_cycle_count(
            event_graph=event_graph, metric_dict=metric_dict,
            workload=workload_name, event='llama')
        # query_execution_time divides the ms metric by 1e3, so this is seconds despite
        # the metric's declared unit; converted back so the column name is honest.
        runtime_ms = query_execution_time(
            event_graph=event_graph, metric_dict=metric_dict,
            workload=workload_name, event='llama') * 1e3

        # The same computation fig_4 reports, run across the whole grid so fig_4's design
        # point can be selected on it rather than transcribed. Shared implementation in
        # utils.py, so the two can never disagree about what the number means.
        bandwidth = bandwidth_summary(
            gemm_demand(event_graph, metric_dict, workload_name), frequency)

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'array_m': array_dim[0],
            'array_n': array_dim[1],
            'pe_count': array_dim[0] * array_dim[1],
            'batch_size': batch_size,
            'frequency': frequency,
            'max_seq_len': max_seq_len,
            'cycle_count': cycle_count,
            'llama_cycle_count': llama_cycle_count,
            'runtime_ms': runtime_ms,
            'runtime_ms_per_sequence': runtime_ms / batch_size,
            # the decode walk generates one token per step from prefill_seq_len to
            # max_seq_len, for every sequence in the batch
            'throughput_tokens_per_s': (batch_size * (max_seq_len - prefill_seq_len)
                                        / (runtime_ms / 1e3)),
            'total_data_moved': bandwidth['total_bytes'],
            'window_cycle_count': bandwidth['total_window'],
            'average_bandwidth': bandwidth['average'],
            'peak_bandwidth': bandwidth['peak'],
            'peak_phase': bandwidth['peak_phase'],
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

if not array_query_df.empty:
    array_query_df = array_query_df.sort_values(
        by=['model', 'frequency', 'array_m', 'array_n', 'batch_size'])

    out_name = 'array_shape_performance_metrics'
    array_query_df.to_csv(output_path + f'{out_name}.csv', index=False)

    df_sci = array_query_df.copy()
    for column in ['cycle_count', 'llama_cycle_count', 'runtime_ms',
                   'runtime_ms_per_sequence', 'throughput_tokens_per_s',
                   'total_data_moved', 'window_cycle_count']:
        df_sci[column] = df_sci[column].apply(lambda x: f'{x:.3e}')
    for column in ['average_bandwidth', 'peak_bandwidth']:
        df_sci[column] = df_sci[column].apply(lambda x: f'{x:.1f}')
    df_sci.to_csv(output_path + f'{out_name}_scientific.csv', index=False)

    # A missing shape would silently leave a blank cell in fig_6's grid, which reads as
    # "no benefit" rather than "not simulated". Say so here instead.
    for (model, frequency, batch_size), group in array_query_df.groupby(
            ['model', 'frequency', 'batch_size']):
        if len(group) != 25:
            print(f'Warning: {model} at {frequency} MHz batch {batch_size} has '
                  f'{len(group)} of 25 array shapes; fig_6 will draw an incomplete grid.')

    # The compute view cannot move with frequency (see the docstring). If it ever does,
    # something has started reading frequency that should not, and every ranking built on
    # this CSV is suspect -- so it is checked here rather than assumed.
    keys = ['model', 'array_m', 'array_n', 'batch_size']
    pivot = array_query_df.pivot_table(index=keys, columns='frequency', values='cycle_count')
    if {1000, 2000}.issubset(pivot.columns):
        drift = (pivot[2000] - pivot[1000]).abs().max()
        if drift > 0:
            print(f'Warning: llama_array cycle_count differs between 1000 and 2000 MHz '
                  f'by up to {drift:.3e}; it is expected to be frequency-invariant.')
else:
    print("Warning: No matching configurations found. CSVs not saved.")
