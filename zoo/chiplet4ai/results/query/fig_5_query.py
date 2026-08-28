"""Stall-free array-utilization query for fig_5, the companion to fig_3.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/array_utilization_nostall_metrics_seqlen_*.csv.

HOW THIS DIFFERS FROM fig_3. Both divide the same numerator -- useful MACs -- by PE-slots.
They differ in which cycles count as "the array was running":

  fig_3  max(array_input, array_weight, array_compute)   what the array actually spends
  fig_5      array_compute alone                         what it spends COMPUTING

A GEMM charges those three lanes in parallel, so the engine takes the largest. Whenever
the weight lane is the largest the array is stalled on a weight load: a Kt x Nt tile costs
array_m cycles to shift in and is then used for only Mt cycles of streaming. That is a
real cost and fig_3 is right to carry it -- but it is a DATAFLOW cost, and it buries the
question this figure asks, which is how well the GEMM's shape maps onto the PE grid at
all.

So fig_5 is pure MAPPING efficiency. Its denominator is the compute lane's cycles, and
what remains in it is exactly the tiling loss:

    utilization = k_utilization * n_utilization * m_utilization

  k  the reduction failing to fill the array's rows (head dim 128 leaves three quarters
     of a 512-row array idle)
  n  the output failing to fill its columns
  m  a partial edge tile in the streamed dimension

Read the two together: fig_5 is the ceiling this mapping could reach if weight loading
were free, and the gap between fig_5 and fig_3 is what weight loading costs.

WHERE THE CYCLES COME FROM. The per-lane cycle count is the edge's own `count * factor`,
which the engine stores in the checkpoint. mapping.py writes each lane's count already
scaled to useful work and puts the true cycles in the factor, so the product is the lane's
cycle count and the ratio is its utilization. The event-graph object exposes counts but
not factors, so the factors are read from the checkpoint JSON directly.

SCOPE: mirrors fig_3_query -- square arrays only, all three SRAMs at the 10 MiB reference
size, the 1000 MHz slice, and the same three max_seq_len slices.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from archx.metric import aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
import pandas as pd
from tqdm import tqdm
import os

logger.remove()

def warn(message):
    print(f'WARNING [fig_5_query]: {message}', file=sys.stderr)

def memory_sizes(architecture_dict):
    return [architecture_dict[sram]['query']['width'] * architecture_dict[sram]['query']['depth']
            * architecture_dict[sram]['query']['bank']
            for sram in ('isram', 'wsram', 'osram')]

def lane_cycles(checkpoint_path):
    """(source, target) -> that edge's cycle count, i.e. its own count * cycle factor."""
    with open(checkpoint_path, 'r') as handle:
        checkpoint = json.load(handle)
    cycles = {}
    for edge in checkpoint['edges']:
        factor = edge.get('factor') or {}
        cycles[(edge['source'], edge['target'])] = edge['count'] * factor.get('cycle_count', 1.0)
    return cycles

def compute_only(event_graph, edge_cycles, workload_name):
    """Useful MACs and the COMPUTE cycles they run in, accumulated per GEMM.

    Both the with-load span and the compute-only span are returned, so the caller can
    report the gap between them rather than leaving the reader to infer it.
    """
    macs = 0.0
    compute_cycles = 0.0
    with_load_cycles = 0.0

    for name in sorted(event_graph.get_all_node_names()):
        if not name.endswith('_arr'):
            continue
        gemm = name[:-len('_arr')]

        # Multiplicity comes off '_dram', which hangs under `llama` alone. '_arr' is
        # reachable through the `llama_array` view as well, and aggregate_event_count sums
        # over every path, so reading it there would double every GEMM.
        multiplicity = aggregate_event_count(
            event_graph=event_graph, workload=workload_name, event=f'{gemm}_dram')
        if multiplicity <= 0:
            continue

        # one pe event per useful MAC: array_compute's count is already utilization
        # scaled, and array.py charges it array_m * array_n pe events
        macs += (event_graph.get_edge_count(name, 'array_compute')
                 * event_graph.get_edge_count('array_compute', 'pe')) * multiplicity

        lanes = {child: edge_cycles.get((name, child), 0.0)
                 for child in ('array_input', 'array_weight', 'array_compute')}
        compute_cycles += lanes['array_compute'] * multiplicity
        # the three lanes are parallel, so the GEMM's own span is their maximum
        with_load_cycles += max(lanes.values()) * multiplicity

    return macs, compute_cycles, with_load_cycles

output_path = 'zoo/chiplet4ai/results/csv/'
runs_path = 'zoo/chiplet4ai/designs/llama/description/configurations.csv'
array_query_df = pd.DataFrame()

# Per-run warnings go to stderr and get lost under the tqdm bar, so the invariant is also
# tallied and reported once at the end, where it cannot be scrolled past.
impossible = []

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(runs_path, 'r') as f:
    runs_df = pd.read_csv(f)
    for index, row in tqdm(runs_df.iterrows(), total=len(runs_df)):
        run_path = row['run_path']

        architecture_dict = load_architecture_dict(run_path + '/architecture.yaml')
        array_dim = architecture_dict['pe']['instance']

        if array_dim[0] != array_dim[1]:
            continue
        # Utilization is a ratio of two cycle counts and so frequency invariant, but the
        # slice is kept so this figure covers exactly the configurations fig_3 does.
        if architecture_dict['pe']['query']['frequency'] != 1000:
            continue
        if any(size != 10 * 2**23 for size in memory_sizes(architecture_dict)):
            continue

        workload_dict = load_workload_dict(run_path + '/workload.yaml')
        event_graph = load_event_graph(run_path + '/checkpoint.json')

        workload_name = workload_dict['name']
        batch_size = workload_dict['configuration']['batch_size']
        max_seq_len = workload_dict['configuration']['max_seq_len']

        pe_count = array_dim[0] * array_dim[1]
        macs, compute_cycles, with_load_cycles = compute_only(
            event_graph=event_graph,
            edge_cycles=lane_cycles(run_path + '/checkpoint.json'),
            workload_name=workload_name)

        utilization = macs / (pe_count * compute_cycles) if compute_cycles > 0 else 0
        with_load = macs / (pe_count * with_load_cycles) if with_load_cycles > 0 else 0

        # Above 1 is not a tight design point, it is a broken measurement: more useful
        # MACs than the array has slots to do them in.
        if utilization > 1 + 1e-9:
            warn(f'{run_path}: utilization {utilization:.4f} > 1 -- more useful MACs '
                 f'({macs:.6g}) than PE slots ({pe_count * compute_cycles:.6g})')
            impossible.append((run_path, utilization))

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'max_seq_len': max_seq_len,
            'pe_count': pe_count,
            'useful_macs': macs,
            'compute_cycle_count': compute_cycles,
            # fig_3's span, carried alongside so the cost of weight loading is one
            # division away instead of a join against another CSV
            'with_load_cycle_count': with_load_cycles,
            'utilization': utilization,
            'utilization_with_load': with_load,
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    if not array_query_df.empty:
        array_query_df = array_query_df.sort_values(by=['model', 'array_dim', 'batch_size', 'max_seq_len'])

        # Three max_seq_len slices, the same ones fig_1 and fig_3 use.
        llama_models = ['llama_3_1_8b', 'llama_3_1_70b', 'llama_3_1_405b']
        slices = {
            'array_utilization_nostall_metrics_seqlen_4096': array_query_df['max_seq_len'] == 4096,
            'array_utilization_nostall_metrics_seqlen_131072': array_query_df['max_seq_len'] == 131072,
            'array_utilization_nostall_metrics_seqlen_mixed': (
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
            for col in ['useful_macs', 'compute_cycle_count', 'with_load_cycle_count']:
                df_slice_sci[col] = df_slice_sci[col].apply(lambda x: f'{x:.3e}')
            for col in ['utilization', 'utilization_with_load']:
                df_slice_sci[col] = df_slice_sci[col].apply(lambda x: f'{x:.4f}')
            df_slice_sci.to_csv(output_path + f'{out_name}_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. CSVs not saved.")

if impossible:
    print(f'\nfig_5_query: {len(impossible)} run(s) with utilization > 1.')
    for run_path, utilization in impossible[:5]:
        print(f'  {utilization:.4f}  {run_path}')
else:
    print('\nfig_5_query: no utilization exceeds 1.')
