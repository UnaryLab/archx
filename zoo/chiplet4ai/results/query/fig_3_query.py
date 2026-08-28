"""Array-utilization query for fig_3, the companion to fig_1.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/array_utilization_metrics_seqlen_*.csv.

WHAT THE NUMBER MEANS: useful MACs divided by the PE slots the array offers while it is
running -- `M*K*N / (array_m * array_n * compute_cycles)`. It is a MAPPING efficiency,
not a system one: the denominator is the `llama_array` compute span fig_1 plots, so
memory stalls never enter and the two figures are read against the same cycles. What
drives it down is tiling loss -- a partial edge tile, a GEMM smaller than the array, and
above all the reduction dimension failing to fill the array's rows (every Llama here has
head dim 128, so the attention GEMMs leave half a 256-row array idle and three quarters
of a 512-row one).

WHERE THE NUMERATOR COMES FROM: mapping.py scales every subevent count by its
utilization, so the `pe` module's event count IS the useful MAC count -- one event per
real multiply-accumulate, padding excluded. It is accumulated per GEMM rather than by
aggregating the `pe` module directly, because `<gemm>_arr` is reachable twice (once under
`llama`, once under the `llama_array` view) and a module-level aggregate would count
every MAC twice.

SCOPE: mirrors fig_1_query -- square arrays only, all three SRAMs at the 10 MiB reference
size, the 1000 MHz slice, and the same three max_seq_len slices.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import query_cycle_count
from archx.metric import aggregate_event_count
from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
import pandas as pd
from tqdm import tqdm
import os

logger.remove()

def warn(message):
    print(f'WARNING [fig_3_query]: {message}', file=sys.stderr)

def memory_sizes(architecture_dict):
    srams = []
    for sram in ['isram', 'wsram', 'osram']:
        sram_query = architecture_dict[sram]['query']
        sram_size = sram_query['width'] * sram_query['depth'] * sram_query['bank']
        srams.append(sram_size)
    return srams

def gemm_events(event_graph):
    return sorted(name[:-len('_arr')] for name in event_graph.get_all_node_names()
                  if name.endswith('_arr'))

def useful_macs_and_cycles(event_graph, metric_dict, workload_name):
    """Useful MACs and the compute span they run in, accumulated per GEMM."""
    macs = 0.0
    cycles = 0.0

    for gemm in gemm_events(event_graph):
        # Multiplicity comes off '_dram', which hangs under `llama` alone. '_arr' is
        # reachable through `llama` AND through the `llama_array` view, and
        # aggregate_event_count sums over every path, so reading it there would double
        # every GEMM. The GEMM charges its three children alike, so the '_dram'
        # multiplicity is the GEMM's.
        multiplicity = aggregate_event_count(
            event_graph=event_graph, workload=workload_name, event=f'{gemm}_dram')
        if multiplicity <= 0:
            continue

        # one pe event per useful MAC: array_compute's count is already utilization
        # scaled, and array.py charges it array_m * array_n pe events
        pe_per_instance = (event_graph.get_edge_count(f'{gemm}_arr', 'array_compute')
                           * event_graph.get_edge_count('array_compute', 'pe'))
        macs += pe_per_instance * multiplicity
        # workload=None is the value of ONE instance; the multiplicity scales it
        cycles += query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                    workload=None, event=f'{gemm}_arr') * multiplicity

    return macs, cycles

output_path = 'zoo/chiplet4ai/results/csv/'
runs_path = f'zoo/chiplet4ai/designs/llama/description/configurations.csv'
array_query_df = pd.DataFrame()

# Per-run warnings go to stderr and get lost under the tqdm bar, so both invariants are
# also tallied and reported once at the end, where they cannot be scrolled past.
view_mismatches = []
impossible = []

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

        # The design space sweeps frequency (1000/2000 MHz). Utilization is a ratio of
        # two cycle counts and is frequency-invariant, but the slice is kept so this
        # figure covers exactly the configurations fig_1 does.
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

        pe_count = array_dim[0] * array_dim[1]
        macs, gemm_cycles = useful_macs_and_cycles(
            event_graph=event_graph, metric_dict=metric_dict, workload_name=workload_name)

        # fig_1's number, and the denominator this figure divides by, so the two figures
        # are read against the same span
        cycle_count = query_cycle_count(
            event_graph=event_graph,
            metric_dict=metric_dict,
            workload=workload_name,
            event='llama_array'
        )

        # The per-GEMM sum above should reproduce it; the two views are built to charge
        # the same GEMMs with the same multiplicities. Say so out loud if they diverge --
        # a silent mismatch would put the numerator and denominator on different spans.
        if cycle_count > 0 and abs(gemm_cycles - cycle_count) > 1e-6 * cycle_count:
            warn(f'{run_path}: per-GEMM compute cycles {gemm_cycles:.6g} != llama_array '
                 f'{cycle_count:.6g}; the two views disagree on multiplicities')
            view_mismatches.append(run_path)

        utilization = macs / (pe_count * cycle_count) if cycle_count > 0 else 0

        # A ratio above 1 is not a tight design point, it is a broken measurement: more
        # useful MACs than the array has slots to do them in. It only happens when the
        # numerator and denominator are counting different multiplicities (the MoE expert
        # factor is the one that has bitten this before), so it is worth its own alarm --
        # a per-run warning buried under tqdm is easy to miss, hence the tally below.
        if utilization > 1 + 1e-9:
            warn(f'{run_path}: utilization {utilization:.4f} > 1 -- more useful MACs '
                 f'({macs:.6g}) than PE slots ({pe_count * cycle_count:.6g})')
            impossible.append((run_path, utilization))

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'max_seq_len': max_seq_len,
            'pe_count': pe_count,
            'useful_macs': macs,
            'cycle_count': cycle_count,
            'utilization': utilization,
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    if not array_query_df.empty:
        array_query_df = array_query_df.sort_values(by=['model', 'array_dim', 'batch_size', 'max_seq_len'])

        # Three max_seq_len slices: all models at 4096, all models at 131072,
        # and a mixed slice (llama at its 131072 point, deepseek at its 1048576
        # point) for comparing them at their respective long-context settings.
        llama_models = ['llama_3_1_8b', 'llama_3_1_70b', 'llama_3_1_405b']
        slices = {
            'array_utilization_metrics_seqlen_4096': array_query_df['max_seq_len'] == 4096,
            'array_utilization_metrics_seqlen_131072': array_query_df['max_seq_len'] == 131072,
            'array_utilization_metrics_seqlen_mixed': (
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
            for col in ['useful_macs', 'cycle_count']:
                df_slice_sci[col] = df_slice_sci[col].apply(lambda x: f'{x:.3e}')
            df_slice_sci['utilization'] = df_slice_sci['utilization'].apply(lambda x: f'{x:.4f}')
            df_slice_sci.to_csv(output_path + f'{out_name}_scientific.csv', index=False)
    else:
        print("Warning: No matching configurations found. CSVs not saved.")

# Reported after the bar is gone, so a broken sweep announces itself instead of being
# read off the CSV as a real result.
if view_mismatches or impossible:
    print(f'\nfig_3_query: {len(view_mismatches)} run(s) where the llama and llama_array '
          f'views disagree on multiplicities, {len(impossible)} run(s) with utilization > 1.')
    for run_path, utilization in impossible[:5]:
        print(f'  {utilization:.4f}  {run_path}')
    if len(impossible) > 5:
        print(f'  ... and {len(impossible) - 5} more')
    print('  The two views must charge the same GEMMs the same number of times; check '
          'llama_dc_array against layer_dc/layer_dc_moe in model.py.')
else:
    print('\nfig_3_query: both views agree on every run, and no utilization exceeds 1.')
