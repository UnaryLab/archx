"""Peak and average DRAM bandwidth per technology node, for fig_4.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/dram_bandwidth_metrics.csv.

WHAT THE NUMBERS MEAN: both are REQUIRED bandwidth -- bytes moved over the on-chip window
they must hide under -- pooled across all three DRAM lanes. The declared `dram: bandwidth:`
never enters the calculation; it is carried in its own column purely so the figure can
draw it as a reference line.

  average = sum(bytes) / sum(window)   over every GEMM: the rate the channel must sustain
                                       across the whole model
  peak    = max over PHASE of          prefill and decode pooled separately: the rate the
            sum(bytes) / sum(window)   hungrier phase must sustain, i.e. what the channel
                                       must be sized for

The window is per GEMM, max(array compute cycles, SRAM port cycles). DRAM is excluded
from it, being the thing being sized. GEMMs run sequentially, so their windows add.

WHY THE PEAK IS POOLED PER PHASE AND NOT TAKEN PER GEMM. The obvious definition --
max(bytes / window) over individual GEMMs -- measures a burst that cannot actually occur,
for two independent reasons:

  PREFETCH. The buffers are double buffered: `_buffer_elements_from` in mapping.py charges
  only `bank // 2` as usable capacity precisely because the other half is filling while the
  active half is read. Traffic for the next tile therefore overlaps the current tile's
  compute, and that does not stop at a GEMM boundary -- the next GEMM's operands load into
  the free half while the current one computes. A GEMM's traffic is smoothed into its
  neighbours' windows, so no GEMM has to be fed inside its own window alone.

  INTERLEAVING. A per-GEMM row here is not a contiguous block of execution. Its
  multiplicity is how many times the GEMM runs across the whole model -- for the decode
  GEMMs, once per generated token -- so `lm_head_dc` is one small GEMM per decode step
  sitting between that step's layer GEMMs, not one long burst. Aggregating it into a single
  row and dividing by its own summed window asks it to sustain, alone, a rate it never runs
  at in isolation.

  Together these made the old number an artifact: at the Llama-8B design point it was set
  by lm_head_dc at 2417 GB/s, a GEMM holding 0.11% of the bytes and 0.0156% of the window,
  while qkt_dc and av_dc -- 99.1% of all traffic and 99.8% of the time -- demand 508 and
  255 GB/s.

A PHASE is the right unit because it is the largest span over which the GEMM mix repeats:
decode runs the same GEMMs every step and prefill the same GEMMs every layer, so a phase
rate is a rate the channel must genuinely hold for the phase's whole duration -- 75 seconds
of prefill for Llama-8B, far past anything a 15 MiB free buffer half can absorb. It is also
where the two phases genuinely differ: prefill streams weights against large-M GEMMs and
demands 1.0-2.1 TB/s, while decode sits at essentially the whole-model average because it
is 99.99% of the runtime.

The old per-GEMM number is still written, as `burst_bandwidth` / `burst_gemm`, so the gap
between the two is visible rather than silently discarded.

THE NODE AXIS is frequency, and only frequency. fig_4 maps Sub-20nm and Sub-10nm to
1000 MHz and Sub-5nm to 2000 MHz. The design space has no technology axis of its own
(description.py fixes technology=7 for the architecture and 45 for the cacti queries), so
Sub-20nm and Sub-10nm are the SAME design point and produce identical bars. Cycle counts
are frequency-invariant here, so the 2000 MHz node is exactly 2x the 1000 MHz one: the
same traffic in half the time.

SCOPE: one (array, batch) design point PER MODEL -- see DESIGN_POINT below -- with all
three SRAMs at the 10 MiB base size, both frequencies, at that model's MAXIMUM context.

The point is no longer fig_2's. fig_2 holds an array fixed while SRAM capacity moves, so
its point exists to be a control; this figure reports the machine actually worth building,
which is what fig_6's aspect-ratio grid measures. The two are now chosen separately and
description.py generates them separately -- fig_2's from `sram_sweep_point`, this one as a
cell of fig_6's max-context grid.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import (query_cycle_count, gemm_demand,
                                            bandwidth_summary, FIG_4_CRITERIA, fig_4_paths,
                                            select_design_point, SELECTION_MARGIN)
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
    print(f'WARNING [fig_4_query]: {message}', file=sys.stderr)

# THREE DESIGN POINTS PER MODEL, one per criterion in FIG_4_CRITERIA, each DERIVED from
# fig_6's grid rather than written down here. fig_6_query sweeps all 25 shapes x 5 batches
# at each model's max context and reports the same bandwidth numbers this script does, from
# the same shared implementation in utils.py; this takes the argmin/argmax of that CSV, so
# every point fig_4 reports is an extremum actually measured and cannot drift through a
# stale transcription.
#
# WHY THREE. "The best design" is not one question, and the three answers disagree: the
# runtime criterion is not work-normalised and so favours small batches, the bandwidth
# criterion selects the most memory-hungry rather than the fastest design, and throughput
# credits a bigger batch for the work it does. See FIG_4_CRITERIA in utils.py.
#
# Ranked at 1000 MHz, fig_4's reference node. Bandwidth is exactly proportional to
# frequency (the cycle counts behind it are frequency-invariant), so its ranking is
# identical at 2000 MHz; the runtime and throughput rankings shift slightly.
SHAPE_CSV = 'zoo/chiplet4ai/results/csv/array_shape_performance_metrics.csv'
RANK_FREQUENCY = 1000
BASE_SRAM_BITS = 10 * 2**23

def memory_sizes(architecture_dict):
    return [architecture_dict[sram]['query']['width'] * architecture_dict[sram]['query']['depth']
            * architecture_dict[sram]['query']['bank']
            for sram in ('isram', 'wsram', 'osram')]

def design_points():
    """{criterion key: ({model: ([m, n], batch)}, {model: max_seq_len})} from fig_6's CSV.

    figure_generation.py runs fig_6_query before this script for exactly this reason. Run
    standalone, it needs that CSV to already exist.
    """
    if not os.path.isfile(SHAPE_CSV):
        raise SystemExit(
            f'fig_4_query: {SHAPE_CSV} not found. It is written by fig_6_query, which must '
            f'run first -- figure_generation.py orders them that way.')

    shapes = pd.read_csv(SHAPE_CSV)
    shapes = shapes[shapes['frequency'] == RANK_FREQUENCY]
    if shapes.empty:
        raise SystemExit(f'fig_4_query: no {RANK_FREQUENCY} MHz rows in {SHAPE_CSV}')

    selections = {}
    for criterion in FIG_4_CRITERIA:
        column, direction = criterion['column'], criterion['direction']
        if column not in shapes.columns:
            raise SystemExit(f'fig_4_query: {SHAPE_CSV} has no {column!r} column; '
                             f'fig_6_query must write it for the '
                             f'{criterion["key"]!r} criterion.')
        points, contexts = {}, {}
        print(f"  [{criterion['key']}] {direction} of {column} -- {criterion['label']} "
              f"({criterion['note']}), within {SELECTION_MARGIN:.1%} of best")
        for model, group in shapes.groupby('model'):
            chosen, best_value = select_design_point(group, column, direction)
            points[model] = ([int(chosen['array_m']), int(chosen['array_n'])],
                             int(chosen['batch_size']))
            # every row for a model shares its max context, so this is that context
            contexts[model] = int(chosen['max_seq_len'])
            gap = abs(chosen[column] - best_value) / best_value if best_value else 0
            print(f'      {model}: {int(chosen["array_m"])}x{int(chosen["array_n"])} '
                  f'batch {int(chosen["batch_size"])} at seq {contexts[model]} '
                  f'-- {column} {chosen[column]:.4g}, {int(chosen["pe_count"]):,} PEs, '
                  f'{gap:.2%} off the best of {len(group)}')
        selections[criterion['key']] = (points, contexts)
    return selections

SELECTIONS = design_points()

# The union of the three criteria's points, so the run loop below walks the configurations
# once rather than three times. At most 3 points x 4 models x 2 frequencies survive it, and
# the criteria overlap in practice, so the union is usually smaller than that.
WANTED = {(model, tuple(array), batch)
          for points, _ in SELECTIONS.values()
          for model, (array, batch) in points.items()}
WANTED_CONTEXT = {model: context
                  for _, contexts in SELECTIONS.values()
                  for model, context in contexts.items()}

output_path = 'zoo/chiplet4ai/results/csv/'
runs_path = f'zoo/chiplet4ai/designs/llama/description/configurations.csv'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# (model, array, batch, frequency) -> the row for that configuration, built once and then
# handed to whichever criteria selected it.
rows = {}

with open(runs_path, 'r') as f:
    runs_df = pd.read_csv(f)
    for index, row in tqdm(runs_df.iterrows(), total=len(runs_df)):
        run_path = row['run_path']

        architecture_dict = load_architecture_dict(run_path + '/architecture.yaml')
        workload_dict = load_workload_dict(run_path + '/workload.yaml')

        workload_name = workload_dict['name']
        array_dim = architecture_dict['pe']['instance']
        batch_size = workload_dict['configuration']['batch_size']
        if (workload_name, tuple(array_dim), batch_size) not in WANTED:
            continue
        if workload_dict['configuration']['max_seq_len'] != WANTED_CONTEXT[workload_name]:
            continue
        if any(size != BASE_SRAM_BITS for size in memory_sizes(architecture_dict)):
            continue

        event_graph = load_event_graph(run_path + '/checkpoint.json')
        metric_dict = load_metric_dict(run_path + '/metric.yaml')

        # BOTH frequencies are kept: frequency is this figure's node axis, not a slice
        # to filter away.
        frequency_mhz = architecture_dict['pe']['query']['frequency']

        demand = gemm_demand(event_graph, metric_dict, workload_name)
        if not demand:
            warn(f'{run_path}: no GEMM carried traffic; row skipped')
            continue

        summary = bandwidth_summary(demand, frequency_mhz)

        rows[(workload_name, tuple(array_dim), batch_size, frequency_mhz)] = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            'max_seq_len': workload_dict['configuration']['max_seq_len'],
            'frequency': frequency_mhz,
            'sram_size': BASE_SRAM_BITS,
            # the channel the architecture declares, in decimal GB/s: never used in the
            # numbers above, carried so fig_4 can draw it as its reference line
            'dram_bandwidth': architecture_dict['dram']['query']['bandwidth'],
            'total_data_moved': summary['total_bytes'],
            'window_cycle_count': summary['total_window'],
            'average_bandwidth': summary['average'],
            'peak_bandwidth': summary['peak'],
            # provenance: which phase sets the peak, so a surprising bar can be traced
            'peak_phase': summary['peak_phase'],
            # the superseded per-GEMM burst, and the GEMM that sets it
            'burst_bandwidth': summary['burst'],
            'burst_gemm': summary['burst_gemm'],
        }

for criterion in FIG_4_CRITERIA:
    points, contexts = SELECTIONS[criterion['key']]
    csv_path, sci_path, _ = fig_4_paths(criterion['key'])

    selected = []
    for model, (array, batch) in points.items():
        found = [value for key, value in rows.items()
                 if key[:3] == (model, tuple(array), batch)]
        if not found:
            warn(f"[{criterion['key']}] {model}: no run at "
                 f"{array[0]}x{array[1]} batch {batch}; every bar for it will be missing")
        selected.extend(found)

    if not selected:
        print(f"Warning: [{criterion['key']}] no matching configurations. No CSV saved.")
        continue

    criterion_df = pd.DataFrame(selected).sort_values(by=['model', 'max_seq_len', 'frequency'])
    criterion_df.to_csv(csv_path, index=False)

    criterion_df_sci = criterion_df.copy()
    for col in ['total_data_moved', 'window_cycle_count']:
        criterion_df_sci[col] = criterion_df_sci[col].apply(lambda x: f'{x:.3e}')
    for col in ['average_bandwidth', 'peak_bandwidth', 'burst_bandwidth']:
        criterion_df_sci[col] = criterion_df_sci[col].apply(lambda x: f'{x:.1f}')
    criterion_df_sci.to_csv(sci_path, index=False)
    print(f"wrote {len(criterion_df)} rows to {csv_path}")
