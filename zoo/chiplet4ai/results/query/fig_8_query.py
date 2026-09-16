"""Per-operator latency breakdown for fig_8.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/operator_latency_metrics{,_scientific}.csv.

WHAT THE NUMBER MEANS: the ARRAY COMPUTE cycles each individual GEMM contributes to the
model's total, split by phase. A GEMM charges array_input, array_weight and array_compute in
parallel and advances at the pace of the slowest; this breakdown reports the compute lane
ALONE, so weight loading and operand streaming are free -- a perfect-memory machine. The
per-lane cycles come from the checkpoint's edge factors; see utils.array_compute_cycles. So

    operator cycles = array_compute cycles(one instance) x multiplicity

and, because the GEMMs run sequentially, summing that product over every GEMM reproduces
the model's whole array_compute total EXACTLY rather than approximately. The query asserts
this; see SUM_TOLERANCE below. That is what makes the breakdown a decomposition of the
reported latency and not a parallel estimate of it.

MULTIPLICITY IS READ OFF '_dram', NOT '_arr'. '_arr' is reachable twice -- once under
`llama` through the GEMM and once under the `llama_array` compute-only view -- and
aggregate_event_count sums over every path, so an '_arr' count would be doubled. '_sram'
and '_dram' hang off `llama` alone. This is the same rule fig_2_query and utils.gemm_demand
follow.

THE HIERARCHY the figure draws, from the GEMM names model.py actually emits
(LAYER_EVENTS_PF / LAYER_EVENTS_DC plus the lm_head pair):

    projection   proj_q, proj_k, proj_v, a_proj      the Q/K/V and output projections
    attention    qkt, av                             the two score/context GEMMs
    ffn          gate_proj, up_proj, down_proj       the feed-forward GEMMs
    lm_head      lm_head                             the vocabulary projection

LM_HEAD IS ITS OWN CLASS rather than being folded into `projection` or dropped. It sits
outside the transformer block (prefill charges it once, decode once per generated token)
and it is not small -- at Llama-8B it is 0.16% of the model, more than every projection
GEMM put together. Dropping it would break the sum above, which is exactly the property
that makes this breakdown trustworthy, so it is reported rather than hidden.

SCOPE: one (array, batch) design point PER MODEL, at 1000 MHz with all three SRAMs at the
10 MiB base size, each model at its own MAXIMUM context. See DESIGN_POINT below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import array_compute_total, gemm_phase, lane_cycles
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
    print(f'WARNING [fig_8_query]: {message}', file=sys.stderr)

# ONE DESIGN POINT PER MODEL: (array shape, batch), READ FROM fig_4's THROUGHPUT CSV.
#
# WHY THE THROUGHPUT CRITERION and not one of the other two. FIG_4_CRITERIA in utils.py
# defines all three and says what each selects: `runtime` is not work-normalised and so
# just crowns whichever batch does the least work, `avg_band` deliberately picks the most
# memory-hungry design rather than the fastest, and `throughput` credits a bigger batch for
# the extra work it does -- "the serving metric, and the one that answers 'which machine
# would you build'". A per-operator LATENCY breakdown is a question about the machine you
# would actually build, so it is reported on that machine.
#
# DERIVED, NEVER TRANSCRIBED, exactly as fig_2_query derives its own point from the
# avg_band CSV: fig_4_query takes the argmax over fig_6's shape grid, so the point here is
# an extremum actually measured and cannot drift out of step through a stale copy.
# figure_generation.py runs fig_6_query -> fig_4_query before this script for that reason.
FIG_4_THROUGHPUT_CSV = 'zoo/chiplet4ai/results/csv/dram_bandwidth_metrics_throughput.csv'

# fig_1, fig_2 and fig_3 all report the 1000 MHz reference slice; cycle counts are
# frequency-invariant in the compute view, but the DRAM lane's bytes-per-cycle is not, so
# the two frequencies are distinct design points and must not be pooled.
REFERENCE_FREQUENCY = 1000
BASE_SRAM_BITS = 10 * 2**23

# The sum over operators is an identity, not an estimate (see the module docstring), so the
# only slack it needs is float64 round-off. The two sides are built by INDEPENDENT routes --
# the operators from the checkpoint's edge factors, the root through the engine's own
# aggregation -- so this is a genuine cross-check and not a restatement of one number. The
# observed residual is 0 for Llama 70B and 405B and ~1.2e-16 relative for Llama 8B and
# DeepSeek; 1e-9 leaves seven orders of headroom and still catches any real structural loss,
# such as a GEMM class going unmatched or a multiplicity being read off the wrong node.
SUM_TOLERANCE = 1e-9

# THE HIERARCHY. Keyed by the GEMM stem, i.e. the event name with its '_pf'/'_dc' phase
# suffix removed, so one entry covers both phases of an operator.
OPERATOR_CLASS = {
    'proj_q': 'projection',
    'proj_k': 'projection',
    'proj_v': 'projection',
    'a_proj': 'projection',
    'qkt': 'attention',
    'av': 'attention',
    'qkt_csa': 'attention',
    'av_csa': 'attention',
    'qkt_hca': 'attention',
    'av_hca': 'attention',
    'gate_proj': 'ffn',
    'up_proj': 'ffn',
    'down_proj': 'ffn',
    'lm_head': 'lm_head',
}


def design_point():
    """{model: ([array_m, array_n], batch, max_seq_len)} from fig_4's throughput CSV."""
    if not os.path.isfile(FIG_4_THROUGHPUT_CSV):
        raise SystemExit(
            f'fig_8_query: {FIG_4_THROUGHPUT_CSV} not found. It is written by fig_4_query, '
            f'which must run first -- figure_generation.py orders them that way.')

    points = {}
    for model, group in pd.read_csv(FIG_4_THROUGHPUT_CSV).groupby('model'):
        array_dims = group['array_dim'].unique()
        batches = group['batch_size'].unique()
        contexts = group['max_seq_len'].unique()
        if len(array_dims) != 1 or len(batches) != 1 or len(contexts) != 1:
            warn(f'{model}: fig_4 throughput reports {len(array_dims)} arrays, '
                 f'{len(batches)} batches and {len(contexts)} contexts; using the first of each')
        points[model] = ([int(side) for side in array_dims[0].split('x')],
                         int(batches[0]), int(contexts[0]))
    return points


DESIGN_POINT = design_point()
for model_name, (array, batch, context) in sorted(DESIGN_POINT.items()):
    print(f'  {model_name}: {array[0]}x{array[1]} batch {batch} at seq {context}')


def memory_sizes(architecture_dict):
    return [architecture_dict[sram]['query']['width'] * architecture_dict[sram]['query']['depth']
            * architecture_dict[sram]['query']['bank']
            for sram in ('isram', 'wsram', 'osram')]


def gemm_events(event_graph):
    """Every GEMM in the graph, by stem. '_dram' hangs off exactly the GEMM nodes."""
    return sorted(name[:-len('_dram')] for name in event_graph.get_all_node_names()
                  if name.endswith('_dram'))


def collect_operators(event_graph, edge_cycles, workload_name):
    """One record per GEMM that actually executes, plus the total they sum to."""
    records = []

    for gemm in gemm_events(event_graph):
        # The GEMM charges its '_arr'/'_sram'/'_dram' children at count 1 each, so the
        # '_dram' multiplicity IS the GEMM's multiplicity -- and unlike '_arr' it is
        # reachable only under `llama`, so it is not double counted through the
        # `llama_array` view.
        multiplicity = aggregate_event_count(
            event_graph=event_graph, workload=workload_name, event=f'{gemm}_dram')
        # A zero-count event is a branch this workload does not take (layer_dc for an MoE
        # model, layer_dc_moe for a dense one); it contributes nothing and is not a gap.
        if multiplicity <= 0:
            continue

        stem = gemm[:-len('_pf')] if gemm.endswith(('_pf', '_dc')) else gemm
        if stem not in OPERATOR_CLASS:
            # Refuse to quietly drop work: an unrecognised GEMM would silently break the
            # sum identity this whole breakdown rests on.
            raise SystemExit(
                f'fig_8_query: GEMM {gemm!r} has stem {stem!r}, which is not in '
                f'OPERATOR_CLASS. model.py has gained an operator; classify it there.')

        # The compute lane of ONE instance of the GEMM; the multiplicity scales it. The
        # lane's own cycles are its edge count x cycle factor (utils.lane_cycles).
        cycles_per_instance = edge_cycles.get((f'{gemm}_arr', 'array_compute'), 0.0)
        phase = gemm_phase(gemm)

        records.append({
            'gemm': gemm,
            'operator': stem,
            'operator_class': OPERATOR_CLASS[stem],
            'phase': phase,
            'multiplicity': multiplicity,
            'cycles_per_instance': cycles_per_instance,
            'cycle_count': cycles_per_instance * multiplicity,
        })

    return records


output_path = 'zoo/chiplet4ai/results/csv/'
runs_path = 'zoo/chiplet4ai/designs/llama/description/configurations.csv'
operator_query_df = pd.DataFrame()

if not os.path.exists(output_path):
    os.makedirs(output_path)

# model -> (sum over operators, the root event's own cycle count). Checked after the loop
# so every model's residual is reported together rather than the run dying on the first.
sum_check = {}
seen = set()

with open(runs_path, 'r') as f:
    runs_df = pd.read_csv(f)
    for index, row in tqdm(runs_df.iterrows(), total=len(runs_df)):
        run_path = row['run_path']

        architecture_dict = load_architecture_dict(run_path + '/architecture.yaml')
        workload_dict = load_workload_dict(run_path + '/workload.yaml')

        workload_name = workload_dict['name']
        if workload_name not in DESIGN_POINT:
            warn(f'{workload_name}: no design point configured, run skipped')
            continue

        point_array, point_batch, point_context = DESIGN_POINT[workload_name]
        array_dim = architecture_dict['pe']['instance']
        batch_size = workload_dict['configuration']['batch_size']
        max_seq_len = workload_dict['configuration']['max_seq_len']

        # cheap filters first, so the event graph is loaded only for the four runs we keep
        if array_dim != point_array or batch_size != point_batch:
            continue
        if max_seq_len != point_context:
            continue
        if architecture_dict['pe']['query']['frequency'] != REFERENCE_FREQUENCY:
            continue
        if any(size != BASE_SRAM_BITS for size in memory_sizes(architecture_dict)):
            continue

        event_graph = load_event_graph(run_path + '/checkpoint.json')
        metric_dict = load_metric_dict(run_path + '/metric.yaml')

        records = collect_operators(
            event_graph, lane_cycles(run_path + '/checkpoint.json'), workload_name)
        if not records:
            warn(f'{run_path}: no GEMM executed; row skipped')
            continue

        # The model's whole array_compute total, and the number the breakdown must
        # reproduce. Read through the engine's own aggregation rather than re-summed from
        # the edge factors above, so the check below compares two INDEPENDENT routes to the
        # same quantity instead of restating one of them.
        root_cycle_count = array_compute_total(event_graph, metric_dict, workload_name)
        sum_check[workload_name] = (sum(record['cycle_count'] for record in records),
                                    root_cycle_count)

        if workload_name in seen:
            warn(f'{workload_name}: a second run matched the design point; '
                 f'the CSV will carry both')
        seen.add(workload_name)

        phase_totals = {}
        class_totals = {}
        for record in records:
            phase_totals[record['phase']] = (
                phase_totals.get(record['phase'], 0.0) + record['cycle_count'])
            key = (record['phase'], record['operator_class'])
            class_totals[key] = class_totals.get(key, 0.0) + record['cycle_count']

        rows = []
        for record in records:
            phase_total = phase_totals[record['phase']]
            class_total = class_totals[(record['phase'], record['operator_class'])]
            rows.append({
                'model': workload_name,
                'array_dim': f'{array_dim[0]}x{array_dim[1]}',
                'batch_size': batch_size,
                'max_seq_len': max_seq_len,
                'frequency': architecture_dict['pe']['query']['frequency'],
                'phase': record['phase'],
                'operator_class': record['operator_class'],
                'operator': record['operator'],
                'gemm': record['gemm'],
                'multiplicity': record['multiplicity'],
                'cycles_per_instance': record['cycles_per_instance'],
                'cycle_count': record['cycle_count'],
                # the two denominators the figure reads, carried so a share can be checked
                # against the row it came from without regrouping the CSV
                'phase_cycle_count': phase_total,
                'class_cycle_count': class_total,
                'share_of_phase': record['cycle_count'] / phase_total if phase_total else 0.0,
                'share_of_model': record['cycle_count'] / root_cycle_count if root_cycle_count else 0.0,
                'model_cycle_count': root_cycle_count,
            })

        rows_df = pd.DataFrame(rows)
        operator_query_df = (pd.concat([operator_query_df, rows_df], ignore_index=True)
                             if not operator_query_df.empty else rows_df)

missing = sorted(set(DESIGN_POINT) - seen)
if missing:
    warn(f'no run matched the design point for: {", ".join(missing)}; '
         f'these models are absent from the figure')

# THE SUM CHECK, reported for every model before anything is written. The breakdown is a
# decomposition of the root cycle count, so a residual above float round-off means the
# walk lost or double counted work and the figure would misattribute latency.
failed = []
for model_name, (operator_total, root_total) in sorted(sum_check.items()):
    residual = abs(operator_total - root_total) / root_total if root_total else 0.0
    print(f'  sum check {model_name}: operators {operator_total:.6e} vs root '
          f'{root_total:.6e}, relative residual {residual:.3e}')
    if residual > SUM_TOLERANCE:
        failed.append(f'{model_name} ({residual:.3e})')
if failed:
    raise SystemExit(
        f'fig_8_query: per-operator cycles do not sum to the root cycle count within '
        f'{SUM_TOLERANCE:.0e} for: {", ".join(failed)}. The breakdown is not a '
        f'decomposition of the reported latency; do not publish it.')

if not operator_query_df.empty:
    # Sort ONCE, numerically, before anything is stringified, so the plain and scientific
    # CSVs stay row-aligned (the same trap fig_2_query documents).
    operator_query_df = operator_query_df.sort_values(
        by=['model', 'phase', 'operator_class', 'operator'])
    operator_query_df.to_csv(output_path + 'operator_latency_metrics.csv', index=False)

    operator_query_df_sci = operator_query_df.copy()
    for col in ['multiplicity', 'cycles_per_instance', 'cycle_count',
                'phase_cycle_count', 'class_cycle_count', 'model_cycle_count']:
        operator_query_df_sci[col] = operator_query_df_sci[col].apply(lambda x: f'{x:.3e}')
    for col in ['share_of_phase', 'share_of_model']:
        operator_query_df_sci[col] = operator_query_df_sci[col].apply(lambda x: f'{x:.4f}')
    operator_query_df_sci.to_csv(output_path + 'operator_latency_metrics_scientific.csv',
                                 index=False)
    print(f'wrote {len(operator_query_df)} rows')
else:
    print('Warning: No matching configurations found. No CSV saved.')
