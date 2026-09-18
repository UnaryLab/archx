"""Per-sequence-length array-utilization query for fig_5.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/array_utilization_nostall_metrics.csv.

WHAT THE NUMBER MEANS: useful MACs divided by the compute lane's PE slots --
`M*K*N / (array_m * array_n * compute_cycles)` -- for ONE step of the model, at one
context length. Weight loading and operand streaming are free (perfect memory), so what
remains is pure MAPPING efficiency: the tiling loss from a reduction that does not fill
the array's rows, an output that does not fill its columns, and partial edge tiles.

AT a context, not UP TO it. fig_5 used to read the folded event graph, whose decode
events have already summed every step from `prefill_seq_len` to `max_seq_len`; that
aggregate cannot be un-summed afterwards. So the mapping is called directly here instead,
once per context. The trick that isolates a single step is the workload itself: setting
`prefill_seq_len = context - 1` and `max_seq_len = context` leaves `array_mapping_decode`
exactly one step to walk, at that context, and leaves every per-step multiplicity in
model.py at 1. The 2048 point is the prefill of the whole prompt rather than a decode
step, since that is the only phase that runs at the prompt length.

NOTHING IS RESTATED. The GEMM shapes come from model.py's own `*_arr` functions and the
multiplicities from its own `_decode_gemm_counts` / `_prefill_gemm_counts` /
`_compressed_gemm_counts`, so this query cannot drift from the model the other figures
read. Only the top-level composition below mirrors `model.llama_array`.

SCOPE: square arrays at the 10 MiB reference SRAM size, the 1000 MHz slice, batch 512.
Llama stops at its 131072 context; only DeepSeek reaches 1048576.
"""

import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
import pandas as pd
from tqdm import tqdm

from archx.architecture import load_architecture_dict
from archx.workload import load_workload_dict
from chiplet4ai.designs.llama import model
from chiplet4ai.results.query.utils import CORE_ARRAY_SIZES, FIG_BATCH_SIZE

logger.remove()

# The prefill length every workload is configured with, and the decode contexts the
# figure steps through. 2048 is the prompt itself; the rest are single decode steps.
PREFILL_CONTEXT = 2048
CONTEXTS = [PREFILL_CONTEXT, 4096, 131072, 1048576]

def warn(message):
    print(f'WARNING [fig_5_query]: {message}', file=sys.stderr)

def memory_sizes(architecture_dict):
    srams = []
    for sram in ['isram', 'wsram', 'osram']:
        sram_query = architecture_dict[sram]['query']
        srams.append(sram_query['width'] * sram_query['depth'] * sram_query['bank'])
    return srams

def gemm_multiplicities(cfg, phase):
    """{GEMM event: how many times one step runs it}, mirroring `model.llama_array`.

    Dense, CSA and HCA layers each carry their own GEMM set, and the lm_head runs once
    outside the layer stack. With the workload trimmed to a single step every per-step
    multiplicity is 1, so what survives here is the layer count and, for DeepSeek's FFN,
    the activated-expert count.
    """
    compressed = model._compressed_layers(cfg)
    dense_layers = cfg['layers'] - sum(compressed.values())

    if phase == 'pf':
        groups = [(dense_layers, model._prefill_gemm_counts(cfg))]
        head, head_count = 'lm_head_pf', 1
    else:
        groups = [(dense_layers, model._decode_gemm_counts(cfg))]
        head, head_count = 'lm_head_dc', cfg['max_seq_len'] - cfg['prefill_seq_len']
    groups += [(compressed[kind], model._compressed_gemm_counts(cfg, kind, phase))
               for kind in model.COMPRESSED_LAYER_KINDS]

    totals = OrderedDict()
    for layers, counts in groups:
        for event, count in counts.items():
            totals[event] = totals.get(event, 0) + layers * count
    totals[head] = totals.get(head, 0) + head_count

    return OrderedDict((event, count) for event, count in totals.items() if count > 0)

def step_workload(workload_dict, context):
    """The workload as one step at `context`, and which phase that step belongs to.

    A decode step is isolated by starting the walk one token short of the context, so
    `_step_config` yields a single stride and every `steps`-scaled multiplicity is 1.
    """
    cfg = deepcopy(workload_dict['configuration'])
    if context == PREFILL_CONTEXT:
        phase = 'pf'
    else:
        phase = 'dc'
        cfg['prefill_seq_len'] = context - 1
        cfg['max_seq_len'] = context
    return OrderedDict({**workload_dict, 'configuration': cfg}), phase

def step_macs_and_cycles(architecture_dict, workload_dict, phase, pe_count):
    """Useful MACs and compute-lane cycles for one step, summed over its GEMMs.

    array_mapping writes each lane's count already scaled to useful work and puts the
    true cycles in the factor, so `count * factor` is the lane's cycle count and
    `count * pe_count` is its MAC count (array.py charges array_m * array_n pe events).
    """
    macs = 0.0
    cycles = 0.0
    for event, multiplicity in gemm_multiplicities(workload_dict['configuration'], phase).items():
        compute = getattr(model, f'{event}_arr')(
            architecture_dict, workload_dict)['subevent']['array_compute']
        macs += compute['count'] * pe_count * multiplicity
        cycles += compute['count'] * compute['factor']['cycle_count'] * multiplicity
    return macs, cycles

output_path = 'zoo/chiplet4ai/results/csv/'
runs_path = 'zoo/chiplet4ai/designs/llama/description/configurations.csv'

if not os.path.exists(output_path):
    os.makedirs(output_path)

runs_df = pd.read_csv(runs_path)

# One architecture per array size: the sweep repeats each shape across frequencies and
# SRAM sizes, and only the reference slice is wanted. The per-run architecture.yaml is
# the materialized one; the description-level arch_path holds swept parameters only.
architectures = OrderedDict()
for run_path in runs_df['run_path'].unique():
    if len(architectures) == len(CORE_ARRAY_SIZES):
        break
    architecture_dict = load_architecture_dict(run_path + '/architecture.yaml')
    array_dim = architecture_dict['pe']['instance']
    if array_dim[0] != array_dim[1] or array_dim[0] not in CORE_ARRAY_SIZES:
        continue
    if architecture_dict['pe']['query']['frequency'] != 1000:
        continue
    if any(size != 10 * 2**23 for size in memory_sizes(architecture_dict)):
        continue
    architectures.setdefault(array_dim[0], architecture_dict)

# One batch-512 workload per model, plus the longest context each model was configured
# for -- the contexts are overridden per point, but a model must not be plotted past the
# context it actually supports.
workloads = OrderedDict()
model_max_context = OrderedDict()
for work_path in runs_df['work_path'].unique():
    workload_dict = load_workload_dict(work_path)
    name = workload_dict['name']
    cfg = workload_dict['configuration']
    model_max_context[name] = max(model_max_context.get(name, 0), cfg['max_seq_len'])
    if cfg['batch_size'] == FIG_BATCH_SIZE:
        workloads.setdefault(name, workload_dict)

rows = []
impossible = []
points = [(name, size, context)
          for name in workloads for size in architectures for context in CONTEXTS
          if context <= model_max_context[name]]

for name, array_size, context in tqdm(points):
    architecture_dict = architectures[array_size]
    step_dict, phase = step_workload(workloads[name], context)
    pe_count = array_size * array_size

    macs, cycles = step_macs_and_cycles(architecture_dict, step_dict, phase, pe_count)
    utilization = macs / (pe_count * cycles) if cycles > 0 else 0

    # Above 1 is not a tight design point, it is a broken measurement: more useful MACs
    # than the array has slots to do them in.
    if utilization > 1 + 1e-9:
        warn(f'{name} {array_size}x{array_size} @ {context}: utilization {utilization:.4f} > 1 '
             f'-- more useful MACs ({macs:.6g}) than PE slots ({pe_count * cycles:.6g})')
        impossible.append((name, array_size, context, utilization))

    rows.append({
        'model': name,
        'array_dim': f'{array_size}x{array_size}',
        'batch_size': FIG_BATCH_SIZE,
        'seq_len': context,
        'phase': 'prefill' if phase == 'pf' else 'decode',
        'pe_count': pe_count,
        'useful_macs': macs,
        'compute_cycle_count': cycles,
        'utilization': utilization,
    })

array_query_df = pd.DataFrame(rows).sort_values(by=['model', 'array_dim', 'seq_len'])
array_query_df.to_csv(output_path + 'array_utilization_nostall_metrics.csv', index=False)

df_sci = array_query_df.copy()
for col in ['useful_macs', 'compute_cycle_count']:
    df_sci[col] = df_sci[col].apply(lambda x: f'{x:.3e}')
df_sci['utilization'] = df_sci['utilization'].apply(lambda x: f'{x:.4f}')
df_sci.to_csv(output_path + 'array_utilization_nostall_metrics_scientific.csv', index=False)

# Reported after the bar is gone, so a broken sweep announces itself instead of being
# read off the CSV as a real result.
if impossible:
    print(f'\nfig_5_query: {len(impossible)} point(s) with utilization > 1.')
    for name, array_size, context, utilization in impossible[:5]:
        print(f'  {utilization:.4f}  {name} {array_size}x{array_size} @ {context}')
    if len(impossible) > 5:
        print(f'  ... and {len(impossible) - 5} more')
else:
    print(f'\nfig_5_query: {len(rows)} points, no utilization exceeds 1.')
