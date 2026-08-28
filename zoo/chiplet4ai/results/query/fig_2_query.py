"""Required-DRAM-bandwidth query for fig_2.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/bandwidth_performance_metrics{,_scientific}.csv.

WHAT THE NUMBER MEANS: the bandwidth the DRAM channel WOULD have to sustain for it never
to be the bottleneck -- DRAM bytes over the on-chip span they must hide under. The
architecture's declared `dram: bandwidth:` never enters: this is a demand, not an
achieved rate, so a figure of required bandwidth against SRAM size is independent of
whatever channel the description happens to declare.

ONE CURVE PER OPERAND: ifmap, wmap and ofmap are reported separately, never pooled. They
share the window (a GEMM advances at the pace of its slowest stream) but their bytes
answer to different parts of the tiling -- ifmap to the N tile count, wmap to the M sweep
count, ofmap to the K folds -- so they fall at different rates as SRAM grows. They are
emitted under fig_2.py's existing column names (`input_/weight_/output_dram_bandwidth`),
which name a demand rather than an achieved rate.

THE WINDOW is per GEMM, `max(array cycles, SRAM cycles)`. The array span alone would
ignore that the SRAM ports already stall the array; including the SRAM span asks the
honest question, "how fast must DRAM be so that it is not the one stalling us". DRAM's
own cycle_count is deliberately excluded from the window -- it is the quantity being
sized. GEMMs run sequentially, so the spans add.

WHY SRAM SIZE MOVES IT: mapping.py folds the GEMM against the active half of each SRAM,
so the DRAM byte counts already carry the reuse -- input is re-streamed once per N tile,
the weight matrix once per M sweep. A bigger SRAM means fewer tiles and fewer sweeps,
hence fewer bytes over an unchanged compute span.

SCOPE: one (array, batch) design point PER MODEL -- see DESIGN_POINT below -- and the
DIAGONAL of the SRAM sweep, only the configurations where isram, wsram and osram are the
same size, so the figure has a single "SRAM size" axis rather than one buffer pooled
against another.

The design point is not shared across models. DeepSeek is reported on 256x256 at batch
256, the Llama models on 128x128 at batch 128, because DeepSeek keeps scaling onto the
wider array where the Llamas have already saturated. description.py generates the sweep
at exactly these points (`sweep_design_point`, which carries the measurement behind that
split); if the two ever disagree a model's rows vanish from the output, so a model with
no surviving row is reported rather than skipped silently.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import query_cycle_count, bandwidth_gbs
from chiplet4ai.common.performance.mapping import _buffer_elements
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
    print(f'WARNING [fig_2_query]: {message}', file=sys.stderr)

# ONE DESIGN POINT PER MODEL: (array shape, batch), READ FROM fig_4's avg_band CSV so
# fig_2's capacity curve and fig_4's bandwidth bars describe the same machine. That point
# is the argmax of `average_bandwidth` over fig_6's grid -- deliberately the most
# memory-hungry configuration rather than the fastest, because a capacity sweep is most
# informative where capacity is under the most pressure.
#
# DERIVED, BUT NOT A FREE CHOICE. Unlike fig_4, which can point anywhere in fig_6's grid
# because every cell of it is simulated, fig_2 needs the OFF-BASE SRAM sizes, and those
# exist only where description.py's `sram_sweep_point` puts them. Reading the CSV keeps
# this file from drifting, but if a re-run moves the argmax, `sram_sweep_point` must move
# with it and the design space must be regenerated -- otherwise this filter looks for a
# point the sweep was never run at. The check below says so out loud instead of quietly
# writing an empty CSV.
FIG_4_AVG_BAND_CSV = 'zoo/chiplet4ai/results/csv/dram_bandwidth_metrics_avg_band.csv'

def design_point():
    if not os.path.isfile(FIG_4_AVG_BAND_CSV):
        raise SystemExit(
            f'fig_2_query: {FIG_4_AVG_BAND_CSV} not found. It is written by fig_4_query, '
            f'which must run first -- figure_generation.py orders them that way.')
    points = {}
    for model, group in pd.read_csv(FIG_4_AVG_BAND_CSV).groupby('model'):
        array_dims = group['array_dim'].unique()
        batches = group['batch_size'].unique()
        if len(array_dims) != 1 or len(batches) != 1:
            warn(f'{model}: fig_4 avg_band reports {len(array_dims)} arrays and '
                 f'{len(batches)} batches; using the first of each')
        points[model] = ([int(side) for side in array_dims[0].split('x')], int(batches[0]))
    return points

DESIGN_POINT = design_point()

# ONE SEQUENCE LENGTH PER MODEL: each at its own long-context setting, the same slice
# fig_1_query calls "mixed". Without this the rows pool sequence lengths -- and because
# description.py only admits the 4096 configurations when every SRAM sits at
# base_sram_size, that pooling hits exactly one point of the sweep, putting a cliff at the
# largest SRAM size that is an artifact rather than a result.
LONG_CONTEXT = {
    'llama_3_1_8b': 131072,
    'llama_3_1_70b': 131072,
    'llama_3_1_405b': 131072,
    'deepseek_v4': 1048576,
}

# memory.py bills one DRAM event per byte, so an edge count IS a byte count.
DRAM_LANES = {
    'input': 'dram_input_read',
    'weight': 'dram_weight_read',
    # Nonzero only when mapping.py found spilling the partial sums cheaper than
    # re-walking the weight matrix; otherwise they accumulate in osram and this lane
    # stays 0, which the column then records as provenance.
    'output_read': 'dram_output_read',
    'output_write': 'dram_output_write',
}

# GEMM CLASSES. Attention and the dense layers answer to capacity in completely different
# ways, and pooling them hides both: attention's operand is the KV cache, streamed once
# per token and far larger than any buffer here, so it is flat at every SRAM size and --
# being the bulk of the traffic at long context -- it flattens whatever it is averaged
# with. The dense layers are where reuse lives. Splitting them makes each visible, and
# the contrast is the point: capacity buys reuse for projection and FFN, and buys nothing
# against the KV cache.
ATTENTION_GEMMS = ('qkt', 'av')

def gemm_class(gemm):
    stem = gemm[:-3] if gemm.endswith(('_pf', '_dc')) else gemm
    return 'attention' if stem in ATTENTION_GEMMS else 'proj/ffn'

def gemm_events(event_graph):
    return sorted(name[:-len('_dram')] for name in event_graph.get_all_node_names()
                  if name.endswith('_dram'))

def collect_demand(event_graph, metric_dict, workload_name):
    """Per-GEMM DRAM bytes and the on-chip span they hide under, summed over the model."""
    classes = ('proj/ffn', 'attention')
    lane_bytes = {cls: {lane: 0.0 for lane in DRAM_LANES} for cls in classes}
    totals = {cls: dict(window=0.0, array=0.0, sram=0.0, dram=0.0, stall=0.0)
              for cls in classes}

    for gemm in gemm_events(event_graph):
        # Multiplicity is read off the '_dram' node, NOT off '_arr'. '_arr' is reachable
        # twice -- once under `llama` through the GEMM and once under the `llama_array`
        # view -- and aggregate_event_count sums over every path, so an '_arr' count
        # would be doubled. '_sram' and '_dram' hang off `llama` alone, and the GEMM
        # charges all three children at count 1, so the '_dram' multiplicity is the
        # GEMM's multiplicity.
        multiplicity = aggregate_event_count(
            event_graph=event_graph, workload=workload_name, event=f'{gemm}_dram')
        sram_multiplicity = aggregate_event_count(
            event_graph=event_graph, workload=workload_name, event=f'{gemm}_sram')
        if multiplicity != sram_multiplicity:
            warn(f'{gemm}: _dram multiplicity {multiplicity} != _sram {sram_multiplicity}; '
                 f'the GEMM no longer charges its children alike, sample dropped')
            continue
        if multiplicity <= 0:
            continue

        # workload=None asks for the value of ONE instance of the event; the
        # multiplicity above scales it, the same decomposition query_cycle_breakdown uses
        gemm_array = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                       workload=None, event=f'{gemm}_arr')
        gemm_sram = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                      workload=None, event=f'{gemm}_sram')

        # What the DECLARED channel would actually take to move this GEMM's bytes. It
        # plays no part in the required-bandwidth columns -- those are a demand -- but it
        # is what turns the demand into a consequence: any excess over the on-chip window
        # is time the array spends waiting on DRAM.
        gemm_dram = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                      workload=None, event=f'{gemm}_dram')

        cls = gemm_class(gemm)
        window = max(gemm_array, gemm_sram)
        totals[cls]['array'] += gemm_array * multiplicity
        totals[cls]['sram'] += gemm_sram * multiplicity
        totals[cls]['dram'] += gemm_dram * multiplicity
        totals[cls]['window'] += window * multiplicity
        totals[cls]['stall'] += max(0.0, gemm_dram - window) * multiplicity

        for lane, event in DRAM_LANES.items():
            lane_bytes[cls][lane] += event_graph.get_edge_count(f'{gemm}_dram', event) * multiplicity

    return lane_bytes, totals

# DECIMAL GB/s, from the shared helper, NOT a local GiB/s one. This used to divide by
# 2**30 while fig_4 divided by 1e9, so the two figures reported the same demand 7.4% apart
# and fig_2's axis said "GB/s" over a GiB/s number. One implementation now, and it is the
# same unit the declared `dram: bandwidth:` is written in.
required_bandwidth = bandwidth_gbs

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
        # The design point is per model, so the array and batch filters need the workload
        # name -- read it here rather than at its later use, so the cheap filters still
        # run before the event graph is walked.
        if workload_dict['name'] not in DESIGN_POINT:
            warn(f"{workload_dict['name']}: no design point configured, run skipped")
            continue
        point_array, point_batch = DESIGN_POINT[workload_dict['name']]
        if array_dim != point_array:
            continue
        if workload_dict['configuration']['batch_size'] != point_batch:
            continue

        isram_bank = architecture_dict['isram']['query']['bank']
        isram_depth = architecture_dict['isram']['query']['depth']
        isram_width = architecture_dict['isram']['query']['width']

        wsram_bank = architecture_dict['wsram']['query']['bank']
        wsram_depth = architecture_dict['wsram']['query']['depth']
        wsram_width = architecture_dict['wsram']['query']['width']

        # description.py direct-constrains osram bank and depth to wsram's, so
        # osram_size == wsram_size on every swept config; the column is still read from the
        # architecture (not copied) so a future de-coupled sweep stays honest.
        osram_bank = architecture_dict['osram']['query']['bank']
        osram_depth = architecture_dict['osram']['query']['depth']
        osram_width = architecture_dict['osram']['query']['width']

        isram_size = isram_bank * isram_depth * isram_width
        wsram_size = wsram_bank * wsram_depth * wsram_width
        osram_size = osram_bank * osram_depth * osram_width

        # DIAGONAL SLICE: keep only the configurations where all three SRAMs are the SAME
        # size, so "SRAM size" is one axis.
        #
        # description.py sweeps isram and wsram independently, and pooling one against the
        # other flattens every curve: a configuration with a large wsram already fetches
        # the input in a single pass, so its asram has nothing left to save, and averaging
        # that saturated point with an unsaturated one hides the effect. Sweeping the
        # buffers together asks the question a reader of this axis actually has -- how much
        # on-chip memory does the design need -- while each operand stays charged to its
        # own buffer.
        if not (isram_size == wsram_size == osram_size):
            continue

        # the capacity mapping.py actually tiles against: the active half of the double
        # buffer, imported rather than re-derived so the CSV credits what the model does
        isram_active_elements = _buffer_elements(architecture_dict, 'isram')[0]
        wsram_active_elements = _buffer_elements(architecture_dict, 'wsram')[0]

        # Reported, not assumed. mapping.py no longer carries an `active_fraction` knob --
        # _buffer_elements hardwires the split at half the banks -- so the fraction is
        # measured back out of the capacity it actually credits. It is 0.5 for every even
        # bank count (all of them here, banks being 2x a vector width) and less when the
        # whole-bank floor bites.
        isram_active_fraction = isram_active_elements / (isram_bank * isram_depth)

        batch_size = workload_dict['configuration']['batch_size']
        max_seq_len = workload_dict['configuration']['max_seq_len']
        workload_name = workload_dict['name']

        if workload_name not in LONG_CONTEXT:
            warn(f'{workload_name}: no long-context length configured, run skipped')
            continue
        if max_seq_len != LONG_CONTEXT[workload_name]:
            continue

        if architecture_dict['pe']['query']['frequency'] != 1000:
            continue

        # SWEEP FILTER: the design point fig_2 reports.
        #
        # NOTE: description.py sweeps SRAM capacity only at `fixed_array_shape`, which is
        # [512, 512]; every other array shape is pinned to base_sram_size by the
        # pe/bank/depth conditional constraint. Until that constant moves, this slice has
        # ONE SRAM size and the figure's x-axis collapses to a single point.
        

        frequency_mhz = architecture_dict['pe']['query']['frequency']
        lane_bytes, totals = collect_demand(
            event_graph=event_graph,
            metric_dict=metric_dict,
            workload_name=workload_name,
        )

        # one row per GEMM class: each carries its own bytes AND its own window, since
        # the classes run in sequence and each hides its traffic under its own span
        for gemm_class_name, class_bytes in lane_bytes.items():
          class_totals = totals[gemm_class_name]
          window_cycles = class_totals['window']
          array_cycles = class_totals['array']
          sram_cycles = class_totals['sram']
          dram_cycles = class_totals['dram']
          dram_stall_cycles = class_totals['stall']
          total_bytes = sum(class_bytes.values())

          array_query_row = {
            'gemm_class': gemm_class_name,
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            # provenance: each model sits at its own long-context length, so the column
            # is not constant down the CSV
            'max_seq_len': max_seq_len,
            # frequency is a swept design axis (1000/2000 MHz); the GB/s columns are
            # per-frequency quantities, so consumers must slice or group on it.
            'frequency': frequency_mhz,
            # asram_size / wsram_size are the DECLARED capacities in bits; the model only
            # tiles against the active half (double buffering), reported alongside.
            # equal by selection (the diagonal filter above), kept as three columns so a
            # future de-coupled slice does not need a schema change
            'asram_size': isram_size,
            'wsram_size': wsram_size,
            'osram_size': osram_size,
            'isram_active_elements': isram_active_elements,
            'wsram_active_elements': wsram_active_elements,
            'active_fraction': isram_active_fraction,
            'input_data_moved': class_bytes['input'],
            'weight_data_moved': class_bytes['weight'],
            'output_read_data_moved': class_bytes['output_read'],
            'output_write_data_moved': class_bytes['output_write'],
            'output_data_moved': class_bytes['output_read'] + class_bytes['output_write'],
            'total_data_moved': total_bytes,
            # provenance: the two spans behind the window, and the on-chip stall the SRAM
            # ports impose on the array (window - array, zero when the array is critical)
            'array_cycle_count': array_cycles,
            'sram_cycle_count': sram_cycles,
            'window_cycle_count': window_cycles,
            'sram_stall_cycle_count': window_cycles - array_cycles,
            # what the DECLARED channel would take, and the wait it imposes on top of the
            # on-chip window: the consequence of the demand columns below
            'dram_cycle_count': dram_cycles,
            'dram_stall_cycle_count': dram_stall_cycles,
            # THE FIGURE'S QUANTITY: demand in GB/s, one column per operand -- ifmap,
            # wmap and ofmap, which fig_2.py plots as its three panels. The three share a
            # window (a GEMM advances at the pace of its slowest stream, so all three
            # streams hide under the same span) but are never pooled: each operand's bytes
            # respond to SRAM size differently -- ifmap to the N tile count, wmap to the M
            # sweep count, ofmap to the K folds -- which is the whole point of the panels.
            #
            # The column names are fig_2.py's schema. They read 'bandwidth' but carry a
            # DEMAND, not an achieved rate: the declared `dram: bandwidth:` never enters.
            'input_dram_bandwidth': required_bandwidth(class_bytes['input'], window_cycles, frequency_mhz),
            'weight_dram_bandwidth': required_bandwidth(class_bytes['weight'], window_cycles, frequency_mhz),
            'output_dram_bandwidth': required_bandwidth(
                class_bytes['output_read'] + class_bytes['output_write'], window_cycles, frequency_mhz),
            # provenance only: what a single shared channel would have to sustain
            'total_required_bandwidth': required_bandwidth(total_bytes, window_cycles, frequency_mhz),
          }

          array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    if not array_query_df.empty:
        # Sort ONCE, numerically, before anything is stringified. The scientific CSV used to
        # re-sort after formatting asram_size / wsram_size with '%.3e', which sorts those keys
        # LEXICOGRAPHICALLY - so the two CSVs disagreed on which config is row 1. Formatting a
        # copy of an already-sorted frame keeps them row-aligned.
        array_query_df = array_query_df.sort_values(
            by=['model', 'gemm_class', 'array_dim', 'batch_size', 'asram_size', 'wsram_size'])
        array_query_df.to_csv(output_path + f'bandwidth_performance_metrics.csv', index=False)

        array_query_df_sci = array_query_df.copy()
        for col in ['asram_size', 'wsram_size', 'osram_size',
                    'input_data_moved', 'weight_data_moved', 'output_read_data_moved',
                    'output_write_data_moved', 'output_data_moved', 'total_data_moved',
                    'array_cycle_count', 'sram_cycle_count', 'window_cycle_count',
                    'sram_stall_cycle_count', 'dram_cycle_count',
                    'dram_stall_cycle_count', 'input_dram_bandwidth',
                    'weight_dram_bandwidth', 'output_dram_bandwidth',
                    'total_required_bandwidth']:
            array_query_df_sci[col] = array_query_df_sci[col].apply(lambda x: f'{x:.3e}')
        array_query_df_sci.to_csv(output_path + f'bandwidth_performance_metrics_scientific.csv', index=False)
        print(f'wrote {len(array_query_df)} rows')
    else:
        print("Warning: No matching configurations found. No CSV saved.")
