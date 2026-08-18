"""Bandwidth query for fig_2.

WARNING: this is a script, not a module. Importing it re-runs the whole sweep and
overwrites results/csv/bandwidth_performance_metrics{,_scientific}.csv.

SCOPE OF THE SWEEP: of the 4,960 configurations in configurations.csv this script keeps
only `array_dim == [512, 512]` (see the `continue` guard below), leaving 4,000 rows
across all five batch sizes and both swept frequencies (1000/2000 MHz; the CSV carries a
`frequency` column and fig_2 plots the 1000 MHz slice). Every number in the CSV and in
fig_2 is conditional on that slice; it is not a sweep over array size. All batches are
kept because the all-or-nothing input law's fit thresholds (SRAM = 2x working set) are
crossed at different SRAM sizes by different batches -- a single-batch slice would hide
the staircase fig_2's input panel exists to show.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from loguru import logger
from chiplet4ai.results.query.utils import query_execution_time, query_cycle_count
from chiplet4ai.common.performance.mapping import _active_fraction, _architecture, _buffer_elements
from chiplet4ai.designs.llama import model as llama_model
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

# The active-buffer definition is mapping.py's (`_active_fraction` / `_buffer_elements`),
# imported rather than re-derived here so the query credits exactly the capacity the model
# does. node.py imports the same helpers; mapping.py itself imports only
# common.performance.utils, so there is no cycle.

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
    return {
        'data_moved': data_moved,
        'transfer_window_cycles': transfer_window,
        'active_avg_bandwidth': active_avg,
    }

def collect_mapping_samples(event_graph, architecture_dict, workload_dict, workload_name, event_suffix, movement_specs, frequency_mhz):
    samples = {name: [] for name in movement_specs}

    for event in sorted(event_graph.get_all_node_names()):
        if not event.endswith(event_suffix):
            continue
        if not hasattr(llama_model, event):
            # Expected for the stall events (they live in `mapping`, not `llama_model`);
            # anything else here is a real gap between the graph and the model.
            warn(f'{event}: no matching llama_model function, sample dropped')
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
                warn(f'{event}: no {spec["event"]} subevent, contributes nothing to {name}')
                continue

            count = float(subevent.get('count', 0)) * event_count
            data_moved = count * spec.get('count_to_bytes', 1)

            factor = subevent.get('factor', {})
            if 'bandwidth' not in factor:
                # A missing factor used to default to 0, which zeroes transfer_window_cycles
                # while data_moved still accumulates - that INFLATES the harmonic mean.
                # Drop the sample loudly instead of contributing a silent zero.
                warn(f'{event}: {spec["event"]} has no bandwidth factor; '
                     f'dropping {data_moved:.6g} B from the {name} aggregate')
                continue
            bandwidth = float(factor['bandwidth'])
            if data_moved > 0 and bandwidth <= 0:
                warn(f'{event}: {spec["event"]} bandwidth is {bandwidth} with {data_moved:.6g} B '
                     f'moved; dropping the sample from the {name} aggregate')
                continue
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

        # mapping.py normalizes through `_architecture` before touching any module
        # (`_cycle_event`, `_ws_schedule` and `gemm` each open with that call), so the query
        # normalizes too and matches mapping.py's own call style. This is a style match, not
        # a correctness requirement: node.py (`_active_elements`) passes the RAW architecture
        # dict to `_buffer_elements` and is equally correct, because for a flat architecture
        # dict `_architecture` RE-EXPORTS THE SAME 'isram'/'wsram'/'osram'/'dram'
        # sub-objects it found there. Both call styles therefore index identical modules.
        arch = _architecture(architecture_dict)
        isram_active_elements = _buffer_elements(arch, 'isram')[0]
        wsram_active_elements = _buffer_elements(arch, 'wsram')[0]
        isram_active_fraction = _active_fraction(arch['isram'])

        batch_size = workload_dict['configuration']['batch_size']
        workload_name = workload_dict['name']

        if architecture_dict['pe']['query']['frequency'] != 1000:
            continue

        # The workload root fans out to `llama` and `llama_array` and takes the larger of
        # the two. Reading the row's execution_time off `llama` (below) is only honest while
        # `llama` IS the critical branch; if `llama_array` ever overtakes it the root
        # silently switches and the row would understate the runtime. Assert it loudly, and
        # do it BEFORE the sweep filter so all 4,960 configs are checked, not just those
        # that reach the CSV.
        #
        # EXACT equality, no tolerance. The root aggregates by `max`
        # (the `sequential_acc` / `parallel_max` aggregation match in aggregate.rs; cited by
        # symbol, not by line, because no Python-side refactor can keep a Rust line number in
        # sync). Three facts TOGETHER make
        # the root bit-identical to one operand rather than an arithmetic blend of both
        # (llama_model.llama_model's subevent dict):
        #   1. BOTH root edges are `aggregation: parallel`, so nothing accumulates
        #      sequentially - `max` selects, it does not add.
        #   2. Both carry `count == 1`, so neither operand is scaled on the way in.
        #   3. Neither carries a `cycle_count` factor (`llama_array`'s only factor is
        #      `dynamic_energy: 0.0`), so neither operand is reweighted.
        # If ANY ONE of the three failed, the root could differ from `llama` by arithmetic
        # rather than by selection and an exact `!=` would be the wrong test. As it stands
        # the root is either bit-identical to `llama` or is `llama_array`'s structurally
        # different value; there is no summation-order path to a one-ULP difference here.
        #
        # The 2e-16 figure quoted elsewhere describes query_cycle_breakdown's
        # `unattributed_cycle_count`, NOT this check. It is an observed maximum RELATIVE
        # residual (`|residual|/total <= 2.0e-16` over a 40-run sample; nonzero on 11 of
        # them, always negative; exactly 0.0 on the audited config), and it must be read
        # relatively: it is a ratio, not an absolute cycle count, and at the 1e9-1e12 cycle
        # totals seen here it is ~1e-4 cycles in absolute terms. Only under that relative
        # reading is it below one ULP of a double. `unattributed_cycle_count` is a raw
        # total - compute - sram - dram subtraction, with NO tolerance constant coded
        # anywhere in that function, so the number is a measurement of that SUM
        # decomposition's residual, where ULP drift is real - not a threshold. It belongs
        # ONLY to that breakdown and has never applied to the `root == llama` check above.
        # Do not conflate the two.
        root_cycles = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                        workload=workload_name, event=workload_name)
        llama_cycles = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                         workload=workload_name, event='llama')
        if root_cycles != llama_cycles:
            # Only queried on the failure path. `llama_array`'s cycle_count is the first
            # number a diagnoser wants: if it equals root_cycles the root simply selected
            # the other operand (the expected way this fires), and if it equals neither,
            # the failure is in the aggregation itself, not in which branch is critical.
            llama_array_cycles = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict,
                                                   workload=workload_name, event='llama_array')
            raise AssertionError(
                f'{run_path}: workload root {workload_name!r} cycle_count {root_cycles!r} != '
                f'llama cycle_count {llama_cycles!r} '
                f'(llama_array cycle_count {llama_array_cycles!r}); '
                f'the critical branch is no longer `llama`, so execution_time is not the runtime'
            )

        # SWEEP FILTER: keep the 512x512 slice, all batch sizes and both frequencies
        # (4,000 of 4,960 configs). Everything downstream, including fig_2, is
        # conditional on this slice.
        if array_dim != [512, 512]:
            continue

        # Execution time comes from the `llama` branch - the same branch the *_dram
        # bandwidth samples below are drawn from, and the one that carries stalls.
        # (The old `llama_array` branch fans out to *_arr nodes only and is stall-free
        # by construction, so a row mixed a stalled quantity with an unstalled one.)
        execution_time = query_execution_time(
            event_graph=event_graph,
            metric_dict=metric_dict,
            workload=workload_name,
            event='llama'
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
                # Output movement is write-through (scalesim_audit.md Resolved 14):
                # dram_output_write carries every osram write (finals + partial-sum
                # updates, ~M*N*k_folds), independent of osram size; dram_output_read is
                # structurally 0 and is queried only as provenance that it stays 0. Both
                # counts are bytes, like the input/weight events.
                'output_read': {'event': 'dram_output_read'},
                'output_write': {'event': 'dram_output_write'},
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
                # `count` on sram_*_write_mapping is a buffer-FILL count, not a word count
                # (mapping.py: input_read_elements / isram_active_elements), so one count is
                # one whole active buffer: active_elements words of `width` bits each.
                'input': {'event': 'sram_input_write_mapping',
                          'count_to_bytes': isram_active_elements * isram_width / 8},
                'weight': {'event': 'sram_weight_write_mapping',
                           'count_to_bytes': wsram_active_elements * wsram_width / 8},
            },
            frequency_mhz=frequency_mhz,
        )

        # Every bandwidth window is the same quantity: the compute span the transfer has to
        # hide under. The four per-operand / per-level windows the CSV used to carry were
        # numerically identical, so they collapse to one column. Keeping the row free of any
        # stalled quantity: after the collapse, `execution_time` is the only stalled number
        # in it. (No self-check here: all four values come from one literal expression, so
        # there is no reachable state in which they disagree. The assert responsibility
        # lives in mapping.py.)
        windows = {
            'input_dram': dram_summary['input']['transfer_window_cycles'],
            'weight_dram': dram_summary['weight']['transfer_window_cycles'],
            'input_sram': sram_summary['input']['transfer_window_cycles'],
            'weight_sram': sram_summary['weight']['transfer_window_cycles'],
        }
        demand_window_cycles = max(windows.values())

        # COMBINED output demand rate, one column. Under write-through the read stream is
        # structurally 0, so the combined rate equals the write rate; the combine is kept
        # (rather than dropping the read term) so a regression that reintroduces read
        # bytes surfaces in the CSV instead of vanishing. Read and write occupy the SAME
        # compute span, so the summarize_bandwidth idiom (sum of per-stream windows) must
        # not be applied across the two streams - it would count the shared span twice and
        # halve the rate. dram_output_write has nonzero bytes in every GEMM (final writes),
        # so its summed window already covers the full span; pooled bytes over that window
        # is the honest combined rate. Split byte totals are kept as provenance columns.
        output_read_moved = dram_summary['output_read']['data_moved']
        output_write_moved = dram_summary['output_write']['data_moved']
        output_window = dram_summary['output_write']['transfer_window_cycles'] \
            or dram_summary['output_read']['transfer_window_cycles']
        output_dram_bandwidth = bandwidth_from_window_cycles(
            output_read_moved + output_write_moved, output_window, frequency_mhz)

        array_query_row = {
            'model': workload_name,
            'array_dim': f'{array_dim[0]}x{array_dim[1]}',
            'batch_size': batch_size,
            # frequency is a swept design axis (1000/2000 MHz); the GiB/s columns are
            # per-frequency quantities, so consumers must slice or group on it.
            'frequency': frequency_mhz,
            # asram_size / wsram_size are the DECLARED capacities in bits; the model only
            # credits `active_fraction` of them with reuse (double buffering).
            'asram_size': isram_bank * isram_depth * isram_width,
            'wsram_size': wsram_bank * wsram_depth * wsram_width,
            'osram_size': osram_bank * osram_depth * osram_width,
            'active_fraction': isram_active_fraction,
            'execution_time': execution_time,
            'input_data_moved': dram_summary['input']['data_moved'],
            'weight_data_moved': dram_summary['weight']['data_moved'],
            'output_read_data_moved': output_read_moved,
            'output_write_data_moved': output_write_moved,
            'output_data_moved': output_read_moved + output_write_moved,
            # provenance column: the compute-span demand window behind the bandwidth columns.
            'demand_window_cycles': demand_window_cycles,
            # The SRAM bandwidths are dropped, not omitted by accident: an operand's SRAM
            # fill bytes equal its DRAM read bytes over this same window, so the SRAM
            # columns were structural duplicates of the DRAM ones (weight bit-identical,
            # input equal to 3e-16).
            'input_dram_bandwidth': dram_summary['input']['active_avg_bandwidth'],
            'weight_dram_bandwidth': dram_summary['weight']['active_avg_bandwidth'],
            'output_dram_bandwidth': output_dram_bandwidth,
        }

        array_query_df = pd.concat([array_query_df, pd.DataFrame([array_query_row])], ignore_index=True) if not array_query_df.empty else pd.DataFrame([array_query_row])

    if not array_query_df.empty:
        # Sort ONCE, numerically, before anything is stringified. The scientific CSV used to
        # re-sort after formatting asram_size / wsram_size with '%.3e', which sorts those keys
        # LEXICOGRAPHICALLY - so the two CSVs disagreed on which config is row 1. Formatting a
        # copy of an already-sorted frame keeps them row-aligned.
        array_query_df = array_query_df.sort_values(
            by=['model', 'array_dim', 'batch_size', 'asram_size', 'wsram_size'])
        array_query_df.to_csv(output_path + f'bandwidth_performance_metrics.csv', index=False)

        array_query_df_sci = array_query_df.copy()
        for col in ['asram_size', 'wsram_size', 'osram_size', 'execution_time',
                    'input_data_moved', 'weight_data_moved', 'output_read_data_moved',
                    'output_write_data_moved', 'output_data_moved', 'demand_window_cycles',
                    'input_dram_bandwidth', 'weight_dram_bandwidth', 'output_dram_bandwidth']:
            array_query_df_sci[col] = array_query_df_sci[col].apply(lambda x: f'{x:.3e}')
        array_query_df_sci.to_csv(output_path + f'bandwidth_performance_metrics_scientific.csv', index=False)
        print(f'wrote {len(array_query_df)} rows')
    else:
        print("Warning: No matching configurations found. No CSV saved.")
