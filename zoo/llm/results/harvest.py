# ---- one-pass metrics harvest ----------------------------------------------
# Extracts, once per run, every scalar the result queries aggregate from a
# checkpoint (the manifest below mirrors the exact (workload, event/tag/module)
# pairs used by zoo.llm.results.query.utils helpers) into a single table:
#   zoo/llm/results/metrics_harvest.csv
# The query helpers then read this table instead of deserializing checkpoints.
# The harvest is incremental: only runs whose checkpoint mtime changed (or new
# runs) are re-extracted, and rows for deleted runs are dropped.
#
#   python -m zoo.llm.results.harvest      # build/refresh the table in parallel
#
# figure_generation and utils.load_yaml() call ensure() automatically, so the
# harvest self-refreshes; running it standalone just pays the cost up front.
from collections import OrderedDict
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST_PATH = os.path.join(_HERE, 'metrics_harvest.csv')

# resolver metadata columns (utils._design_index reads these back from the table)
META_COLUMNS = ['design', 'model', 'network', 'subarch', 'arch_dim',
                'max_seq_len', 'batch_size', 'kv_heads', 'run_path', 'checkpoint_mtime']

# the event scopes and proxy modules the queries reference
SCOPES = ['gemm', 'nonlinear', 'projection', 'attention', 'ffn', 'softmax', 'silu']
BREAKDOWN_TAGS = ['accumulator', 'fifo', 'pe', 'nonlinear', 'vector', 'tc', 'value_reuse',
                  'array', 'node_memory', 'router']
COUNT_MODULES = ['multiplier', 'and_gate', 'register_vector', 'accumulator_vector',
                 'magnitude_register', 'adder_vector']


def manifest():
    """key -> (kind, metric, workload, target); mirrors the utils helper call sites."""
    jobs = OrderedDict()
    runtime_pairs = ([('llama_2', 'llama_2')]
                     + [('llama_2', s) for s in ['projection', 'attention', 'ffn', 'nonlinear']]
                     + [(s, s) for s in SCOPES])
    for w, e in runtime_pairs:
        jobs[f'runtime|{w}|{e}'] = ('event', 'runtime', w, e)
    for w, e in [('llama_2', 'llama_2')] + [(s, s) for s in SCOPES]:
        jobs[f'cycle_count|{w}|{e}'] = ('event', 'cycle_count', w, e)
    for t in BREAKDOWN_TAGS:
        jobs[f'tag_dynamic_energy|llama_2|{t}'] = ('tag', 'dynamic_energy', 'llama_2', t)
        jobs[f'tag_leakage_power|None|{t}'] = ('tag', 'leakage_power', None, t)
        jobs[f'tag_area|llama_2|{t}'] = ('tag', 'area', 'llama_2', t)
    for w in ['llama_2', None] + SCOPES:
        jobs[f'tag_dynamic_energy|{w}|onchip'] = ('tag', 'dynamic_energy', w, 'onchip')
        jobs[f'tag_leakage_power|{w}|onchip'] = ('tag', 'leakage_power', w, 'onchip')
    jobs['tag_area|llama_2|onchip'] = ('tag', 'area', 'llama_2', 'onchip')
    jobs['tag_area|None|onchip'] = ('tag', 'area', None, 'onchip')
    jobs['area_module|llama_2|pe_fifo'] = ('event', 'area', 'llama_2', 'pe_fifo')
    for s in SCOPES:
        for m in COUNT_MODULES:
            jobs[f'count|{s}|{m}'] = ('count', None, s, m)
    return jobs


_WORKER_METRIC_CACHE = {}


def _metric_dict(metric_path):
    from archx.metric import load_metric_dict
    if metric_path not in _WORKER_METRIC_CACHE:
        _WORKER_METRIC_CACHE[metric_path] = load_metric_dict(metric_path)
    return _WORKER_METRIC_CACHE[metric_path]


def extract_row(job):
    """Harvest one run: meta columns + every manifest scalar (NaN where the
    aggregation raises, e.g. a tag/module the design does not have)."""
    from archx.event import load_event_graph
    from archx.metric import aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
    from zoo.llm.results.query import utils

    workload_config = utils._read_yaml_cached(job['work_path'])['workload']['configuration']
    arch_modules = utils._read_yaml_cached(job['arch_path'])['architecture']['module']

    row = {
        'design': job['design'],
        'model': job['workload_name'],
        'network': utils._network_label(arch_modules),
        'subarch': workload_config.get('subarch', ''),
        'arch_dim': utils._dim_label(job['design'], arch_modules),
        'max_seq_len': workload_config['max_seq_len'],
        'batch_size': workload_config['batch_size'],
        'kv_heads': workload_config.get('kv_heads'),
        'run_path': job['run_path'],
        'checkpoint_mtime': os.stat(job['checkpoint_path']).st_mtime_ns,
    }

    event_graph = load_event_graph(job['checkpoint_path'])
    metric_dict = _metric_dict(job['metric_path'])
    for key, (kind, metric, workload, target) in manifest().items():
        try:
            if kind == 'event':
                value = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict,
                                               metric=metric, workload=workload, event=target)['value']
            elif kind == 'tag':
                value = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict,
                                             metric=metric, workload=workload, tag=target)['value']
            else:  # count
                value = aggregate_event_count(event_graph=event_graph, workload=workload, event=target)
        except Exception:
            value = float('nan')
        row[key] = value
    return row


def _pending_jobs():
    """(kept_rows, jobs): existing fresh rows to keep, and runs needing harvest."""
    from zoo.llm.results.query.utils import _DESIGN_ROOT, _DESIGNS

    existing = {}
    if os.path.exists(HARVEST_PATH):
        # round_trip: kept rows are re-written verbatim, so parsing must be exact
        # (the default C parser is off by up to 1 ulp)
        for row in pd.read_csv(HARVEST_PATH, low_memory=False, float_precision='round_trip').to_dict('records'):
            existing[row['run_path']] = row

    kept, jobs = [], []
    for design in _DESIGNS:
        configurations_path = os.path.join(_DESIGN_ROOT, design, 'description', 'configurations.csv')
        if not os.path.exists(configurations_path):
            continue
        for row in pd.read_csv(configurations_path).to_dict('records'):
            run_path = row['run_path']
            checkpoint_path = row['checkpoint_path']
            try:
                mtime = os.stat(checkpoint_path).st_mtime_ns
            except OSError:
                continue  # run not executed yet
            old = existing.get(run_path)
            if old is not None and old['checkpoint_mtime'] == mtime:
                kept.append(old)
            else:
                jobs.append({'design': design, 'run_path': run_path, 'checkpoint_path': checkpoint_path,
                             'metric_path': row['metric_path'], 'arch_path': row['arch_path'],
                             'work_path': row['work_path'], 'workload_name': row['workload_name']})
    return kept, jobs


def ensure(verbose=True):
    """Bring metrics_harvest.csv up to date; near-instant when nothing changed.

    Silences the loguru default sink: the per-aggregation DEBUG logging would
    otherwise emit millions of lines while extracting."""
    from loguru import logger
    logger.remove()
    kept, jobs = _pending_jobs()
    total_expected = len(kept) + len(jobs)
    if not jobs:
        if os.path.exists(HARVEST_PATH) and len(pd.read_csv(HARVEST_PATH, low_memory=False)) == total_expected:
            return HARVEST_PATH
    if verbose and jobs:
        print(f'  [harvest] extracting metrics for {len(jobs)} runs ({len(kept)} already fresh).')
    rows = list(kept)
    if jobs:
        workers = max(1, (os.cpu_count() or 2) - 2)
        with multiprocessing.get_context('fork').Pool(workers) as pool:
            rows.extend(tqdm(pool.imap_unordered(extract_row, jobs, chunksize=8),
                             total=len(jobs), desc='Harvest', unit='run',
                             dynamic_ncols=True, disable=not verbose))
    df = pd.DataFrame(rows).sort_values('run_path').reset_index(drop=True)
    df = df[META_COLUMNS + list(manifest().keys())]
    df.to_csv(HARVEST_PATH, index=False, float_format='%.17g')  # 17 sig digits: exact float64 round-trip
    if verbose and jobs:
        print(f'  [harvest] wrote {len(df)} rows to {os.path.relpath(HARVEST_PATH)}.')
    return HARVEST_PATH


if __name__ == '__main__':
    ensure()
