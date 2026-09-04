from archx.metric import query_module_metric, aggregate_event_metric, aggregate_tag_metric, aggregate_event_count
from collections import OrderedDict
from archx.utils import get_prod, read_yaml
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
import statistics
import numpy as np
import os
import re
import sys
import pandas as pd
from loguru import logger

# ---- new-run-tree resolver + harvested-metrics fast path -------------------
# Runs live under zoo/mugi/designs/<design>/description/ and are indexed by that
# design's configurations.csv (the source of truth). load_yaml() still accepts
# the legacy path layout
#   <root>/<design>/<network>/<subarch>/<dim>/<model>/max_seq_len_N/batch_size_N[/kv_heads_N][/full_termination]
# and resolves it to the matching run directory. Points that no longer exist in
# the sweep raise MissingRunError (callers skip them).
#
# Metric access goes through zoo/mugi/results/metrics_harvest.csv (built by
# zoo.mugi.results.harvest, incrementally, one checkpoint open per run ever):
# load_yaml() returns a HarvestRow instead of a deserialized event graph, and
# the query_* helpers below read the pre-harvested scalars from it. Loading the
# real checkpoint is still available via load_checkpoint_yaml().

_DESIGN_ROOT = os.path.join('zoo', 'mugi', 'designs')
_DESIGNS = ('carat', 'mugi', 'simd', 'systolic', 'tensor')
_DIM_MODULE = 'adder'  # instance is [node_x, node_y, dim] in every design
_INDEX_CACHE = {}
_YAML_CACHE = {}
_HARVEST_ROWS = None  # run_path -> harvested row dict

class MissingRunError(Exception):
    pass

def _design_token_index(tokens):
    """Index of the design-name token in a split path, or None.

    Anchored on the 'designs' directory that precedes it, so an enclosing
    directory that happens to share a design's name (zoo/mugi vs the 'mugi'
    design) cannot shadow the real one. Falls back to a plain scan for paths
    that carry a design name without the 'designs' parent.
    """
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i - 1] == 'designs' and tokens[i] in _DESIGNS:
            return i
    for i, token in enumerate(tokens):
        if token in _DESIGNS:
            return i
    return None

class HarvestRow:
    """Stand-in for a loaded event graph: helpers read pre-harvested scalars.

    A missing column means the harvest manifest does not cover the requested
    aggregation (extend zoo/mugi/results/harvest.py); a NaN value means the
    aggregation raised at harvest time (absent tag/module), so reading it
    raises the same way the live aggregation would.
    """
    def __init__(self, row):
        self._row = row

    def get(self, key):
        if key not in self._row:
            raise KeyError(f'<{key}> is not in the harvest manifest; add it to '
                           f'zoo/mugi/results/harvest.py and re-run the harvest.')
        value = self._row[key]
        if pd.isna(value):
            raise ValueError(f'aggregation <{key}> failed at harvest time (absent tag/module).')
        return value

def _harvest_rows():
    global _HARVEST_ROWS
    if _HARVEST_ROWS is None:
        from zoo.mugi.results import harvest
        harvest.ensure()
        # round_trip: pandas' default float parser is off by up to 1 ulp, which
        # would leak last-digit drift into every query output
        harvest_df = pd.read_csv(harvest.HARVEST_PATH, low_memory=False, float_precision='round_trip')
        _HARVEST_ROWS = {row['run_path']: row for row in harvest_df.to_dict('records')}
    return _HARVEST_ROWS

def _harvest_row(run_path):
    rows = _harvest_rows()
    row = rows.get(run_path)
    mtime = os.stat(run_path + '/checkpoint.json').st_mtime_ns
    if row is None or row['checkpoint_mtime'] != mtime:
        # checkpoint changed since the table was built: re-extract this run in
        # memory (run `python -m zoo.mugi.results.harvest` to persist it)
        from zoo.mugi.results import harvest
        print(f'  [harvest] refreshing stale metrics for <{run_path}>.')
        tokens = [token for token in run_path.split(os.sep) if token]
        design = tokens[_design_token_index(tokens)]
        configurations_df = pd.read_csv(os.path.join(_DESIGN_ROOT, design, 'description', 'configurations.csv'))
        config_row = configurations_df[configurations_df['run_path'] == run_path].iloc[0]
        row = harvest.extract_row({'design': design, 'run_path': run_path,
                                   'checkpoint_path': config_row['checkpoint_path'],
                                   'metric_path': config_row['metric_path'],
                                   'arch_path': config_row['arch_path'],
                                   'work_path': config_row['work_path'],
                                   'workload_name': config_row['workload_name']})
        rows[run_path] = row
    return HarvestRow(row)

def _alias(name):
    # the new event graphs collapse the per-model root events into one 'llama_2' root
    if isinstance(name, str) and name.startswith('llama_2'):
        return 'llama_2'
    return name

def _read_yaml_cached(path):
    if path not in _YAML_CACHE:
        _YAML_CACHE[path] = read_yaml(path)
    return _YAML_CACHE[path]

def _dim_label(design, arch_modules):
    if design == 'tensor':
        return '8x16x16'
    dim = arch_modules[_DIM_MODULE]['instance'][2]
    return f'{dim}x8' if design in ('mugi', 'carat') else f'{dim}x{dim}'

def _network_label(arch_modules):
    node = arch_modules['isram']['instance']
    return 'single_node' if node[0] == 1 and node[1] == 1 else f'multi_node_{node[0]}x{node[1]}'

def _design_index(design):
    # the harvest table carries the resolver metadata, so indexing a design no
    # longer yaml-parses every workload/architecture config
    if design in _INDEX_CACHE:
        return _INDEX_CACHE[design]
    index_df = pd.DataFrame(list(_harvest_rows().values()))
    index_df = index_df[index_df['design'] == design][
        ['model', 'network', 'subarch', 'arch_dim', 'max_seq_len', 'batch_size', 'kv_heads', 'run_path']
    ].copy()
    index_df['subarch'] = index_df['subarch'].fillna('')
    _INDEX_CACHE[design] = index_df.reset_index(drop=True)
    return _INDEX_CACHE[design]

def _resolve_run(path):
    tokens = [token for token in os.path.normpath(path).split(os.sep) if token]
    design_idx = _design_token_index(tokens)
    assert design_idx is not None, f'No design name found in run path <{path}>.'
    design = tokens[design_idx]

    network = subarch = arch_dim = model = None
    max_seq_len = batch_size = kv_heads = None
    for token in tokens[design_idx + 1:]:
        if token == 'single_node' or token.startswith('multi_node'):
            network = token
        elif re.fullmatch(r'\d+x\d+(x\d+)?', token):
            arch_dim = token
        elif token.startswith('llama_2'):
            model = token
        elif token.startswith('max_seq_len_'):
            max_seq_len = int(token.rsplit('_', 1)[1])
        elif token.startswith('batch_size_'):
            batch_size = int(token.rsplit('_', 1)[1])
        elif token.startswith('kv_heads_'):
            kv_heads = int(token.rsplit('_', 1)[1])
        elif token.startswith('full_termination') or token.startswith('node_stationary'):
            pass
        else:
            subarch = token

    # the old GQA workload collapsed into llama_2_70b with kv_heads swept
    if model == 'llama_2_70b_GQA':
        model = 'llama_2_70b'
        kv_heads = 8 if kv_heads is None else kv_heads
    elif model == 'llama_2_70b' and kv_heads is None:
        kv_heads = 64

    match = _design_index(design)
    match = match[(match['model'] == model) & (match['network'] == network)]
    if subarch:
        match = match[match['subarch'] == subarch]
    if arch_dim is not None:
        match = match[match['arch_dim'] == arch_dim]
    if max_seq_len is not None:
        match = match[match['max_seq_len'] == max_seq_len]
    if batch_size is not None:
        match = match[match['batch_size'] == batch_size]
    if model == 'llama_2_70b' and kv_heads is not None:
        match = match[match['kv_heads'] == kv_heads]

    if len(match) == 0:
        print(f'  [skip] no run for <{path}>; point not in the generated sweep.')
        raise MissingRunError(path)
    assert len(match) == 1, f'Ambiguous run path <{path}>: {len(match)} matches.'
    return match.iloc[0]['run_path']

def geomean(dict_list: list[OrderedDict]) -> OrderedDict:
    geomean_dict = OrderedDict()

    for dict in dict_list:
        for key, value in dict.items():
            if key not in geomean_dict:
                if isinstance(value, str):
                    geomean_dict[key] = value
                else:
                    geomean_dict[key] = []

            if not isinstance(value, str):
                geomean_dict[key].append(value)

    for key, value in geomean_dict.items():
        if isinstance(value, list):
            geomean_dict[key] = statistics.geometric_mean(value)

    return geomean_dict

def query_cycle_count(event_graph, metric_dict, workload, event) -> OrderedDict:
    workload, event = _alias(workload), _alias(event)
    if isinstance(event_graph, HarvestRow):
        return event_graph.get(f'cycle_count|{workload}|{event}')
    cycle_count_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='cycle_count', workload=workload, event=event)
    return cycle_count_dict['value']

def query_execution_time(event_graph, metric_dict, workload, event) -> OrderedDict:
    workload, event = _alias(workload), _alias(event)
    if isinstance(event_graph, HarvestRow):
        return event_graph.get(f'runtime|{workload}|{event}') / 10**3 # ms -> s
    execution_time_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='runtime', workload=workload, event=event)
    return execution_time_dict['value'] / 10**3 # ms -> s

def query_dynamic_energy(event_graph, metric_dict, workload, event=None, tag=None) -> OrderedDict:
    workload, event = _alias(workload), _alias(event)
    if isinstance(event_graph, HarvestRow):
        key = f'dynamic_energy|{workload}|{event}' if tag is None else f'tag_dynamic_energy|{workload}|{tag}'
        return event_graph.get(key) / 10**9 # nJ -> J
    if tag is None:
        dynamic_energy_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='dynamic_energy', workload=workload, event=event)
    else:
        dynamic_energy_dict = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='dynamic_energy', workload=workload, tag=tag)
    return dynamic_energy_dict['value'] / 10**9 # nJ -> J

def query_dynamic_energy_carbon(event_graph, metric_dict, workload, event=None, tag=None) -> OrderedDict:
    workload, event = _alias(workload), _alias(event)
    if isinstance(event_graph, HarvestRow):
        # the tag branch historically aggregates with workload=event
        key = f'dynamic_energy|{workload}|{event}' if tag is None else f'tag_dynamic_energy|{event}|{tag}'
        return event_graph.get(key) / 10**9 # nJ -> J
    if tag is None:
        dynamic_energy_dict = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='dynamic_energy', workload=workload, event=event)
    else:
        dynamic_energy_dict = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='dynamic_energy', workload=event, tag=tag)
    return dynamic_energy_dict['value'] / 10**9 # nJ -> J


def query_leakage_power(event_graph, metric_dict, workload, event=None, tag=None) -> OrderedDict:
    workload, event = _alias(workload), _alias(event)
    if isinstance(event_graph, HarvestRow):
        # the tag branch historically aggregates with workload=event
        key = f'leakage_power|{workload}|{event}' if tag is None else f'tag_leakage_power|{event}|{tag}'
        return event_graph.get(key) / 10**3 # mW -> W
    if tag is None:
        leakage_power = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='leakage_power', workload=workload, event=event)
    else:
        leakage_power =  aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='leakage_power', workload=event, tag=tag)
    return leakage_power['value'] / 10**3 # mW -> W

def query_area(event_graph, metric_dict, workload=None, tag=None, module=None) -> np.float64:
    workload = _alias(workload)

    if isinstance(event_graph, HarvestRow):
        key = f'area_module|{workload}|{module}' if module is not None else f'tag_area|{workload}|{tag}'
        return event_graph.get(key)

    if module is not None:
        area = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric='area', workload=workload, event=module)['value']

    elif tag is not None:
        area = aggregate_tag_metric(event_graph=event_graph, metric_dict=metric_dict, metric='area', workload=workload, tag=tag)['value']

    return area

def _event_count(event_graph, workload, event):
    if isinstance(event_graph, HarvestRow):
        return event_graph.get(f'count|{workload}|{event}')
    return aggregate_event_count(event_graph=event_graph, workload=workload, event=event)

def query_operational_carbon(tag, event_graph, metric_dict, workload, event, CI) -> OrderedDict:
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
    dynamic_energy = query_dynamic_energy_carbon(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event, tag=tag)
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event, tag=tag)
    power = (leakage_power + (dynamic_energy / execution_time)) # W -> mW

    op_carbon = CI * (power / 1000) * (execution_time / 3600) # W -> KW, s -> H
    op_carbon /= 1000  # convert to kgCO2eq

    return op_carbon

def query_tag_power(tag, event_graph, metric_dict, workload, event) -> OrderedDict:
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=workload, event=event)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=workload, tag=tag)
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=workload, tag=tag)
    power = (leakage_power + (dynamic_energy / execution_time)) * 10**3 # W -> mW

    return power

def compute_throughput(performance_metrics_dict: OrderedDict) -> OrderedDict:

    flops = performance_metrics_dict['flops']
    execution_time = performance_metrics_dict['execution_time']
    throughput = flops/execution_time
    performance_metrics_dict['throughput'] = throughput

    return performance_metrics_dict

def compute_latency(performance_metrics_dict: OrderedDict) -> OrderedDict:

    flops = performance_metrics_dict['flops']

    execution_time = performance_metrics_dict['execution_time']

    latency = execution_time/flops

    performance_metrics_dict['latency'] = latency


    return performance_metrics_dict

def compute_throughput_efficiancy(performance_metrics_dict: OrderedDict) -> OrderedDict:

    flops = performance_metrics_dict['flops']
    energy = performance_metrics_dict['energy']
    power = performance_metrics_dict['power']
    execution_time = performance_metrics_dict['execution_time']

    throughput = flops/execution_time
    energy_efficiancy = throughput / (energy + (power * execution_time))
    power_efficiancy = throughput / (power + (energy / execution_time))
    
    performance_metrics_dict['throughput'] = throughput
    performance_metrics_dict['energy_efficiency'] = energy_efficiancy
    performance_metrics_dict['power_efficiency'] = power_efficiancy

    return performance_metrics_dict

def query_throughput_metrics(event_graph, metric_dict, module, workload, event)-> OrderedDict:
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = _event_count(event_graph, _alias(event), module)

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time
    })

    return performance_metrics_dict

def query_performance_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:
    # logger.add(sys.stdout, level="DEBUG")
    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    # logger.remove()
    cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    
    pe_count = _event_count(event_graph, _alias(event), module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
        'power': leakage_power,
        'cycle_count': cycle_count,
    })

    return performance_metrics_dict

def query_throughput_energy_metrics_workload(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = _event_count(event_graph, _alias(event), module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
    })

    return performance_metrics_dict

def query_throughput_energy_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = _event_count(event_graph, _alias(event), module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
    })

    return performance_metrics_dict

def query_performance_gemm_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = _event_count(event_graph, _alias(event), module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')

    flops = pe_count * 2 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
        'power': leakage_power,
    })

    return performance_metrics_dict

def query_performance_nonlinear_metrics(event_graph, metric_dict, module, workload, event) -> OrderedDict:

    execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)
    pe_count = _event_count(event_graph, _alias(event), module)
    dynamic_energy = query_dynamic_energy(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    leakage_power = query_leakage_power(event_graph=event_graph, metric_dict=metric_dict, workload=event, tag='onchip')
    cycle_count = query_cycle_count(event_graph=event_graph, metric_dict=metric_dict, workload=event, event=event)

    flops = pe_count * 3 / 10**9 # GFLOPS

    performance_metrics_dict = OrderedDict({
        'flops': flops,
        'execution_time': execution_time,
        'energy': dynamic_energy,
        'power': leakage_power,
        'cycle_count': cycle_count,
    })

    return performance_metrics_dict

def load_yaml(path):
    # fast path: metric access is served from the harvest table, so nothing is
    # deserialized here; the query helpers accept the HarvestRow as event_graph
    run_path = _resolve_run(path)
    yaml_dict = OrderedDict({
        'architecture_dict': None,
        'event_graph': _harvest_row(run_path),
        'metric_dict': None,
        'event_dict': None,
    })

    return yaml_dict

def load_checkpoint_yaml(path):
    # full loader (deserializes the populated A-Graph); for ad-hoc inspection
    run_path = _resolve_run(path)
    yaml_dict = OrderedDict({
        'architecture_dict': load_architecture_dict(run_path + '/architecture.yaml'),
        'event_graph': load_event_graph(run_path + '/checkpoint.json'),
        'metric_dict': load_metric_dict(run_path + '/metric.yaml'),
        'event_dict': read_yaml(run_path + '/event.yaml'),
    })

    return yaml_dict