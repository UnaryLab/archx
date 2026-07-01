import importlib.util
import sys
from collections import OrderedDict
from loguru import logger

from archx._core import ArchxGraph
from archx.utils import get_path


key_aggregation = 'aggregation'
key_operation = 'operation'
legal_specified = ['parallel', 'sequential']
key_value = 'value'
key_unit = 'unit'
key_subevent = 'subevent'
key_count = 'count'
key_factor = 'factor'
_function_cache = {}


def import_function_from_path(file_path: str, function: str) -> callable:
    full_path = get_path(file_path)
    cache_key = (full_path, function)
    if cache_key in _function_cache:
        return _function_cache[cache_key]

    spec = importlib.util.spec_from_file_location(function, full_path)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    if hasattr(module_py, function) and callable(getattr(module_py, function)):
        _function_cache[cache_key] = getattr(module_py, function)
        return _function_cache[cache_key]
    else:
        logger.error(f'Invalid function <{function}> at <{full_path}>.')
        exit()


def simulate_performance_one_event(
    event_graph: ArchxGraph,
    architecture_dict: OrderedDict,
    workload_dict: OrderedDict,
    event_name: str,
) -> ArchxGraph:
    """Run the performance model for a single event node and update graph edges."""

    assert event_graph.has_node(event_name), \
        logger.error(f'Invalid event <{event_name}>.')

    performance_path = event_graph.get_performance_path(event_name)

    if performance_path is None or performance_path == 'None':
        assert event_graph.is_leaf(event_name), \
            logger.error(f'Missing performance model for event <{event_name}>.')
        logger.info(f'Module <{event_name}> has no performance model.')
        return event_graph

    # Load and execute the performance model
    performance_model = import_function_from_path(performance_path, function=event_name)
    performance_dict = performance_model(
        architecture_dict=architecture_dict,
        workload_dict=workload_dict,
    )
    assert performance_dict is not None, \
        logger.error(f'No performance model returned for event <{event_name}>')

    # Handle extra metrics returned by performance models (used in 'specified' mode).
    # These are stored as specified_metrics on the node.
    if len(performance_dict) > 1:
        for metric_key in performance_dict:
            if metric_key == key_subevent:
                continue
            val = performance_dict[metric_key]
            event_graph.set_node_specified_metric(
                event_name,
                metric_key,
                float(val[key_value]),
                str(val[key_unit]),
            )

    # Update edge properties for each out-neighbour
    for target_name in event_graph.get_out_neighbors(event_name):
        assert target_name in performance_dict[key_subevent], \
            logger.error(f'  Missing subevent <{target_name}> in the '
                         f'<{event_name}> performance model.')

        subevent_cfg = performance_dict[key_subevent][target_name]

        # count
        if key_count not in subevent_cfg:
            event_graph.set_edge_count(event_name, target_name, 1.0)
            logger.warning(f'  Missing count in subevent <{target_name}>; default to 1.')
        else:
            event_graph.set_edge_count(event_name, target_name,
                                       float(subevent_cfg[key_count]))

        # aggregation
        if key_aggregation in subevent_cfg:
            agg = subevent_cfg[key_aggregation].lower()
            assert agg in legal_specified, \
                logger.error(f'  Invalid aggregation <{subevent_cfg[key_aggregation]}> in the performance model of '
                             f'event <{event_name}> at <{performance_path}>; '
                             f'legal values: {legal_specified}.')
            event_graph.set_edge_aggregation(event_name, target_name, agg)
            if event_graph.is_leaf(target_name):
                logger.warning(f'  Ignore aggregation <{agg}> between event '
                               f'<{event_name}> and module <{target_name}>; '
                               f'aggregation only takes effect between events.')
        else:
            logger.warning(f'  Missing aggregation in subevent <{target_name}>; '
                           f'default to <parallel>.')
            event_graph.set_edge_aggregation(event_name, target_name, 'parallel')

        # operation (only valid for module targets)
        if key_operation in subevent_cfg:
            assert event_graph.is_leaf(target_name), \
                logger.error(f'  Invalid operation between event <{event_name}> '
                             f'and event <{target_name}>; operation should be '
                             f'between event and module.')
            event_graph.set_edge_operation(event_name, target_name,
                                           dict(subevent_cfg[key_operation]))

        # factor
        if key_factor in subevent_cfg:
            factor = subevent_cfg[key_factor]
            assert isinstance(factor, dict), \
                logger.error(f'  Invalid factor <{factor}> between event <{event_name}> '
                             f'and module <{target_name}>; factor should be a dict.')
            event_graph.set_edge_factor(event_name, target_name,
                                        {k: float(v) for k, v in factor.items()})

        edge_count = event_graph.get_edge_count(event_name, target_name)
        edge_agg = event_graph.get_edge_aggregation(event_name, target_name)
        logger.debug(f'  Event <{event_name}> has <{edge_count}> subevent '
                     f'<{target_name}> with specified aggregation <{edge_agg}>.')

    logger.success(f'Simulate event <{event_name}> at <{performance_path}>.')
    return event_graph


def simulate_performance_all_events(
    event_graph: ArchxGraph,
    architecture_dict: OrderedDict,
    workload_dict: OrderedDict,
) -> ArchxGraph:
    for event_name in event_graph.get_all_node_names():
        event_graph = simulate_performance_one_event(
            event_graph, architecture_dict, workload_dict, event_name)

    logger.success('Simulate all events.')
    return event_graph
