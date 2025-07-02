import importlib, sys
import graph_tool.all as gt

from collections import OrderedDict
from loguru import logger

from archx.utils import get_path


key_aggregation = 'aggregation'
key_operation = 'operation'
legal_specified = ['parallel', 'sequential']
key_value = 'value'
key_unit = 'unit'
key_subevent = 'subevent'
key_count = 'count'
key_factor = 'factor'


def import_function_from_path(file_path: str, function: str) -> callable:
    full_path = get_path(file_path)
    
    spec = importlib.util.spec_from_file_location(function, full_path)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    # get the specified function dynamically
    if hasattr(module_py, function) and callable(getattr(module_py, function)):
        target_function = getattr(module_py, function)
        return target_function
    else:
        logger.error(f'Invalid function <{function}> at <{full_path}>.')
        exit()
    

def simulate_performance_one_event(event_graph: gt.Graph, architecture_dict: OrderedDict, workload_dict: OrderedDict, event_name: str) -> gt.Graph:
    # run performance model for a single event node
    v = gt.find_vertex(event_graph, event_graph.vp.event, event_name)
    assert len(v) > 0, logger.error(f'Invalid event <{event_name}>.')

    v = v[0]
    performance_path = event_graph.vp.performance[v]
    if performance_path is None or performance_path == 'None':
        assert v.out_degree() == 0, logger.error(f'Missing performance model for event <{event_name}>.')
        logger.info(f'Module <{event_name}> has no performance model.')
    else:
        # if the current node is not a leaf node, update edges with performance model
        performance_model = import_function_from_path(performance_path, function=event_name)
        performance_dict = performance_model(architecture_dict=architecture_dict, workload_dict=workload_dict)
        assert performance_dict is not None, logger.error(f'No performance model returned for event <{event_name}>')
        # process additional metrics in specified mode
        if len(list(performance_dict.keys())) > 1:
            metric_key_list = list(performance_dict.keys())
            metric_key_list.remove(key_subevent)
            for metric_key in metric_key_list:
                if metric_key not in event_graph.vp:
                    event_graph.vp[metric_key] = event_graph.new_vertex_property('object')
                event_graph.vp[metric_key][v] = OrderedDict({key_value: performance_dict[metric_key][key_value], key_unit: performance_dict[metric_key][key_unit]})

        # iterate over all out-edges
        for e in v.out_edges():
            edge_source = event_graph.vp.event[e.source()]
            edge_target = event_graph.vp.event[e.target()]

            assert edge_target in performance_dict[key_subevent], logger.error(f'  Missing subevent <{edge_target}> in the <{edge_source}> performance model.')
            if key_count not in performance_dict[key_subevent][edge_target]:
                event_graph.ep.count[e] = 1.
                logger.warning(f'  Missing count in subevent <{edge_target}>; default to 1.')
            else:
                event_graph.ep.count[e] = performance_dict[key_subevent][edge_target][key_count]

            # default specified aggregation is parallel, which will be reduced with maximum
            if key_aggregation in performance_dict[key_subevent][edge_target]:
                assert performance_dict[key_subevent][edge_target][key_aggregation].lower() in legal_specified, logger.error(f'  Invalid aggregation <{performance_dict["subevent"][edge_target][key_aggregation]}> in the performance model of event <{event_name}> at <{performance_path}>; legal values: {legal_specified}.')
                event_graph.ep.aggregation[e] = performance_dict[key_subevent][edge_target][key_aggregation].lower()
                if e.target().out_degree() == 0:
                    logger.warning(f'  Ignore aggregation <{event_graph.ep.aggregation[e]}> between event <{edge_source}> and module <{edge_target}>; aggregation only takes effect between events.')
            else:
                logger.warning(f'  Missing aggregation in subevent <{edge_target}>; default to <parallel>.')
                event_graph.ep.aggregation[e] = 'parallel'

            # default operation is empty, in which case, query is a direct dictionary lookup
            if key_operation in performance_dict[key_subevent][edge_target]:
                assert e.target().out_degree() == 0, logger.error(f'  Invalid operation between event <{event_name}> and event <{event_graph.vp.event[e.target()]}>; operation should be between event and module.')
                event_graph.ep.operation[e] = performance_dict[key_subevent][edge_target][key_operation]
            
            # default factor is an empty dict, which is used to scale queried results
            if key_factor in performance_dict[key_subevent][edge_target]:
                event_graph.ep.factor[e] = performance_dict[key_subevent][edge_target][key_factor]
                assert isinstance(event_graph.ep.factor[e], dict), logger.error(f'  Invalid factor <{event_graph.ep.factor[e]}> between event <{event_name}> and module <{event_graph.vp.event[e.target()]}>; factor should be a dict.')

            logger.debug(f'  Event <{edge_source}> has <{event_graph.ep.count[e]}> subevent <{edge_target}> with specified aggregation <{event_graph.ep.aggregation[e]}>.')

        logger.success(f'Simulate event <{event_name}> at <{performance_path}>.')

    return event_graph


def simulate_performance_all_events(event_graph: gt.Graph, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> gt.Graph:
    # Iterate over all event nodes
    for v in event_graph.vertices():
        event_name = event_graph.vp.event[v]
        event_graph = simulate_performance_one_event(event_graph, architecture_dict, workload_dict, event_name)
    
    logger.success(f'Simulate all events.')
    
    return event_graph

