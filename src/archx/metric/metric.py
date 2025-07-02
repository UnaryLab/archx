import graph_tool.all as gt
from collections import OrderedDict

from loguru import logger

from archx.interface import query_interface
from archx.utils import read_yaml, write_yaml, get_path, create_dir, get_prod


key_value = 'value'
key_unit = 'unit'
legal_aggregation = ['module', 'summation', 'specified']
legal_aggregation_tag = ['module', 'summation']
key_metric = 'metric'
key_instance = 'instance'
key_factor = 'factor'
key_aggregation = 'aggregation'
single_op_metric_format = '{\'' + key_value + '\': ' + 'float' + ', \'' + key_unit + '\': ' + 'str' + '}'


def create_metric_dict(metric_file: str) -> OrderedDict:
    """
    All metrics need to be include in this file
    """
    metric_file_full_path = get_path(metric_file)
    metric_dict = read_yaml(get_path(metric_file))[key_metric]

    for metric in metric_dict:
        # each metric requires unit, aggregation
        assert key_unit in metric_dict[metric], logger.error(f'Missing <unit> in metric <{metric}>.')
        
        if key_aggregation not in metric_dict[metric]:
            metric_dict[metric][key_aggregation] = 'summation'

        assert key_aggregation in metric_dict[metric], logger.error(f'Missing <aggregation> in metric <{metric}>.')
        this_aggregation = metric_dict[metric][key_aggregation].lower()
        assert this_aggregation in legal_aggregation, logger.error(f'Invalid aggregation <{this_aggregation}> in metric <{metric}>; legal values: {legal_aggregation}.')
        metric_dict[metric][key_aggregation] = this_aggregation
    
    logger.success(f'Create metric dictionary from <{metric_file_full_path}>.')

    return metric_dict


def save_metric_dict(metric_dict: OrderedDict, save_path: str) -> None:
    save_path = get_path(save_path, check_exist=False)
    metric_dict_ckpt= OrderedDict({key_metric: metric_dict})
    write_yaml(save_path, metric_dict_ckpt)
    logger.success(f'Save metric dictionary to <{save_path}>.')


def load_metric_dict(ckpt_path: str) -> OrderedDict:
    """
    Load metric from a checkpoint
    """
    full_path = get_path(ckpt_path)
    metric_dict = read_yaml(full_path)[key_metric]
    logger.success(f'Load metric dictionary from <{full_path}>.')
    return metric_dict


def create_event_metrics(event_graph: gt.Graph, architecture_dict: OrderedDict, metric_dict: OrderedDict, run_dir: str=None) -> gt.Graph:
    """
    Update the event graph, add metrics to each node.
    """

    # add event metric to vertex properties
    metric_property = event_graph.new_vertex_property('object')
    event_graph.vp.metric = metric_property

    # add tag to vertex properties
    tag_property = event_graph.new_vertex_property('object')
    event_graph.vp.tag = tag_property

    # initialize all metrics to 0.0
    for v in event_graph.vertices():
        event_graph.vp.metric[v] = OrderedDict({})
        for metric in metric_dict:
            event_graph.vp.metric[v][metric] = OrderedDict({key_value: 0.0, key_unit: metric_dict[metric][key_unit]})
        event_name = event_graph.vp.event[v]
        logger.info(f'Create metrics for event <{event_name}>.')

    logger.success(f'Create metrics for all events.')

    event_graph = create_module_metrics(event_graph, architecture_dict, run_dir)

    return event_graph


def create_module_metrics(event_graph: gt.Graph, architecture_dict: OrderedDict, run_dir: str=None) -> gt.Graph:
    """
    This function queries the interface for each architecture modules in the event graph
    """

    create_dir(run_dir)
    full_path = get_path(run_dir)

    # all modules have out degree of 0
    for v in gt.find_vertex(event_graph, 'out', 0):
        module_name = event_graph.vp.event[v]
        assert module_name in architecture_dict, logger.error(f'Invalid module <{module_name}>.')
        module_class = architecture_dict[module_name]['query']['class']
        query = architecture_dict[module_name]['query']
        result = query_interface(module_name, query, output_dir=full_path)

        # if query generates new results, update metric with new results
        event_graph.vp.metric[v] = OrderedDict({k: result.get(k, event_graph.vp.metric[v][k]) for k in event_graph.vp.metric[v]})
        
        # get number of instances for an architecture module
        event_graph.vp.metric[v][key_instance] = get_prod(architecture_dict[module_name][key_instance])

        event_graph.vp.tag[v] = architecture_dict[module_name]['tag']

        logger.info(f'Create metrics for module <{module_name}> with class <{module_class}>.')
    
    logger.success(f'Create metrics for all modules.')
    
    return event_graph


def query_module_metric(event_graph: gt.Graph, metric_dict: str, metric: str, module: str=None, operation: str=None) -> OrderedDict:
    """
    Query the metric of a module or an operation in the module.
    """
    assert metric in metric_dict, logger.error(f'Invalid metric <{metric}>.')
    assert key_aggregation in metric_dict[metric], logger.error(f'Missing aggregation in metric <{metric}>.')
    assert metric_dict[metric][key_aggregation] in legal_aggregation, logger.error(f'Invalid aggregation <{metric_dict[metric][key_aggregation]}> for metric <{metric}>; legal values: {legal_aggregation}.')
    
    # get the module node
    module_node = gt.find_vertex(event_graph, event_graph.vp.event, module)
    assert len(module_node) == 1, logger.error(f'Invalid module <{module}>.')
    module_node = module_node[0]
    assert module_node.out_degree() == 0, logger.error(f'Invalid event <{module}>; this function requires a module.')

    module_metric = event_graph.vp.metric[module_node][metric]

    # if a module has multiple operations, check if the operation is legal
    # this only works for certain modules, e.g., sram has multiple operations (read, write)
    legal_ops = []

    if (key_value not in module_metric) & (key_unit not in module_metric):
        for key in module_metric:
            legal_ops.append(key)

    
    if len(legal_ops) == 0:
        # single-operation module
        if operation is not None:
            logger.warning(f'Ignore operation <{operation}> for metric <{metric}> in module <{module}>; this module requires no specified operation.')
        query_result = module_metric
    else:
        # multi-operation module
        # example event metric:
        # OrderedDict([('read', OrderedDict([('value', 0.474466), ('unit', 'nJ')])), ('write', OrderedDict([('value', 0.499877), ('unit', 'nJ')]))])
        assert operation in legal_ops, logger.error(f'Invalid operation <{operation}> for metric <{metric}> in module <{module}>; legal values: {legal_ops}.')
        query_result = module_metric[operation]
    
    if operation is None:
        logger.success(f'Query metric <{metric}> for module <{module}>.')
    else:
        logger.success(f'Query metric <{metric}> for operation <{operation}> in module <{module}>.')
    
    return query_result


def aggregate_event_count(event_graph: gt.Graph, workload: str=None, event: str=None) -> OrderedDict:
    # find the number of event underworkload

    # get the event node
    event_node = gt.find_vertex(event_graph, event_graph.vp.event, event)
    assert len(event_node) > 0, logger.error(f'Invalid event <{event}>.')
    event_node = event_node[0]

    # validate event in workload
    if workload is not None and workload != event:
        # get the workload node
        workload_node = gt.find_vertex(event_graph, event_graph.vp.event, workload)
        assert len(workload_node) > 0, logger.error(f'Invalid workload <{workload}>.')
        workload_node = workload_node[0]
        paths = []
        for path in gt.all_paths(event_graph, workload_node, event_node):
            paths.append(path)
        assert len(paths) > 0, logger.error(f'Invalid event <{event}> in workload <{workload}>.')

        # get the event count from workload to event
        total_event_count = 0.
        for path in gt.all_paths(event_graph, workload_node, event_node):
            path_event_count = 1.
            # calculate total event count based on the path
            for idx in range(len(path)-1):
                edge = event_graph.edge(path[idx], path[idx+1])
                path_event_count *= event_graph.ep.count[edge]
                logger.debug(f'  Path (<{event_graph.vp.event[edge.source()]}> -> <{event_graph.vp.event[edge.target()]}>) updates event count to <{path_event_count}>.')
            total_event_count += path_event_count
            logger.debug(f'  Total event count (<{workload}> -> <{event}>) is increased by <{path_event_count}>.')
        # get the metric value from event to all its subevents
        logger.debug(f'  Total value (<{workload}> -> <{event}>) = <{total_event_count}>.')

    else:
        total_event_count = 1.
        logger.debug(f'  Total value (<{event}>) = <{total_event_count}>.')

    if workload is not None:
        logger.success(f'Aggregate event count for event <{event}> in workload <{workload}>.')
    else:
        logger.success(f'Aggregate event count for event <{event}>.')

    return total_event_count


def aggregate_event_metric(event_graph: gt.Graph, metric_dict: str, metric: str, workload: str=None, event: str=None) -> OrderedDict:
    """
    If workload is none, the metric will be aggregated to the event.
    If workload is not none, the metric will be aggregated to the workload.
    """
    assert metric in metric_dict, logger.error(f'Invalid metric <{metric}>.')
    assert key_aggregation in metric_dict[metric], logger.error(f'Missing aggregation in metric <{metric}>.')
    assert metric_dict[metric][key_aggregation] in legal_aggregation, logger.error(f'Invalid aggregation <{metric_dict[metric][key_aggregation]}> for metric <{metric}>; legal values: {legal_aggregation}.')
    
    metric_cfg = metric_dict[metric]
    
    # get the event node
    event_node = gt.find_vertex(event_graph, event_graph.vp.event, event)
    assert len(event_node) > 0, logger.error(f'Invalid event <{event}>.')
    event_node = event_node[0]

    # validate event in workload
    if workload is not None and workload != event:
        # get the workload node
        workload_node = gt.find_vertex(event_graph, event_graph.vp.event, workload)
        assert len(workload_node) > 0, logger.error(f'Invalid workload <{workload}>.')
        workload_node = workload_node[0]
        paths = []
        for path in gt.all_paths(event_graph, workload_node, event_node):
            paths.append(path)
        assert len(paths) > 0, logger.error(f'Invalid event <{event}> in workload <{workload}>.')

    # reset the metric values of all events to 0.0
    for v in event_graph.vertices():
        # module metric is not reset
        if v.out_degree() != 0:
            if event_graph.vp.metric[v][metric][key_value] != 0:
                logger.debug(f'  Reset metric <{metric}> in event <{event_graph.vp.event[v]}> to 0.')
            event_graph.vp.metric[v][metric][key_value] = 0.

    # output metric
    event_metric = OrderedDict({key_value: 0.0, key_unit: metric_cfg[key_unit]})

    aggregation = metric_cfg[key_aggregation]

    # module aggregation mode: only sum all leaf nodes from current node
    if aggregation == 'module':
        # the input can be either a module or an event, i.e., no limitations on the output degree of event_node
        # factor has no impact on aggregation in the module mode
        if workload is not None:
            logger.warning(f'Ignore workload <{workload}> in aggregation <{aggregation}>.')

        module_sum = 0.
        evaluated = []

        module_sum += aggregate_module(graph=event_graph, current_node=event_node, sum=0, metric=metric, evaluated=evaluated, top_event=event)
        
        event_metric[key_value] = module_sum
        if workload is None:
            logger.debug(f'  Total value (<{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}>.')
        else:
            logger.debug(f'  Total value (<{workload}> -> <{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}>.')

    # summation aggregation mode: sum all child nodes from current node
    elif aggregation == 'summation':
        # check if the event is a single-operation module
        is_single_operation, legal_ops = check_single_operation(event_graph=event_graph, metric=metric, event_node=event_node)

        if workload is None or workload == event:
            # if event is a module, summation aggregation requires single-operation module, since no operation is specified in multi-operation module
            if event_node.out_degree() == 0:
                assert is_single_operation, logger.error(f'Invalid module <{event}> for aggregation <{aggregation}>; this aggregation does not support multi-operation module.')
            
            event_graph = aggregate_summation(event_graph=event_graph, start_node=event_node, metric=metric)
            event_metric[key_value] = event_graph.vp.metric[event_node][metric][key_value]
            if workload is None:
                logger.debug(f'  Total value (<{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}>.')
            else:
                logger.debug(f'  Total value (<{workload}> -> <{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}>.')

        else:
            is_multi_operation_module = (not is_single_operation) & (event_node.out_degree() == 0)

            if is_multi_operation_module:
                legal_op_count_dict = OrderedDict()
                legal_op_metric_dict = OrderedDict()
                for legal_op in legal_ops:
                    legal_op_count_dict[legal_op] = 0.
                    legal_op_metric_dict[legal_op] = 0.

                for path in gt.all_paths(event_graph, workload_node, event_node):
                    path_event_count = 1.
                    path_event_factor = 1.
                    # calculate total event count based on the path
                    for idx in range(len(path)-1):
                        edge = event_graph.edge(path[idx], path[idx+1])
                        path_event_count *= event_graph.ep.count[edge]
                        if metric in event_graph.ep.factor[edge]:
                            path_event_factor *= event_graph.ep.factor[edge][metric]
                        logger.debug(f'  Path (<{event_graph.vp.event[edge.source()]}> -> <{event_graph.vp.event[edge.target()]}>) updates event count to <{path_event_count}> and event factor to <{path_event_factor}>.')
                    operation = event_graph.ep.operation[edge].get(metric).lower()
                    legal_op_metric_dict[operation] = get_metric_value(event_graph=event_graph, edge=edge, metric=metric)
                    delta_event_count = path_event_count * path_event_factor
                    legal_op_count_dict[operation] += delta_event_count
                    logger.debug(f'  Total event count (<{event}> : <{operation}>) is increased by <{delta_event_count}> = count <{path_event_count}> * factor <{path_event_factor}>.')

                for legal_op in legal_ops:
                    op_metric_value = legal_op_metric_dict[legal_op] * legal_op_count_dict[legal_op]
                    event_metric[key_value] += op_metric_value
                    logger.debug(f'  Total value (<{event}> : <{legal_op}>) = <{op_metric_value}> <{event_metric[key_unit]}> = single value <{legal_op_metric_dict[legal_op]}> * count <{legal_op_count_dict[legal_op]}>.')
                logger.debug(f'  Total value (<{workload}> -> <{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}>.')

            else:
                # get the event count from workload to event
                total_event_count = 0.
                for path in gt.all_paths(event_graph, workload_node, event_node):
                    path_event_count = 1.
                    path_event_factor = 1.
                    # calculate total event count based on the path
                    for idx in range(len(path)-1):
                        edge = event_graph.edge(path[idx], path[idx+1])
                        path_event_count *= event_graph.ep.count[edge]
                        if metric in event_graph.ep.factor[edge]:
                            path_event_factor *= event_graph.ep.factor[edge][metric]
                        logger.debug(f'  Path (<{event_graph.vp.event[edge.source()]}> -> <{event_graph.vp.event[edge.target()]}>) updates event count to <{path_event_count}> and event factor to <{path_event_factor}>.')
                    delta_event_count = path_event_count * path_event_factor
                    total_event_count += delta_event_count
                    logger.debug(f'  Total event count (<{workload}> -> <{event}>) is increased by <{delta_event_count}> = count <{path_event_count}> * factor <{path_event_factor}>.')
                # get the metric value from event to all its subevents
                event_graph = aggregate_summation(event_graph=event_graph, start_node=event_node, metric=metric)
                event_metric[key_value] = event_graph.vp.metric[event_node][metric][key_value] * total_event_count
                logger.debug(f'  Total value (<{workload}> -> <{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}> = single value <{event_graph.vp.metric[event_node][metric][key_value]}> * count <{total_event_count}>.')

    # specified aggregation mode: parallel/sequential is taken into account
    elif aggregation == 'specified':
        # for specified mode, the input can not be a module, i.e., the output degree shall be large than 0
        assert event_node.out_degree() > 0, logger.error(f'Invalid module <{event}> for aggregation <{aggregation}>; this aggregation requires an event.')
        
        if workload is None or workload == event:
            total_event_count = 1.
        else:
            # get the event count from workload to event
            total_event_count = 0.
            for path in gt.all_paths(event_graph, workload_node, event_node):
                path_event_count = 1.
                path_event_factor = 1.
                # calculate total event count based on the path
                for idx in range(len(path)-1):
                    edge = event_graph.edge(path[idx], path[idx+1])
                    path_event_count *= event_graph.ep.count[edge]
                    if metric in event_graph.ep.factor[edge]:
                        path_event_factor *= event_graph.ep.factor[edge][metric]
                    logger.debug(f'  Path (<{event_graph.vp.event[edge.source()]}> -> <{event_graph.vp.event[edge.target()]}>) updates event count to <{path_event_count}> and event factor to <{path_event_factor}>.')
                delta_event_count = path_event_count * path_event_factor
                total_event_count += delta_event_count
                logger.debug(f'  Total event count (<{workload}> -> <{event}>) is increased by <{delta_event_count}> = count <{path_event_count}> * factor <{path_event_factor}>.')
        
        event_graph = aggregate_specified(event_graph=event_graph, start_node=event_node, metric=metric)
        event_metric[key_value] = event_graph.vp.metric[event_node][metric][key_value] * total_event_count
        if workload is None:
            logger.debug(f'  Total value (<{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}> = single value <{event_graph.vp.metric[event_node][metric][key_value]}>.')
        else:
            logger.debug(f'  Total value (<{workload}> -> <{event}>) = <{event_metric[key_value]}> <{event_metric[key_unit]}> = single value <{event_graph.vp.metric[event_node][metric][key_value]}> * count <{total_event_count}>.')
    
    if workload is not None:
        logger.success(f'Aggregate metric <{metric}> for event <{event}> in workload <{workload}> with aggregation <{aggregation}>.')
    else:
        logger.success(f'Aggregate metric <{metric}> for event <{event}> with aggregation <{aggregation}>.')

    return event_metric


def aggregate_module(graph: gt.Graph, current_node: gt.Vertex, sum: float, metric: str, evaluated: list, top_event: str) -> float:
    # leaf nodes will return their own value
    if current_node.out_degree() == 0:
        if current_node not in evaluated:
            key_list = list(graph.vp.metric[current_node][metric].keys())
            key_list.sort()
            expected_key_list = [key_value, key_unit]
            expected_key_list.sort()
            assert key_list == expected_key_list, logger.error(f'Invalid metric <{metric}> for module <{graph.vp.event[current_node]}>.')
            evaluated.append(current_node)
            logger.info(f'Aggregate metric <{metric}> for module <{graph.vp.event[current_node]}> in event <{top_event}>.')
            total_metric_value = graph.vp.metric[current_node][metric][key_value] * graph.vp.metric[current_node][key_instance]
            logger.debug(f'  Total value (<{top_event}> -> <{graph.vp.event[current_node]}>) = <{total_metric_value}> <{graph.vp.metric[current_node][metric][key_unit]}> = single value <{graph.vp.metric[current_node][metric][key_value]}> * instance <{graph.vp.metric[current_node][key_instance]}>.')
            return sum + total_metric_value
        else:
            logger.info(f'Ignore repeated module <{graph.vp.event[current_node]}>.')
            return 0
    
    # non-leaf nodes will aggregate their children
    else:
        temp_sum = 0.
        for child in current_node.out_neighbors():
            logger.debug(f'  Check subevent <{graph.vp.event[current_node]}> -> <{graph.vp.event[child]}>.')
            temp_sum += aggregate_module(graph, child, sum, metric, evaluated, graph.vp.event[current_node])
        return temp_sum


def aggregate_summation(event_graph: gt.Graph, start_node: gt.Vertex, metric: str) -> gt.Graph:
    # topological sort for aggregation, from module to start event
    topo_order = topological_sort_reverse(event_graph, start_node=start_node)

    # aggregate metric upwards through the graph
    for v in topo_order:
        if event_graph.vertex(v).out_degree() == 0:
            logger.info(f'Aggregate metric <{metric}> for module <{event_graph.vp.event[v]}>.')
        else:
            logger.info(f'Aggregate metric <{metric}> for event <{event_graph.vp.event[v]}>.')

        for e in event_graph.vertex(v).in_edges():
            # only calculate if the parent event is on the path (in topo_order)
            if e.source() in topo_order:
                # the count of subevents in event
                subevent_count = event_graph.ep.count[e]
                if metric in event_graph.ep.factor[e]:
                    event_factor = event_graph.ep.factor[e][metric]
                else:
                    event_factor = 1.
                # get the metric value of target node
                edge_target_metric = get_metric_value(event_graph=event_graph, edge=e, metric=metric)
                # update the metric value of source node
                total_metric_value = subevent_count * edge_target_metric * event_factor
                event_graph.vp.metric[e.source()][metric][key_value] += total_metric_value
                logger.debug(f'  Total value (<{event_graph.vp.event[e.source()]}> -> <{event_graph.vp.event[e.target()]}>) = <{total_metric_value}> <{event_graph.vp.metric[e.source()][metric][key_unit]}> = single value <{edge_target_metric}> * count <{subevent_count}> * factor <{event_factor}>.')

    return event_graph


def aggregate_specified(event_graph: gt.Graph, start_node: gt.Vertex, metric: str) -> gt.Graph:
    # topological sort for aggregation, from module to start event
    # start from the event connected to modules and ignore modules
    topo_order = topological_sort_reverse(event_graph, start_node=start_node)

    # aggregate metric upwards through the graph
    for v in topo_order:
        event_name = event_graph.vp.event[v]
        if event_graph.vertex(v).out_degree() == 0:
            logger.info(f'Ignore module <{event_name}>.')
        else:
            logger.info(f'Aggregate metric <{metric}> for event <{event_name}>.')

            parallel_max = 0.
            sequential_acc = 0.

            connect_leaf_only = True
            connect_leaf_any = False

            for e in event_graph.vertex(v).out_edges():
                edge_target_event = event_graph.vp.event[e.target()]
                if e.target().out_degree() == 0:
                    assert key_value in event_graph.vp.metric[e.target()][metric], logger.error(f'Invalid metric <{metric}> for event <{edge_target_event}>; legal metric: {single_op_metric_format}.')
                    connect_leaf_any = True
                    logger.debug(f'  Ignore module <{edge_target_event}>.')
                
                else:
                    connect_leaf_only = False
                    metric_mode = event_graph.ep.aggregation[e]
                    if metric in event_graph.ep.factor[e]:
                        event_factor = event_graph.ep.factor[e][metric]
                    else:
                        event_factor = 1.
                    metric_value = event_graph.vp.metric[e.target()][metric][key_value] * event_graph.ep.count[e] * event_factor
                    logger.debug(f'  Total value (<{edge_target_event}>) = <{metric_value}> <{event_graph.vp.metric[e.target()][metric][key_unit]}> = single value <{event_graph.vp.metric[e.target()][metric][key_value]}> * count <{event_graph.ep.count[e]}> * factor <{event_factor}>.')

                    if metric_mode == 'parallel':
                        if metric_value > parallel_max:
                            logger.debug(f'  Update parallel maximum metric value to <{metric_value}>, based on event <{edge_target_event}>.')
                            parallel_max = metric_value
                    else:
                        logger.debug(f'  Update sequential accumulated metric value by <{metric_value}>, based on event <{edge_target_event}>.')
                        sequential_acc += metric_value
            
            # the final value is the sum of sequential acc and maximum parallel
            event_graph.vp.metric[v][metric][key_value] = sequential_acc + parallel_max

            # report design errors in the performance model
            # case1: if an event is only connected to modules, it should have a performance model with metric defined
            if connect_leaf_only is True:
                assert event_graph.vp[metric][v] is not None, logger.error(f'  Missing metric <{metric}> in event <{event_graph.vp.event[v]}>, since it is only connected to modules; check the performance model.')
                event_graph.vp.metric[v][metric][key_value] = event_graph.vp[metric][v][key_value]
            
            # case2: if an event is connected to no modules, it should not have a performance model with metric defined
            if connect_leaf_any is False:
                assert event_graph.vp[metric][v] is None, logger.error(f'  Invalid metric <{metric}> in event <{event_graph.vp.event[v]}>, since it is connected to no modules; check the performance model.')
            
            logger.debug(f'  Total value (<{event_graph.vp.event[v]}>) = <{event_graph.vp.metric[v][metric][key_value]}> <{event_graph.vp.metric[v][metric][key_unit]}>.')

    return event_graph


def topological_sort_reverse(graph: gt.Graph, start_node: gt.Vertex) -> list:
    # topological sort with dfs
    # output is not deterministic due to use of dfs
    topo_order = list(gt.topological_sort(graph))

    # remove redundant nodes from the start event
    to_remove = []
    for v in topo_order:
        if v != start_node:
            node_list = gt.shortest_path(graph, start_node, v)[0]
            if len(node_list) == 0:
                to_remove.append(v)
    for v in to_remove:
        topo_order.remove(v)
    
    topo_order.reverse()

    return topo_order


def get_metric_value(event_graph: gt.Graph, edge: gt.Edge, metric: str) -> float:
    target_event_node = edge.target()
    edge_target_event = event_graph.vp.event[target_event_node]
    event_metric = event_graph.vp.metric[target_event_node][metric]

    is_single_operation, legal_ops = check_single_operation(event_graph=event_graph, metric=metric, event_node=target_event_node)
    
    if is_single_operation:
        # event or single-operation module
        return event_metric[key_value]
    else:
        # multi-operation module
        # example event metric:
        # OrderedDict([('read', OrderedDict([('value', 0.474466), ('unit', 'nJ')])), ('write', OrderedDict([('value', 0.499877), ('unit', 'nJ')]))])
        operation = event_graph.ep.operation[edge].get(metric).lower()
        assert operation in legal_ops, logger.error(f'Invalid operation <{operation}> for metric <{metric}> in module <{edge_target_event}>; legal values: {legal_ops}.')
        return event_metric[operation][key_value]
    

def check_single_operation(event_graph: gt.Graph, metric: str, event_node: gt.Edge) -> bool:
    is_single_operation = False
    legal_ops = []

    if (key_value not in event_graph.vp.metric[event_node][metric]) | (key_unit not in event_graph.vp.metric[event_node][metric]):
        for key in event_graph.vp.metric[event_node][metric]:
            legal_ops.append(key)
    else:
        assert len(event_graph.vp.metric[event_node][metric]) == 2, logger.error(f'Invalid metric <{metric}> for event <{event_graph.vp.event[event_node]}>; legal metric: {single_op_metric_format}.')

    if len(legal_ops) == 0:
        is_single_operation = True
    return is_single_operation, legal_ops


def aggregate_tag_metric(event_graph: gt.Graph, metric_dict: str, metric: str, workload: str=None, tag: str=None) -> OrderedDict:
    """
    Aggregate the metric according to a tag.
    if workload is none, the metric will be aggregated to the tag.
    if workload is not none, the metric will be aggregated to the workload.
    """
    
    assert tag is not None, logger.error(f'Missing tag in aggregation of metric <{metric}>.')
    assert metric in metric_dict, logger.error(f'Invalid metric <{metric}>.')
    assert key_aggregation in metric_dict[metric], logger.error(f'Missing aggregation in metric <{metric}>.')
    assert metric_dict[metric][key_aggregation] != 'specified', logger.error(f'Invalid aggregation <{metric_dict[metric][key_aggregation]}> for tag <{tag}>; legal values: {legal_aggregation_tag}.')

    # find all modules with the tag
    tag_nodes = []
    for v in gt.find_vertex(event_graph, 'out', 0):
        if tag in event_graph.vp.tag[v]:
            tag_nodes.append(v)
    assert len(tag_nodes) > 0, logger.error(f'Invalid tag <{tag}>.')

    tag_metric = OrderedDict({key_value: 0.0, key_unit: metric_dict[metric][key_unit]})

    # for each module with the tag, aggregate the metric
    for tag_node in tag_nodes:
        module_name = event_graph.vp.event[tag_node]
        module_metric = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict, metric=metric, workload=workload, event=module_name)
        assert tag_metric[key_unit] == module_metric[key_unit], logger.error(f'Inconsistent unit in metric <{metric}> for module <{module_name}> with tag <{tag}>.')
        tag_metric[key_value] += module_metric[key_value]
        logger.debug(f'  Total value (module <{module_name}>) = <{module_metric[key_value]}> <{module_metric[key_unit]}>.')
    
    logger.debug(f'  Total value (tag <{tag}>) = <{tag_metric[key_value]}> <{tag_metric[key_unit]}>.')

    if workload is None:
        logger.success(f'Aggregate metric <{metric}> for tag <{tag}>.')
    else:
        logger.success(f'Aggregate metric <{metric}> for tag <{tag}> in workload <{workload}>.')

    return tag_metric

