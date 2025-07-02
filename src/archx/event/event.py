import graph_tool.all as gt

from collections import OrderedDict
from loguru import logger

from archx.utils import read_yaml, get_path


def create_event_graph(event_file: str) -> gt.Graph:
    """
    create an event graph, whose node are events.
    each event has its own performance model.
    """

    # read the event configuration file and convert to a dictionary
    event_file_full_path = get_path(event_file)
    event_dict = read_yaml(event_file_full_path)['event']

    # init graph
    event_graph = gt.Graph(directed = True)

    # add vertex properties
    # event name
    event_property = event_graph.new_vertex_property('string')
    event_graph.vp.event = event_property

    # event performance model
    performance_property = event_graph.new_vertex_property('string')
    event_graph.vp.performance = performance_property

    # a map of the event name to the node index
    event_to_index = {}

    event_list = []

    # for each event, create a graph node
    for event, properties in event_dict.items():
        v = event_graph.add_vertex()
        event_graph.vp.event[v] = event
        event_graph.vp.performance[v] = properties.get('performance', None)
        event_to_index[event] = v
        event_list.append(event)
    
    # add architecture modules as leaf node
    for event, properties in event_dict.items():
        subevents = properties.get('subevent', [])
        for subevent in subevents:
            if subevent not in event_list:
                v = event_graph.add_vertex()
                event_graph.vp.event[v] = subevent
                event_graph.vp.performance[v] = None
                event_to_index[subevent] = v
                event_list.append(subevent)
    
    # extract all edges and add to graph
    # these edge attributes are the results of performance models
    # default subevent count to 1
    # default aggregation to parallel
    edges_with_properties = {
        (event_to_index[index], event_to_index[neighbor], 1.0, 'parallel', None, None)
        for index, neighbors in event_dict.items()
        for neighbor in neighbors.get('subevent', [])
    }

    event_graph.add_edge_list(edges_with_properties, eprops = [('count', 'double'), ('aggregation', 'string'), ('operation', 'object'), ('factor', 'object')])

    # default operation to empty OrderedDict
    # if not empty, it should be an OrderedDict, where the key-value pair will point out the query path in the result dictionary
    # for example, dynamic_energy: read, with take the value of read in dynamic_energy from sram output. 
    # the factor tells how to scale the queried result in a multiplicative way
    for e in event_graph.edges():
        event_graph.ep.operation[e] = OrderedDict({})
        event_graph.ep.factor[e] = OrderedDict({})

    logger.success(f'Create event graph from <{event_file_full_path}>.')

    return event_graph


def save_event_graph(event_graph: gt.Graph, save_path: str) -> None:
    save_path = get_path(save_path, check_exist=False)
    event_graph.save(save_path)
    logger.success(f'Save event graph to <{save_path}>.')


def load_event_graph(ckpt_path: str) -> gt.Graph:
    full_path = get_path(ckpt_path)
    event_graph_ckpt = gt.load_graph(full_path)
    logger.success(f'Load event graph from <{full_path}>.')
    return event_graph_ckpt

