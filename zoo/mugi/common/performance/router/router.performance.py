from collections import OrderedDict
from archx.utils import get_prod

def router_event(router, architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict[router]['query']['frequency']

    router_width = architecture_dict[router]['instance'][-1]
    router_bandwidth = architecture_dict[router]['query']['bandwidth'] * 2**23

    cycle_count = 1 / router_width / router_bandwidth

    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    router_dict = OrderedDict({'count': 1})
    performance_dict['subevent'] = OrderedDict({router: router_dict})
    return performance_dict

def irouter_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    return router_event('irouter', architecture_dict, workload_dict)

def wrouter_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    return router_event('wrouter', architecture_dict, workload_dict)

def orouter_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    return router_event('orouter', architecture_dict, workload_dict)