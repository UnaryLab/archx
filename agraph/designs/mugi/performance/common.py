from collections import OrderedDict
from archx.utils import get_prod

def instruction(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['imux']['query']['frequency']
    imux_dim = architecture_dict['imux']['instance'][-1]
    wmux_dim = architecture_dict['wmux']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    imux_dict = OrderedDict({'count': imux_dim})
    wmux_dict = OrderedDict({'count': wmux_dim})

    performance_dict['subevent'] = OrderedDict({'imux': imux_dict})
    
    performance_dict['subevent'].update({'wmux': wmux_dict})
    return performance_dict

def counter_reuse(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))['configuration']
    cycles = workload_dict['cycles']
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['counter']['query']['frequency']
    counter_dim = architecture_dict['counter']['instance'][-1]
    counter_register_dim = get_prod(architecture_dict['counter_register']['instance'][-2:])

    cycle_count = cycles / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    counter_dict = OrderedDict({'count': counter_dim})
    counter_register_dict = OrderedDict({'count': counter_register_dim})

    performance_dict['subevent'] = OrderedDict({'counter': counter_dict,
                                                'counter_register': counter_register_dict})
    return performance_dict

def vector(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))['configuration']
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    multiplier_vector_dim = OrderedDict({'count': multiplier_vector_dim})
    register_vector_dim = OrderedDict({'count': register_vector_dim})

    performance_dict['subevent'] = OrderedDict({'multiplier_vector': multiplier_vector_dim,
                                                'register_vector': register_vector_dim})
    return performance_dict