from collections import OrderedDict
from archx.utils import get_prod

def butterfly(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    pe_dim = get_prod(architecture_dict['pe']['instance'])
    mappings = workload_dict['configuration']['mappings']
    frequency = architecture_dict['pe']['query']['frequency']
    
    performance_dict['cycle_count'] = OrderedDict({'value': mappings, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': mappings / 1000 / frequency, 'unit': 'ms'})

    pe_dict = OrderedDict({'count': pe_dim * mappings})

    performance_dict['subevent'] = OrderedDict({'pe': pe_dict})
    return performance_dict