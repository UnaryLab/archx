from collections import OrderedDict
from archx.utils import get_prod

def input_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['isram']['query']['frequency']

    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'isram': sram_dict})

    return performance_dict

def weight_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['wsram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'wsram': sram_dict})

    return performance_dict

def output_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['osram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict