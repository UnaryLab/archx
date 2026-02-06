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

def input_offchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['isram']['query']['frequency']

    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
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

def weight_offchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['wsram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'wsram': sram_dict})

    return performance_dict

def output_offchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['wsram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict

def output_offchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['osram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict

def output_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['osram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict

def output_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['wsram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict

def dram_event(width, event_type, architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    
    performance_dict = OrderedDict()

    gigabyte = (2 ** 33) # 1 GB = 2^33 bits
    megabyte = (2 ** 23) # 1 MB = 2^23 bits
    frequency = architecture_dict['dram']['query']['frequency']
    if 'irouter' in architecture_dict:
        bandwidth = architecture_dict['irouter']['query']['bandwidth'] * megabyte
    else:
        bandwidth = architecture_dict['dram']['query']['bandwidth'] * gigabyte

    performance_dict['cycle_count'] = OrderedDict({'value': width / bandwidth, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': (width/bandwidth) / 1000 / frequency, 'unit': 'ms'})
    dram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': event_type})})
    performance_dict['subevent'] = OrderedDict({'dram': dram_dict})

    return performance_dict

def dram_input_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    isram_width = architecture_dict['isram']['query']['width']
    return dram_event(isram_width, 'read', architecture_dict, workload_dict)

def dram_weight_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    wsram_width = architecture_dict['wsram']['query']['width']
    return dram_event(wsram_width, 'read', architecture_dict, workload_dict=None)

def dram_output_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    osram_width = architecture_dict['osram']['query']['width']
    return dram_event(osram_width, 'read', architecture_dict, workload_dict=None)

def dram_output_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    osram_width = architecture_dict['osram']['query']['width']
    return dram_event(osram_width, 'write', architecture_dict, workload_dict=None)