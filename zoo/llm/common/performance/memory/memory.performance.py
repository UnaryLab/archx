from collections import OrderedDict
from archx.utils import get_prod

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

def sram_event(event_type, sram, architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict[sram]['query']['frequency']

    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    events = 1 / router_dim

    performance_dict['cycle_count'] = OrderedDict({'value': events, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': events / 1000 / frequency, 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': event_type})})
    performance_dict['subevent'] = OrderedDict({sram: sram_dict})

    return performance_dict

def isram_offchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('read', 'isram', architecture_dict, workload_dict)

def isram_offchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('write', 'isram', architecture_dict, workload_dict)

def wsram_offchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('read', 'wsram', architecture_dict, workload_dict)

def wsram_offchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('write', 'wsram', architecture_dict, workload_dict)

def osram_offchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('read', 'osram', architecture_dict, workload_dict)

def osram_offchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('write', 'osram', architecture_dict, workload_dict)

def isram_onchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('read', 'isram', architecture_dict, workload_dict)

def isram_onchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('write', 'isram', architecture_dict, workload_dict)

def wsram_onchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('read', 'wsram', architecture_dict, workload_dict)

def wsram_onchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('write', 'wsram', architecture_dict, workload_dict)

def osram_onchip_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('read', 'osram', architecture_dict, workload_dict)

def osram_onchip_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    return sram_event('write', 'osram', architecture_dict, workload_dict)