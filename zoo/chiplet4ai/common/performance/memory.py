from collections import OrderedDict

def dram_input_read(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    bandwidth = architecture_dict['dram']['query']['bandwidth']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    # unit convention: 'bandwidth' is DECIMAL GB/s (same as mapping._dram_bytes_per_cycle),
    # so ms per byte is 1000 / (GB/s * 1e9).
    performance_dict['runtime'] = OrderedDict({'value': 1000 / (bandwidth * 1e9), 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    dram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'dram': dram_dict})

    return performance_dict

def dram_weight_read(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    bandwidth = architecture_dict['dram']['query']['bandwidth']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    # unit convention: 'bandwidth' is DECIMAL GB/s (same as mapping._dram_bytes_per_cycle),
    # so ms per byte is 1000 / (GB/s * 1e9).
    performance_dict['runtime'] = OrderedDict({'value': 1000 / (bandwidth * 1e9), 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    dram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'dram': dram_dict})

    return performance_dict

def dram_output_write(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    bandwidth = architecture_dict['dram']['query']['bandwidth']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    # unit convention: 'bandwidth' is DECIMAL GB/s (same as mapping._dram_bytes_per_cycle),
    # so ms per byte is 1000 / (GB/s * 1e9).
    performance_dict['runtime'] = OrderedDict({'value': 1000 / (bandwidth * 1e9), 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    dram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'dram': dram_dict})

    return performance_dict

def dram_output_read(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    bandwidth = architecture_dict['dram']['query']['bandwidth']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    # unit convention: 'bandwidth' is DECIMAL GB/s (same as mapping._dram_bytes_per_cycle),
    # so ms per byte is 1000 / (GB/s * 1e9).
    performance_dict['runtime'] = OrderedDict({'value': 1000 / (bandwidth * 1e9), 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    dram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'dram': dram_dict})

    return performance_dict

def sram_input_read(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['isram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'isram': sram_dict})

    return performance_dict

def sram_input_write(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['isram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'isram': sram_dict})

    return performance_dict

def sram_weight_read(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['wsram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'wsram': sram_dict})

    return performance_dict

def sram_weight_write(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['wsram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'wsram': sram_dict})

    return performance_dict

def sram_output_read(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['osram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict

def sram_output_write(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    frequency = architecture_dict['osram']['query']['frequency']

    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['bandwidth'] = OrderedDict({'value': 1, 'unit': 'GiB/s'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'osram': sram_dict})

    return performance_dict
