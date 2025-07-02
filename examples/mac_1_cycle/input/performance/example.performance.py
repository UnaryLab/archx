from collections import OrderedDict

from archx.utils import get_prod


def gemm16(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    workload_dict = workload_dict['gemm16']
    performance_dict = gemm(architecture_dict, workload_dict)
    return performance_dict


def gemm32(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    workload_dict = workload_dict['gemm32']
    performance_dict = gemm(architecture_dict, workload_dict)
    return performance_dict


def gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    # this is a very simple performance model for matrix multiplication
    performance_dict = OrderedDict()

    mac_array_size = get_prod(architecture_dict['multiplier']['instance'])
    mac_total = workload_dict['configuration']['m'] * workload_dict['configuration']['k'] * workload_dict['configuration']['n']
    # number of calls to mac_array
    mac_array_total = mac_total / mac_array_size

    # number of memory accesses
    # no data reuse is considered
    word_size = architecture_dict['multiplier']['query']['bitwidth']
    sram_rd_num = word_size * 2 * mac_total / architecture_dict['sram']['query']['width']
    sram_wr_num = sram_rd_num / 2

    mac_array_dict = OrderedDict({'count': mac_array_total})
    sram_rd_dict = OrderedDict({'count': sram_rd_num})
    sram_wr_dict = OrderedDict({'count': sram_wr_num})
    
    performance_dict['subevent'] = OrderedDict({'mac_array': mac_array_dict, 
                                                'sram_rd': sram_rd_dict, 
                                                'sram_wr': sram_wr_dict})

    return performance_dict


def mac_array(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    # this is an array of mac units
    performance_dict = OrderedDict()

    # 1 cycle per mac
    # as mac units are parallel, 1 cycle per mac_array
    performance_dict['cycle_count'] = OrderedDict({'value': 1., 'unit': 'cycles'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / architecture_dict['multiplier']['query']['frequency'] , 'unit': 'ms'})

    # by default, architecture modules have
    # 'count' is 1
    # 'aggregation' is 'parallel'

    mac_array_size = get_prod(architecture_dict['multiplier']['instance'])
    multiplier_dict = OrderedDict({'count': mac_array_size})
    adder_dict = OrderedDict({'count': mac_array_size})
    
    performance_dict['subevent'] = OrderedDict({'multiplier' : multiplier_dict, 
                                                'adder' : adder_dict})

    return performance_dict


def sram_rd(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    performance_dict = OrderedDict()
    performance_dict['cycle_count'] = OrderedDict({'value': 1., 'unit': 'cycles'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / architecture_dict['sram']['query']['frequency'] , 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy' : 'read'})})
    performance_dict['subevent'] = OrderedDict({'sram' : sram_dict})
    return performance_dict


def sram_wr(architecture_dict: OrderedDict, workload_dict: OrderedDict=None):
    performance_dict = OrderedDict()
    performance_dict['cycle_count'] = OrderedDict({'value': 1., 'unit': 'cycles'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / architecture_dict['sram']['query']['frequency'] , 'unit': 'ms'})
    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy' : 'write'})})
    performance_dict['subevent'] = OrderedDict({'sram' : sram_dict})
    return performance_dict

