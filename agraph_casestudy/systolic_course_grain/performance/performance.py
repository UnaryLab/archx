from collections import OrderedDict
from archx.utils import get_prod

def gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    isram_bank = architecture_dict['isram']['query']['bank']
    isram_width = architecture_dict['isram']['query']['width']

    wsram_bank = architecture_dict['wsram']['query']['bank']
    wsram_width = architecture_dict['wsram']['query']['width']

    osram_bank = architecture_dict['osram']['query']['bank']
    osram_width = architecture_dict['osram']['query']['width']

    m = workload_dict['gemm']['configuration']['m']
    k = workload_dict['gemm']['configuration']['k']
    n = workload_dict['gemm']['configuration']['n']
    array_dim = workload_dict['gemm']['configuration']['array_dim']
    bitwidth = workload_dict['gemm']['configuration']['bitwidth']

    isram_reads = (m * k * (n / array_dim) *  bitwidth) / min((isram_bank * isram_width), (array_dim * bitwidth))
    wsram_reads = (k * n * bitwidth) / min((wsram_bank * wsram_width), (array_dim * bitwidth))
    osram_writes = (m * k * (n / array_dim) * bitwidth) / min((osram_bank * osram_width), (array_dim * bitwidth))

    input_mappings = (m / array_dim) * (k / array_dim) * (n)
    weight_mappings = (m / array_dim) * (k / array_dim) * (n)
    # weight_mappings = (k / array_dim) * (n)

    isram_reads_dict = OrderedDict({'count': isram_reads})
    wsram_reads_dict = OrderedDict({'count': wsram_reads})
    osram_writes_dict = OrderedDict({'count': osram_writes})
    input_array_dict = OrderedDict({'count': input_mappings})
    weight_array_dict = OrderedDict({'count': weight_mappings})

    performance_dict['subevent'] = OrderedDict({'input_reads': isram_reads_dict,
                                                'weight_reads': wsram_reads_dict,
                                                'output_writes': osram_writes_dict,
                                                'input_array': input_array_dict})

    return performance_dict

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

def input_array(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    ififo_dim = get_prod(architecture_dict['ififo']['instance'])
    wfifo_dim = get_prod(architecture_dict['wfifo']['instance'])
    ofifo_dim = get_prod(architecture_dict['ofifo']['instance'])
    output_adder_dim = get_prod(architecture_dict['output_adder']['instance'])
    pe = get_prod(architecture_dict['pe']['instance'])

    frequency = architecture_dict['pe']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    ififo_dict = OrderedDict({'count': ififo_dim})
    wfifo_dict = OrderedDict({'count': wfifo_dim})
    ofifo_dict = OrderedDict({'count': ofifo_dim})
    output_adder_dict = OrderedDict({'count': output_adder_dim})
    pe_dict = OrderedDict({'count': pe})

    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict,
                                                'wfifo': wfifo_dict,
                                                'ofifo': ofifo_dict,
                                                'output_adder': output_adder_dict,
                                                'pe': pe_dict})
    return performance_dict