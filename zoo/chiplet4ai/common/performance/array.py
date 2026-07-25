from collections import OrderedDict
from archx.utils import get_prod

def array_input(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    ififo_dim = get_prod(architecture_dict['ififo']['instance'])
    frequency = architecture_dict['ififo']['query']['frequency']

    # hardware events
    ififo_dict = OrderedDict({'count': ififo_dim})

    # performance metrics
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict})

    return performance_dict

def array_weight(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()
    
    # architecture parameters
    wfifo_dim = get_prod(architecture_dict['wfifo']['instance'])
    weight_reg_dim = get_prod(architecture_dict['weight_reg']['instance'])
    weight_path_reg_dim = get_prod(architecture_dict['weight_path_reg']['instance'])
    weight_en_reg_dim = get_prod(architecture_dict['weight_en_reg']['instance'])
    weight_path_en_reg_dim = get_prod(architecture_dict['weight_path_en_reg']['instance'])
    frequency = architecture_dict['wfifo']['query']['frequency']

    # hardware events
    wfifo_dict = OrderedDict({'count': wfifo_dim})
    weight_reg_dict = OrderedDict({'count': weight_reg_dim})
    weight_path_reg_dict = OrderedDict({'count': weight_path_reg_dim})
    weight_en_reg_dict = OrderedDict({'count': weight_en_reg_dim})
    weight_path_en_reg_dict = OrderedDict({'count': weight_path_en_reg_dim})

    # performance metrics
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['subevent'] = OrderedDict({
        'wfifo': wfifo_dict,
        'weight_reg': weight_reg_dict,
        'weight_path_reg': weight_path_reg_dict,
        'weight_en_reg': weight_en_reg_dict,
        'weight_path_en_reg': weight_path_en_reg_dict
    })

    return performance_dict

def array_compute(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    pe_dim = get_prod(architecture_dict['pe']['instance'])
    ofifo_dim = architecture_dict['ofifo']['instance'][0]
    output_adder_dim = architecture_dict['output_adder']['instance'][0]
    frequency = architecture_dict['pe']['query']['frequency']

    # hardware events
    pe_dict = OrderedDict({'count': pe_dim})
    ofifo_dict = OrderedDict({'count': ofifo_dim})
    output_adder_dict = OrderedDict({'count': output_adder_dim})

    # performance metrics
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    performance_dict['subevent'] = OrderedDict({
        'pe': pe_dict,
        'ofifo': ofifo_dict,
        'output_adder': output_adder_dict
    })

    return performance_dict