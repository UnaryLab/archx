from collections import OrderedDict
from archx.utils import get_prod

def fir(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    mappings = workload_dict['configuration']['mappings']

    performance_dict['subevent'] = OrderedDict({'fir_mult': OrderedDict({'count': mappings}),
                                                'fir_adder': OrderedDict({'count': mappings}),
                                                'fir_control': OrderedDict({'count': mappings})})
    return performance_dict

def fir_mult(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    mult_dim = get_prod(architecture_dict['mult']['instance'])

    bitwidth = architecture_dict['pnm']['query']['width']
    cycles = (2**bitwidth)
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycles * (9 * bitwidth) * 1e-9, 'unit': 'ms'})

    mult_dict = OrderedDict({'count': mult_dim})
    
    performance_dict['subevent'] = OrderedDict({'mult': mult_dict})
    return performance_dict

def fir_adder(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    acc_dim = get_prod(architecture_dict['acc']['instance'])

    bitwidth = architecture_dict['pnm']['query']['width']
    cycles = (2**bitwidth)
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycles * (12 * bitwidth) * 1e-9, 'unit': 'ms'})

    acc_dict = OrderedDict({'count': acc_dim})

    performance_dict['subevent'] = OrderedDict({'acc': acc_dict})
    return performance_dict

def fir_control(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    mappings = workload_dict['configuration']['mappings']

    shift_reg_dim = get_prod(architecture_dict['shift_reg']['instance'])
    pnm_dim = get_prod(architecture_dict['pnm']['instance'])
    b2rc_dim = 1
    control_dict = get_prod(architecture_dict['control']['instance'])

    bitwidth = architecture_dict['pnm']['query']['width']
    cycles = (2**bitwidth) * mappings
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycles * (20 * bitwidth) * 1e-9, 'unit': 'ms'})

    shift_reg_dict = OrderedDict({'count': shift_reg_dim})
    pnm_dict = OrderedDict({'count': pnm_dim})
    b2rc_dict = OrderedDict({'count': b2rc_dim})
    control_dict = OrderedDict({'count': control_dict})

    performance_dict['subevent'] = OrderedDict({'shift_reg': shift_reg_dict,
                                                'pnm': pnm_dict,
                                                'b2rc': b2rc_dict,
                                                'control': control_dict})
    return performance_dict