from collections import OrderedDict
from archx.utils import get_prod

def fir(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:

    performance_dict = OrderedDict()

    mult_dim = get_prod(architecture_dict['mult']['instance'])
    acc_dim = get_prod(architecture_dict['acc']['instance'])
    shift_reg_dim = get_prod(architecture_dict['shift_reg']['instance'])
    pnm_dim = get_prod(architecture_dict['pnm']['instance'])
    b2rc_dim = get_prod(architecture_dict['b2rc']['instance'])
    pnm_dim = get_prod(architecture_dict['pnm']['instance'])
    control_dict = get_prod(architecture_dict['control']['instance'])

    frequency = architecture_dict['mult']['query']['frequency']
    bitwidth = architecture_dict['pnm']['query']['width']
    cycles = (2**bitwidth)
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycles * (20 * bitwidth) * 1e-9, 'unit': 'ms'})

    mult_dict = OrderedDict({'count': mult_dim})
    acc_dict = OrderedDict({'count': acc_dim})
    shift_reg_dict = OrderedDict({'count': shift_reg_dim})
    pnm_dict = OrderedDict({'count': pnm_dim})
    b2rc_dict = OrderedDict({'count': b2rc_dim})
    control_dict = OrderedDict({'count': control_dict})

    performance_dict['subevent'] = OrderedDict({'mult': mult_dict,
                                                'acc': acc_dict,
                                                'shift_reg': shift_reg_dict,
                                                'pnm': pnm_dict,
                                                'b2rc': b2rc_dict,
                                                'control': control_dict})
    return performance_dict