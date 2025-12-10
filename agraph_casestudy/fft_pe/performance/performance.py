from collections import OrderedDict
from archx.utils import get_prod

def butterfly(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:

    performance_dict = OrderedDict()

    regA_dim = get_prod(architecture_dict['regA']['instance'])
    regB_dim = get_prod(architecture_dict['regB']['instance'])
    multiplier_dim = get_prod(architecture_dict['multiplier']['instance'])
    adderA_dim = get_prod(architecture_dict['adderA']['instance'])
    adderB_dim = get_prod(architecture_dict['adderB']['instance'])
    not_gate_dim = get_prod(architecture_dict['not_gate']['instance'])

    mappings = workload_dict['butterfly']['configuration']['mappings']

    cycle_count = mappings
    frequency = architecture_dict['multiplier']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    regA_dict = OrderedDict({'count': regA_dim * mappings})
    regB_dict = OrderedDict({'count': regB_dim * mappings})
    multiplier_dict = OrderedDict({'count': multiplier_dim * mappings})
    adderA_dict = OrderedDict({'count': adderA_dim * mappings})
    adderB_dict = OrderedDict({'count': adderB_dim * mappings})
    not_gate_dict = OrderedDict({'count': not_gate_dim * mappings})

    performance_dict['subevent'] = OrderedDict({'regA': regA_dict,
                                                'regB': regB_dict,
                                                'multiplier': multiplier_dict,
                                                'adderA': adderA_dict,
                                                'adderB': adderB_dict,
                                                'not_gate': not_gate_dict})

    return performance_dict