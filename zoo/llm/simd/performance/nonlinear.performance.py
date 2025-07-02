from collections import OrderedDict
from archx.utils import get_prod

def gemm_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    performance_dict['cycle_count'] = OrderedDict({'value': 0, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 0, 'unit': 'ms'})

    ififo_dict = OrderedDict({'count': 0})
    wfifo_dict = OrderedDict({'count': 0})
    input_register_dict = OrderedDict({'count': 0})
    weight_register_dict = OrderedDict({'count': 0})
    multiplier_dict = OrderedDict({'count': 0})
    pe_register_dict = OrderedDict({'count': 0})
    accumulator_dict = OrderedDict({'count': 0})
    accumulator_register_dict = OrderedDict({'count': 0})
    adder_dict = OrderedDict({'count': 0})
    ofifo_dict = OrderedDict({'count': 0})

    if 'int_to_fp' in architecture_dict:
        int_to_fp_dict = OrderedDict({'count': 0})
    if 'int_to_fp_figna' in architecture_dict:
        int_to_fp_figna_dict = OrderedDict({'count': 0})
    if 'ch_aloc' in architecture_dict:
        ch_aloc_dict = OrderedDict({'count': 0})
    if 'ch_dealoc' in architecture_dict:
        ch_dealoc_dict = OrderedDict({'count': 0})
    if 'prealigner' in architecture_dict:
        prealigner_dict = OrderedDict({'count': 0})


    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict,
                                                'wfifo': wfifo_dict,
                                                'input_register': input_register_dict,
                                                'weight_register': weight_register_dict,
                                                'multiplier': multiplier_dict,
                                                'pe_register': pe_register_dict,
                                                'accumulator': accumulator_dict,
                                                'accumulator_register': accumulator_register_dict,
                                                'adder': adder_dict,
                                                'ofifo': ofifo_dict})
    if 'int_to_fp' in architecture_dict:
        performance_dict['subevent']['int_to_fp'] = int_to_fp_dict
    if 'int_to_fp_figna' in architecture_dict:
        performance_dict['subevent']['int_to_fp_figna'] = int_to_fp_figna_dict
    if 'ch_aloc' in architecture_dict:
        performance_dict['subevent']['ch_aloc'] = ch_aloc_dict
    if 'ch_dealoc' in architecture_dict:
        performance_dict['subevent']['ch_dealoc'] = ch_dealoc_dict
    if 'prealigner' in architecture_dict:
        performance_dict['subevent']['prealigner'] = prealigner_dict

    return performance_dict

def vector_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    accumulator_vector_dim = architecture_dict['accumulator_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]
    mac_register_vector = architecture_dict['mac_register_vector']['instance'][-1]

    exp_mult_cycles = workload_dict['configuration']['exp_mult_cycles']
    div_mult_cycles = workload_dict['configuration']['division_mult_cycles']
    accumulation_cycles = workload_dict['configuration']['accumulation_cycles']

    multiplication_cycles = exp_mult_cycles + div_mult_cycles

    cycle_count = (multiplication_cycles + accumulation_cycles) / router_dim 
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    multiplier_vector_dict = OrderedDict({'count': multiplier_vector_dim * multiplication_cycles})
    accumulator_vector_dict = OrderedDict({'count': accumulator_vector_dim * accumulation_cycles})
    register_vector_dict = OrderedDict({'count': register_vector_dim * multiplication_cycles})
    mac_register_vector_dict = OrderedDict({'count': mac_register_vector * accumulation_cycles})

    performance_dict['subevent'] = OrderedDict({'multiplier_vector': multiplier_vector_dict,
                                                'accumulator_vector': accumulator_vector_dict,
                                                'register_vector': register_vector_dict,
                                                'mac_register_vector': mac_register_vector_dict})
    return performance_dict