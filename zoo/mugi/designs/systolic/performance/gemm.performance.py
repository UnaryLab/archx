from collections import OrderedDict
from archx.utils import get_prod

def input_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['ififo']['query']['frequency']
    ififo_dim = architecture_dict['ififo']['instance'][-1]
    input_register_dim = get_prod(architecture_dict['input_register']['instance'][-2:])

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    ififo_dict = OrderedDict({'count': ififo_dim})
    input_register_dict = OrderedDict({'count': input_register_dim})

    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict,
                                                'input_register': input_register_dict})
    return performance_dict

def weight_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['wfifo']['query']['frequency']
    wfifo_dim = architecture_dict['wfifo']['instance'][-1]
    weight_register_dim = get_prod(architecture_dict['weight_register']['instance'][-2:])

    if 'int_to_fp' in architecture_dict:
        int_to_fp_dim = architecture_dict['int_to_fp']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    wfifo_dict = OrderedDict({'count': wfifo_dim})
    weight_register_dict = OrderedDict({'count': weight_register_dim})
    if 'int_to_fp' in architecture_dict:
        int_to_fp_dict = OrderedDict({'count': int_to_fp_dim})

    performance_dict['subevent'] = OrderedDict({'wfifo': wfifo_dict,
                                                'weight_register': weight_register_dict})
    if 'int_to_fp' in architecture_dict:
        performance_dict['subevent']['int_to_fp'] = int_to_fp_dict
    return performance_dict

def array_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier']['query']['frequency']

    array_width = architecture_dict['multiplier']['instance'][-1]
    array_height = architecture_dict['multiplier']['instance'][-2]

    multiplier_dim = get_prod(architecture_dict['multiplier']['instance'][-2:])
    pe_register_dim = get_prod(architecture_dict['pe_register']['instance'][-2:])
    accumulator_dim = get_prod(architecture_dict['accumulator']['instance'][-2:])
    icnt_dim = get_prod(architecture_dict['icnt']['instance'][-2:])
    icmp_dim = get_prod(architecture_dict['icmp']['instance'][-2:])
    iadd_dim = get_prod(architecture_dict['iadd']['instance'][-2:])
    wcnt_dim = get_prod(architecture_dict['wcnt']['instance'][-2:])
    wcmp_dim = get_prod(architecture_dict['wcmp']['instance'][-2:])
    wadd_dim = get_prod(architecture_dict['wadd']['instance'][-2:])
    adder_dim = architecture_dict['adder']['instance'][-1]
    ofifo_dim = architecture_dict['ofifo']['instance'][-1]

    if 'ch_aloc' in architecture_dict:
        ch_aloc_dim = get_prod(architecture_dict['ch_aloc']['instance'][-1])
    if 'ch_dealoc' in architecture_dict:
        ch_dealoc_dim = get_prod(architecture_dict['ch_dealoc']['instance'][-1])
    if 'int_to_fp_figna' in architecture_dict:
        int_to_fp_figna_dim = get_prod(architecture_dict['int_to_fp_figna']['instance'][-1])
    if 'prealigner' in architecture_dict:
        prealigner_dim = get_prod(architecture_dict['prealigner']['instance'][-1])

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    multiplier_dict = OrderedDict({'count': multiplier_dim})
    pe_register_dict = OrderedDict({'count': pe_register_dim})
    accumulator_dict = OrderedDict({'count': accumulator_dim})
    icnt_dict = OrderedDict({'count': icnt_dim})
    icmp_dict = OrderedDict({'count': icmp_dim})
    iadd_dict = OrderedDict({'count': iadd_dim})
    wcnt_dict = OrderedDict({'count': wcnt_dim})
    wcmp_dict = OrderedDict({'count': wcmp_dim})
    wadd_dict = OrderedDict({'count': wadd_dim})
    adder_dict = OrderedDict({'count': adder_dim * cycle_count})
    ofifo_dict = OrderedDict({'count': ofifo_dim * cycle_count})

    if 'ch_aloc' in architecture_dict:
        ch_aloc_dict = OrderedDict({'count': ch_aloc_dim})
    if 'ch_dealoc' in architecture_dict:
        ch_dealoc_dict = OrderedDict({'count': ch_dealoc_dim})
    if 'int_to_fp_figna' in architecture_dict:
        int_to_fp_figna_dict = OrderedDict({'count': int_to_fp_figna_dim})
    if 'prealigner' in architecture_dict:
        prealigner_dict = OrderedDict({'count': prealigner_dim})

    performance_dict['subevent'] = OrderedDict({'multiplier': multiplier_dict,
                                                'pe_register': pe_register_dict,
                                                'accumulator': accumulator_dict,
                                                'adder': adder_dict,
                                                'ofifo': ofifo_dict,
                                                'icnt': icnt_dict,
                                                'icmp': icmp_dict,
                                                'iadd': iadd_dict,
                                                'wcnt': wcnt_dict,
                                                'wcmp': wcmp_dict,
                                                'wadd': wadd_dict})
    if 'ch_aloc' in architecture_dict:
        performance_dict['subevent']['ch_aloc'] = ch_aloc_dict
    if 'ch_dealoc' in architecture_dict:
        performance_dict['subevent']['ch_dealoc'] = ch_dealoc_dict
    if 'int_to_fp_figna' in architecture_dict:
        performance_dict['subevent']['int_to_fp_figna'] = int_to_fp_figna_dict
    if 'prealigner' in architecture_dict:
        performance_dict['subevent']['prealigner'] = prealigner_dict

    return performance_dict

def vector_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier_vector']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]

    if 'pwl_comparator' in architecture_dict:
        pwl_comparator_dict = OrderedDict({'count': 0})
    if 'pwl_encoder' in architecture_dict:
        pwl_encoder_dict = OrderedDict({'count': 0})
    if 'pwl_register' in architecture_dict:
        pwl_register_dict = OrderedDict({'count': 0})
    if 'pipeline_register' in architecture_dict:
        pipeline_register_dict = OrderedDict({'count': 0})
    if 'adder_vector' in architecture_dict:
        adder_vector_dict = OrderedDict({'count': 0})
    if 'taylor_register' in architecture_dict:
        taylor_register_dict = OrderedDict({'count': 0})

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    multiplier_vector_dict = OrderedDict({'count': multiplier_vector_dim})
    accumulator_vector_dict = OrderedDict({'count': 0})
    register_vector_dict = OrderedDict({'count': register_vector_dim})
    mac_register_vector_dict = OrderedDict({'count': 0})

    performance_dict['subevent'] = OrderedDict({'multiplier_vector': multiplier_vector_dict,
                                                'accumulator_vector': accumulator_vector_dict,
                                                'register_vector': register_vector_dict,
                                                'mac_register_vector': mac_register_vector_dict})
    if 'pwl_comparator' in architecture_dict:
        performance_dict['subevent']['pwl_comparator'] = pwl_comparator_dict
    if 'pwl_encoder' in architecture_dict:
        performance_dict['subevent']['pwl_encoder'] = pwl_encoder_dict
    if 'pwl_register' in architecture_dict:
        performance_dict['subevent']['pwl_register'] = pwl_register_dict
    if 'pipeline_register' in architecture_dict:
        performance_dict['subevent']['pipeline_register'] = pipeline_register_dict
    if 'adder_vector' in architecture_dict:
        performance_dict['subevent']['adder_vector'] = adder_vector_dict
    if 'taylor_register' in architecture_dict:
        performance_dict['subevent']['taylor_register'] = taylor_register_dict
    return performance_dict