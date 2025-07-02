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
    adder_dict = OrderedDict({'count': 0})
    ofifo_dict = OrderedDict({'count': 0})
    icnt_dict = OrderedDict({'count': 0})
    icmp_dict = OrderedDict({'count': 0})
    iadd_dict = OrderedDict({'count': 0})
    wcnt_dict = OrderedDict({'count': 0})
    wcmp_dict = OrderedDict({'count': 0})
    wadd_dict = OrderedDict({'count': 0})

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
                                                'adder': adder_dict,
                                                'ofifo': ofifo_dict,
                                                'icnt': icnt_dict,
                                                'icmp': icmp_dict,
                                                'iadd': iadd_dict,
                                                'wcnt': wcnt_dict,
                                                'wcmp': wcmp_dict,
                                                'wadd': wadd_dict})
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

def softmax_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier_vector']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    accumulator_vector_dim = architecture_dict['accumulator_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]
    mac_register_vector_dim = architecture_dict['mac_register_vector']['instance'][-1]

    if 'pwl_comparator' in architecture_dict:
        pwl_comparator_dim = architecture_dict['pwl_comparator']['instance'][-1]
    if 'pwl_encoder' in architecture_dict:
        pwl_encoder_dim = architecture_dict['pwl_encoder']['instance'][-1]
    if 'pwl_register' in architecture_dict:
        pwl_register_dim = architecture_dict['pwl_register']['instance'][-1]
    if 'pipeline_register' in architecture_dict:
        pipeline_register_dim = architecture_dict['pipeline_register']['instance'][-1]
    if 'adder_vector' in architecture_dict:
        adder_vector_dim = architecture_dict['adder_vector']['instance'][-1]
    if 'taylor_register' in architecture_dict:
        taylor_register_dim = architecture_dict['taylor_register']['instance'][-1]

    if 'pwl_comparator' in architecture_dict:
        division_cycles = workload_dict['configuration']['approximate_division_cycles']
        cycle_count = workload_dict['configuration']['pwl_cycles'] + division_cycles
        multiplier_count = multiplier_vector_dim
        adder_vector_count = adder_vector_dim
        accumulator_count = accumulator_vector_dim
        register_count = register_vector_dim * cycle_count
        mac_register_count = accumulator_vector_dim
    elif 'taylor_register' in architecture_dict:
        division_cycles = workload_dict['configuration']['approximate_division_cycles']
        cycle_count = taylor_register_dim + division_cycles
        multiplier_count = multiplier_vector_dim * taylor_register_dim
        adder_vector_count = adder_vector_dim * taylor_register_dim
        accumulator_count = accumulator_vector_dim
        register_count = register_vector_dim * cycle_count
        mac_register_count = accumulator_vector_dim
    else:
        exp_mult_cycles = workload_dict['configuration']['exp_mult_cycles']
        div_mult_cycles = workload_dict['configuration']['division_mult_cycles']
        accumulation_cycles = workload_dict['configuration']['accumulation_cycles']
        multiplication_cycles = exp_mult_cycles + div_mult_cycles
        cycle_count = multiplication_cycles + accumulation_cycles

        multiplier_count = multiplier_vector_dim * multiplication_cycles
        accumulator_count = accumulator_vector_dim * accumulation_cycles
        register_count = register_vector_dim * multiplication_cycles
        mac_register_count = mac_register_vector_dim * accumulation_cycles

    cycle_count = cycle_count / router_dim

    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    multiplier_vector_dict = OrderedDict({'count': multiplier_count})
    accumulator_vector_dict = OrderedDict({'count': accumulator_count})
    register_vector_dict = OrderedDict({'count': register_count})
    mac_register_vector_dict = OrderedDict({'count': mac_register_count})

    if 'pwl_comparator' in architecture_dict:
        pwl_comparator_dict = OrderedDict({'count': pwl_comparator_dim})
    if 'pwl_encoder' in architecture_dict:
        pwl_encoder_dict = OrderedDict({'count': pwl_encoder_dim})
    if 'pwl_register' in architecture_dict:
        pwl_register_dict = OrderedDict({'count': pwl_register_dim})
    if 'pipeline_register' in architecture_dict:
        pipeline_register_dict = OrderedDict({'count': pipeline_register_dim})
    if 'adder_vector' in architecture_dict:
        adder_vector_dict = OrderedDict({'count': adder_vector_count})
    if 'taylor_register' in architecture_dict:
        taylor_register_dict = OrderedDict({'count': taylor_register_dim})

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

def silu_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier_vector']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    accumulator_vector_dim = architecture_dict['accumulator_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]
    mac_register_vector_dim = architecture_dict['mac_register_vector']['instance'][-1]

    if 'pwl_comparator' in architecture_dict:
        pwl_comparator_dim = architecture_dict['pwl_comparator']['instance'][-1]
    if 'pwl_encoder' in architecture_dict:
        pwl_encoder_dim = architecture_dict['pwl_encoder']['instance'][-1]
    if 'pwl_register' in architecture_dict:
        pwl_register_dim = architecture_dict['pwl_register']['instance'][-1]
    if 'pipeline_register' in architecture_dict:
        pipeline_register_dim = architecture_dict['pipeline_register']['instance'][-1]
    if 'adder_vector' in architecture_dict:
        adder_vector_dim = architecture_dict['adder_vector']['instance'][-1]
    if 'taylor_register' in architecture_dict:
        taylor_register_dim = architecture_dict['taylor_register']['instance'][-1]

    if 'pwl_comparator' in architecture_dict:
        division_cycles = workload_dict['configuration']['approximate_division_cycles']
        cycle_count = workload_dict['configuration']['pwl_cycles'] + division_cycles
        multiplier_count = multiplier_vector_dim
        adder_vector_count = adder_vector_dim
        accumulator_count = 0
        register_count = register_vector_dim * cycle_count
        mac_register_count = 0
    elif 'taylor_register' in architecture_dict:
        division_cycles = workload_dict['configuration']['approximate_division_cycles']
        cycle_count = taylor_register_dim + division_cycles
        multiplier_count = multiplier_vector_dim * taylor_register_dim
        adder_vector_count = adder_vector_dim * taylor_register_dim
        accumulator_count = 0
        register_count = register_vector_dim * cycle_count
        mac_register_count = 0
    else:
        exp_mult_cycles = workload_dict['configuration']['exp_mult_cycles']
        div_mult_cycles = workload_dict['configuration']['division_mult_cycles']
        accumulation_cycles = workload_dict['configuration']['accumulation_cycles']
        multiplication_cycles = exp_mult_cycles + div_mult_cycles
        cycle_count = multiplication_cycles + accumulation_cycles

        multiplier_count = multiplier_vector_dim * multiplication_cycles
        accumulator_count = accumulator_vector_dim * accumulation_cycles
        register_count = register_vector_dim * multiplication_cycles
        mac_register_count = mac_register_vector_dim * accumulation_cycles

    cycle_count = cycle_count / router_dim

    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    multiplier_vector_dict = OrderedDict({'count': multiplier_count})
    accumulator_vector_dict = OrderedDict({'count': accumulator_count})
    register_vector_dict = OrderedDict({'count': register_count})
    mac_register_vector_dict = OrderedDict({'count': mac_register_count})

    if 'pwl_comparator' in architecture_dict:
        pwl_comparator_dict = OrderedDict({'count': pwl_comparator_dim})
    if 'pwl_encoder' in architecture_dict:
        pwl_encoder_dict = OrderedDict({'count': pwl_encoder_dim})
    if 'pwl_register' in architecture_dict:
        pwl_register_dict = OrderedDict({'count': pwl_register_dim})
    if 'pipeline_register' in architecture_dict:
        pipeline_register_dict = OrderedDict({'count': pipeline_register_dim})
    if 'adder_vector' in architecture_dict:
        adder_vector_dict = OrderedDict({'count': adder_vector_count})
    if 'taylor_register' in architecture_dict:
        taylor_register_dict = OrderedDict({'count': taylor_register_dim})

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
