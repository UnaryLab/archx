from collections import OrderedDict
from archx.utils import get_prod

# NOTE: The four subarchitectures (mac, figna, pwl, taylor) are folded into a single
# AGraph design. Every subarch-specific module is always present in architecture_dict;
# inactive ones are instance-gated to 0 (so their area and instance-derived counts vanish).
# gemm_nonlinear zeroes every module count, so it is agnostic to the active subarch.
# softmax_nonlinear / silu_nonlinear choose their cycle-count formula from the swept
# 'subarch' workload flag rather than from module presence (which is now always true).


def gemm_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    performance_dict['cycle_count'] = OrderedDict({'value': 0, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 0, 'unit': 'ms'})

    # Every module in the (superset) gemm_nonlinear event contributes zero here.
    zero_modules = ['ififo', 'wfifo', 'input_register', 'weight_register', 'multiplier',
                    'pe_register', 'accumulator', 'adder', 'ofifo', 'icnt', 'icmp', 'iadd',
                    'wcnt', 'wcmp', 'wadd', 'int_to_fp', 'int_to_fp_figna', 'ch_aloc',
                    'ch_dealoc', 'prealigner']
    performance_dict['subevent'] = OrderedDict((name, OrderedDict({'count': 0})) for name in zero_modules)
    return performance_dict


def _vector_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict, is_silu: bool)->OrderedDict:
    performance_dict = OrderedDict()
    subarch = workload_dict['configuration']['subarch']
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier_vector']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    accumulator_vector_dim = architecture_dict['accumulator_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]
    mac_register_vector_dim = architecture_dict['mac_register_vector']['instance'][-1]

    # subarch-specific modules; absent from the architecture when the subarch is inactive.
    pwl_comparator_dim = architecture_dict['pwl_comparator']['instance'][-1] if 'pwl_comparator' in architecture_dict else 0
    pwl_encoder_dim = architecture_dict['pwl_encoder']['instance'][-1] if 'pwl_encoder' in architecture_dict else 0
    pwl_register_dim = architecture_dict['pwl_register']['instance'][-1] if 'pwl_register' in architecture_dict else 0
    pipeline_register_dim = architecture_dict['pipeline_register']['instance'][-1] if 'pipeline_register' in architecture_dict else 0
    adder_vector_dim = architecture_dict['adder_vector']['instance'][-1] if 'adder_vector' in architecture_dict else 0
    taylor_register_dim = architecture_dict['taylor_register']['instance'][-1] if 'taylor_register' in architecture_dict else 0

    if subarch == 'pwl':
        division_cycles = workload_dict['configuration']['approximate_division_cycles']
        cycle_count = workload_dict['configuration']['pwl_cycles'] + division_cycles
        multiplier_count = multiplier_vector_dim
        adder_vector_count = adder_vector_dim
        accumulator_count = 0 if is_silu else accumulator_vector_dim
        register_count = register_vector_dim * cycle_count
        mac_register_count = 0 if is_silu else accumulator_vector_dim
    elif subarch == 'taylor':
        division_cycles = workload_dict['configuration']['approximate_division_cycles']
        cycle_count = taylor_register_dim + division_cycles
        multiplier_count = multiplier_vector_dim * taylor_register_dim
        adder_vector_count = adder_vector_dim * taylor_register_dim
        accumulator_count = 0 if is_silu else accumulator_vector_dim
        register_count = register_vector_dim * cycle_count
        mac_register_count = 0 if is_silu else accumulator_vector_dim
    else:  # mac / figna
        exp_mult_cycles = workload_dict['configuration']['exp_mult_cycles']
        div_mult_cycles = workload_dict['configuration']['division_mult_cycles']
        accumulation_cycles = workload_dict['configuration']['accumulation_cycles']
        multiplication_cycles = exp_mult_cycles + div_mult_cycles
        cycle_count = multiplication_cycles + accumulation_cycles

        multiplier_count = multiplier_vector_dim * multiplication_cycles
        adder_vector_count = 0
        accumulator_count = accumulator_vector_dim * accumulation_cycles
        register_count = register_vector_dim * multiplication_cycles
        mac_register_count = mac_register_vector_dim * accumulation_cycles

    cycle_count = cycle_count / router_dim

    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    performance_dict['subevent'] = OrderedDict({
        'multiplier_vector': OrderedDict({'count': multiplier_count}),
        'accumulator_vector': OrderedDict({'count': accumulator_count}),
        'register_vector': OrderedDict({'count': register_count}),
        'mac_register_vector': OrderedDict({'count': mac_register_count}),
        'pwl_comparator': OrderedDict({'count': pwl_comparator_dim}),
        'pwl_encoder': OrderedDict({'count': pwl_encoder_dim}),
        'pwl_register': OrderedDict({'count': pwl_register_dim}),
        'pipeline_register': OrderedDict({'count': pipeline_register_dim}),
        'adder_vector': OrderedDict({'count': adder_vector_count}),
        'taylor_register': OrderedDict({'count': taylor_register_dim}),
    })
    return performance_dict


def softmax_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    return _vector_nonlinear(architecture_dict, workload_dict, is_silu=False)


def silu_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    return _vector_nonlinear(architecture_dict, workload_dict, is_silu=True)
