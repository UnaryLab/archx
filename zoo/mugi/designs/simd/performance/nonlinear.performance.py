from collections import OrderedDict
from archx.utils import get_prod

# NOTE: The two subarchitectures (mac, figna) are folded into a single AGraph design.
# Every subarch-specific module (int_to_fp for mac; ch_aloc/ch_dealoc/int_to_fp_figna/
# prealigner for figna) is always present in architecture_dict; inactive ones are
# instance-gated to 0 (so their area and instance-derived counts vanish). gemm_nonlinear
# zeroes every module count, so it is agnostic to the active subarch. mac and figna share
# the same vector nonlinear cycle formula, so vector_nonlinear needs no subarch branch.


def gemm_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    performance_dict['cycle_count'] = OrderedDict({'value': 0, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 0, 'unit': 'ms'})

    # Every module in the (superset) gemm_nonlinear event contributes zero here.
    zero_modules = ['ch_aloc', 'ch_dealoc', 'int_to_fp_figna', 'prealigner', 'int_to_fp',
                    'ififo', 'wfifo', 'input_register', 'weight_register', 'multiplier',
                    'pe_register', 'accumulator', 'accumulator_register', 'adder', 'ofifo']
    performance_dict['subevent'] = OrderedDict((name, OrderedDict({'count': 0})) for name in zero_modules)
    return performance_dict


def vector_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    accumulator_vector_dim = architecture_dict['accumulator_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]
    mac_register_vector_dim = architecture_dict['mac_register_vector']['instance'][-1]

    exp_mult_cycles = workload_dict['configuration']['exp_mult_cycles']
    div_mult_cycles = workload_dict['configuration']['division_mult_cycles']
    accumulation_cycles = workload_dict['configuration']['accumulation_cycles']

    multiplication_cycles = exp_mult_cycles + div_mult_cycles
    cycle_count = (multiplication_cycles + accumulation_cycles) / router_dim

    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    performance_dict['subevent'] = OrderedDict({
        'multiplier_vector': OrderedDict({'count': multiplier_vector_dim * multiplication_cycles}),
        'accumulator_vector': OrderedDict({'count': accumulator_vector_dim * accumulation_cycles}),
        'register_vector': OrderedDict({'count': register_vector_dim * multiplication_cycles}),
        'mac_register_vector': OrderedDict({'count': mac_register_vector_dim * accumulation_cycles}),
    })
    return performance_dict
