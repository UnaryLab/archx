from collections import OrderedDict
from archx.utils import get_prod

def gemm_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    performance_dict['cycle_count'] = OrderedDict({'value': 0, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 0, 'unit': 'ms'})

    ififo_dict = OrderedDict({'count': 0})
    accumulator_dict = OrderedDict({'count': 0})
    accumulator_register_dict = OrderedDict({'count': 0})
    exp_scale_dict = OrderedDict({'count': 0})
    wreg_dict = OrderedDict({'count': 0})
    counter_dict = OrderedDict({'count': 0})
    counter_register_dict = OrderedDict({'count': 0})
    wfifo_dict = OrderedDict({'count': 0})
    comparator_dict = OrderedDict({'count': 0})
    creg_dict = OrderedDict({'count': 0})
    ireg_dict = OrderedDict({'count': 0})
    areg_dict = OrderedDict({'count': 0})
    z_flag_dict = OrderedDict({'count': 0})
    nan_flag_dict = OrderedDict({'count': 0})
    sign_flag_dict = OrderedDict({'count': 0})
    temporal_register_dict = OrderedDict({'count': 0})
    and_gate_dict = OrderedDict({'count': 0})
    or_gate_dict = OrderedDict({'count': 0})
    pe_fifo_dict = OrderedDict({'count': 0})
    sign_xor_dict = OrderedDict({'count': 0})
    shifterexp_dict = OrderedDict({'count': 0})
    sign_fifo_dict = OrderedDict({'count': 0})
    adder_dict = OrderedDict({'count': 0})
    ofifo_dict = OrderedDict({'count': 0})

    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict,
                                                'accumulator': accumulator_dict,
                                                'accumulator_register': accumulator_register_dict,
                                                'exp_scale': exp_scale_dict,
                                                'wreg': wreg_dict,
                                                'counter': counter_dict,
                                                'counter_register': counter_register_dict,
                                                'wfifo': wfifo_dict,
                                                'comparator': comparator_dict,
                                                'creg': creg_dict,
                                                'ireg': ireg_dict,
                                                'areg': areg_dict,
                                                'z_flag': z_flag_dict,
                                                'nan_flag': nan_flag_dict,
                                                'sign_flag': sign_flag_dict,
                                                'sign_fifo': sign_fifo_dict,
                                                'temporal_register': temporal_register_dict,
                                                'and_gate': and_gate_dict,
                                                'or_gate': or_gate_dict,
                                                'pe_fifo': pe_fifo_dict,
                                                'sign_xor': sign_xor_dict,
                                                'shifterexp': shifterexp_dict,
                                                'adder': adder_dict,
                                                'ofifo': ofifo_dict})
    return performance_dict

def vector_nonlinear(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier_vector']['query']['frequency']
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
    
    multiplier_vector_dict = OrderedDict({'count': multiplier_vector_dim * multiplication_cycles})
    accumulator_vector_dict = OrderedDict({'count': accumulator_vector_dim * accumulation_cycles})
    register_vector_dict = OrderedDict({'count': register_vector_dim * multiplication_cycles})
    mac_register_vector_dict = OrderedDict({'count': mac_register_vector_dim * accumulation_cycles})

    performance_dict['subevent'] = OrderedDict({'multiplier_vector': multiplier_vector_dict,
                                                'accumulator_vector': accumulator_vector_dict,
                                                'register_vector': register_vector_dict,
                                                'mac_register_vector': mac_register_vector_dict})
    return performance_dict