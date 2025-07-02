from collections import OrderedDict
from archx.utils import get_prod

def input_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['ififo']['query']['frequency']

    ififo_dim = architecture_dict['ififo']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    ififo_dict = OrderedDict({'count': ififo_dim})

    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict})
    return performance_dict

def counter_reuse(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['counter']['query']['frequency']
    counter_dim = architecture_dict['counter']['instance'][-1]
    creg_dim = get_prod(architecture_dict['creg']['instance'][-2:])

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    counter_dict = OrderedDict({'count': counter_dim})
    creg_dict = OrderedDict({'count': creg_dim})

    performance_dict['subevent'] = OrderedDict({'counter': counter_dict,
                                                'creg': creg_dict})
    return performance_dict

def input_reuse_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['accumulator']['query']['frequency']

    accumulator_dim = architecture_dict['accumulator']['instance'][-1]
    accumulator_register_dim = get_prod(architecture_dict['accumulator_register']['instance'][-2:])
    exp_scale_dim = architecture_dict['exp_scale']['instance'][-1]
    wreg_dim = architecture_dict['wreg']['instance'][-1]

    cycle_count = 8 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    accumulator_dict = OrderedDict({'count': accumulator_dim * cycle_count})
    accumulator_register_dict = OrderedDict({'count': accumulator_register_dim * cycle_count})
    exp_scale_dict = OrderedDict({'count': exp_scale_dim * cycle_count})
    wreg_dict = OrderedDict({'count': wreg_dim * cycle_count})

    performance_dict['subevent'] = OrderedDict({'accumulator': accumulator_dict,
                                                 'accumulator_register': accumulator_register_dict,
                                                 'exp_scale': exp_scale_dict,
                                                 'wreg': wreg_dict})
    return performance_dict

def weight_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['wfifo']['query']['frequency']

    wfifo_dim = architecture_dict['wfifo']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    wfifo_dict = OrderedDict({'count': wfifo_dim})
    
    performance_dict['subevent'] = OrderedDict({'wfifo': wfifo_dict})
    return performance_dict

def weight_reuse_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['comparator']['query']['frequency']

    comparator_dim = architecture_dict['comparator']['instance'][-1]
    ireg_dim = architecture_dict['ireg']['instance'][-1]
    z_flag_dim = architecture_dict['z_flag']['instance'][-1]
    nan_flag_dim = architecture_dict['nan_flag']['instance'][-1]
    sign_fifo_dim = architecture_dict['sign_fifo']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    comparator_dict = OrderedDict({'count': comparator_dim})
    ireg_dict = OrderedDict({'count': ireg_dim})
    z_flag_dict = OrderedDict({'count': z_flag_dim})
    nan_flag_dict = OrderedDict({'count': nan_flag_dim})
    sign_fifo_dict = OrderedDict({'count': sign_fifo_dim})

    performance_dict['subevent'] = OrderedDict({'comparator': comparator_dict,
                                                'ireg': ireg_dict,
                                                'z_flag': z_flag_dict,
                                                'nan_flag': nan_flag_dict,
                                                'sign_fifo': sign_fifo_dict})
    return performance_dict

def array_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['temporal_register']['query']['frequency']
    
    temporal_register_dim = get_prod(architecture_dict['temporal_register']['instance'][-2:])
    and_gate_dim = get_prod(architecture_dict['and_gate']['instance'][-2:])
    or_gate_dim = get_prod(architecture_dict['or_gate']['instance'][-2:])
    areg_dim = get_prod(architecture_dict['areg']['instance'][-2:])
    pe_fifo_dim = architecture_dict['pe_fifo']['instance'][-1]
    sign_xor_dim = architecture_dict['sign_xor']['instance'][-1]
    shifterexp = architecture_dict['shifterexp']['instance'][-1]
    adder_dim = architecture_dict['adder']['instance'][-1]
    ofifo_dim = architecture_dict['ofifo']['instance'][-1]

    cycle_count = 8 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    temporal_register_dict = OrderedDict({'count': temporal_register_dim})
    and_gate_dict = OrderedDict({'count': and_gate_dim})
    or_gate_dict = OrderedDict({'count': or_gate_dim})
    areg_dict = OrderedDict({'count': areg_dim})
    pe_fifo_dict = OrderedDict({'count': pe_fifo_dim * cycle_count})
    sign_xor_dict = OrderedDict({'count': sign_xor_dim * cycle_count})
    shifterexp_dict = OrderedDict({'count': shifterexp * cycle_count})
    adder_dict = OrderedDict({'count': adder_dim * cycle_count})
    ofifo_dict = OrderedDict({'count': ofifo_dim * cycle_count})

    performance_dict['subevent'] = OrderedDict({'temporal_register': temporal_register_dict,
                                                 'and_gate': and_gate_dict,
                                                 'or_gate': or_gate_dict,
                                                 'areg': areg_dict,
                                                 'pe_fifo': pe_fifo_dict,
                                                 'sign_xor': sign_xor_dict,
                                                 'shifterexp': shifterexp_dict,
                                                 'adder': adder_dict,
                                                 'ofifo': ofifo_dict})
    return performance_dict

def vector_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier_vector']['query']['frequency']
    multiplier_vector_dim = architecture_dict['multiplier_vector']['instance'][-1]
    register_vector_dim = architecture_dict['register_vector']['instance'][-1]
    mac_register_vector_dim = architecture_dict['mac_register_vector']['instance'][-1]

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
    return performance_dict