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

def input_reuse_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    workload_dict = next(iter(workload_dict.values()))['configuration']
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['multiplier']['query']['frequency']
    cycles = workload_dict['cycles']

    multiplier_dim = architecture_dict['multiplier']['instance'][-1]
    multiplier_register_dim = get_prod(architecture_dict['multiplier_register']['instance'][-2:])

    cycle_count = cycles / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    multiplier_dict = OrderedDict({'count': multiplier_dim * cycle_count})
    multiplier_register_dict = OrderedDict({'count': multiplier_register_dim * cycle_count})

    performance_dict['subevent'] = OrderedDict({'multiplier': multiplier_dict,
                                                'multiplier_register': multiplier_register_dict})
    return performance_dict

def weight_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['wfifo']['query']['frequency']

    wfifo_dim = architecture_dict['wfifo']['instance'][-1]
    magnitude_register_dim = architecture_dict['magnitude_register']['instance'][-1]
    sign_register_dim = architecture_dict['sign_register']['instance'][-1]
    sign_fifo_dim = architecture_dict['sign_fifo']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    wfifo_dict = OrderedDict({'count': wfifo_dim})
    magnitude_register_dict = OrderedDict({'count': magnitude_register_dim})
    sign_register_dict = OrderedDict({'count': sign_register_dim})
    sign_fifo_dict = OrderedDict({'count': sign_fifo_dim})

    performance_dict['subevent'] = OrderedDict({'wfifo': wfifo_dict,
                                                'magnitude_register': magnitude_register_dict,
                                                'sign_register': sign_register_dict,
                                                'sign_fifo': sign_fifo_dict})
    return performance_dict

def weight_reuse_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1

    frequency = architecture_dict['comparator']['query']['frequency']

    comparator_dim = architecture_dict['comparator']['instance'][-1]

    cycle_count = 1 / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    comparator_dict = OrderedDict({'count': comparator_dim})

    performance_dict['subevent'] = OrderedDict({'comparator': comparator_dict})
    return performance_dict

def array_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1
    cycles = next(iter(workload_dict.values()))['configuration']['cycles']

    frequency = architecture_dict['temporal_register']['query']['frequency']
    
    temporal_register_dim = get_prod(architecture_dict['temporal_register']['instance'][-2:])
    and_gate_dim = get_prod(architecture_dict['and_gate']['instance'][-2:])
    or_gate_dim = get_prod(architecture_dict['or_gate']['instance'][-3:-1])
    sign_xor_dim = architecture_dict['sign_xor']['instance'][-1]
    adder_dim = architecture_dict['adder']['instance'][-1]
    ofifo_dim = architecture_dict['ofifo']['instance'][-1]

    cycle_count = cycles / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})
    
    temporal_register_dict = OrderedDict({'count': temporal_register_dim})
    and_gate_dict = OrderedDict({'count': and_gate_dim})
    or_gate_dict = OrderedDict({'count': or_gate_dim})
    sign_xor_dict = OrderedDict({'count': sign_xor_dim * cycles})
    adder_dict = OrderedDict({'count': adder_dim * cycles})
    ofifo_dict = OrderedDict({'count': ofifo_dim * cycles})

    performance_dict['subevent'] = OrderedDict({'temporal_register': temporal_register_dict,
                                                'and_gate': and_gate_dict,
                                                'or_gate': or_gate_dict,
                                                'sign_xor': sign_xor_dict,
                                                'adder': adder_dict,
                                                'ofifo': ofifo_dict})

    return performance_dict

def array_fifo_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    router_dim = get_prod(architecture_dict['irouter']['instance']) if 'irouter' in architecture_dict else 1
    cycles = next(iter(workload_dict.values()))['configuration']['cycles']

    frequency = architecture_dict['pe_fifo']['query']['frequency']

    or_gate_dim = get_prod(architecture_dict['or_gate']['instance'][-3:1])
    pe_fifo_dim = architecture_dict['pe_fifo']['instance'][-1]

    cycle_count = cycles / router_dim
    performance_dict['cycle_count'] = OrderedDict({'value': cycle_count, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycle_count / 1000 / frequency, 'unit': 'ms'})

    or_gate_dict = OrderedDict({'count': or_gate_dim})
    pe_fifo_dict = OrderedDict({'count': pe_fifo_dim * cycles})

    performance_dict['subevent'] = OrderedDict({'or_gate': or_gate_dict,
                                                'pe_fifo': pe_fifo_dict})
    return performance_dict

def nonlinear_gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    
    performance_dict['cycle_count'] = OrderedDict({'value': 0, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 0, 'unit': 'ms'})

    round_dict = OrderedDict({'count': 0})
    sign_mantissa_register_dict = OrderedDict({'count': 0})
    exponent_register_dict = OrderedDict({'count': 0})
    exp_clamp_dict = OrderedDict({'count': 0})
    exp_norm_dict = OrderedDict({'count': 0})
    exp_select_dict = OrderedDict({'count': 0})
    max_tree_dict = OrderedDict({'count': 0})
    max_tree_register_dict = OrderedDict({'count': 0})

    if 'window_select' in architecture_dict:
        window_select_dict = OrderedDict({'count': 0})
    if 'lut_register' in architecture_dict:
        lut_register_dict = OrderedDict({'count': 0})
    if 'lut_decoder' in architecture_dict:
        lut_decoder_dict = OrderedDict({'count': 0})

    performance_dict['subevent'] = OrderedDict({'round': round_dict,
                                                 'sign_mantissa_register': sign_mantissa_register_dict,
                                                 'exponent_register': exponent_register_dict,
                                                 'exp_clamp': exp_clamp_dict,
                                                 'exp_norm': exp_norm_dict,
                                                 'exp_select': exp_select_dict,
                                                 'max_tree': max_tree_dict,
                                                 'max_tree_register': max_tree_register_dict})
    
    if 'window_select' in architecture_dict:
        performance_dict['subevent']['window_select'] = window_select_dict
    if 'lut_register' in architecture_dict:
        performance_dict['subevent']['lut_register'] = lut_register_dict
    if 'lut_decoder' in architecture_dict:
        performance_dict['subevent']['lut_decoder'] = lut_decoder_dict
        
    return performance_dict