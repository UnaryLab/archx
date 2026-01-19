from collections import OrderedDict
from archx.utils import get_prod

# TODO: separate ai_chiplet from weight and activation

def ai_chiplet_compute(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    ififo_inst = architecture_dict['ififo']['instance']
    wfifo_inst = architecture_dict['wfifo']['instance']
    ofifo_inst = architecture_dict['ofifo']['instance']
    multiplier_inst = architecture_dict['multiplier']['instance']
    adder_inst = architecture_dict['adder']['instance']
    act_en_reg_inst= architecture_dict['act_en_reg']['instance']
    mult_en_reg_inst= architecture_dict['mult_en_reg']['instance']
    acc_en_reg_inst= architecture_dict['acc_en_reg']['instance']
    sum_en_reg_inst= architecture_dict['sum_en_reg']['instance']
    act_reg_inst= architecture_dict['act_reg']['instance']
    sum_reg_inst= architecture_dict['sum_reg']['instance']
    act_mux_inst = architecture_dict['act_mux']['instance']
    add_mux_inst = architecture_dict['add_mux']['instance']
    sum_mux_inst = architecture_dict['sum_mux']['instance']

    frequency = architecture_dict['isram']['query']['frequency']

    cycles = multiplier_inst[1] * multiplier_inst[2]
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    ififo_dict = OrderedDict({'count': get_prod(ififo_inst) * cycles})
    wfifo_dict = OrderedDict({'count': get_prod(wfifo_inst) * cycles})
    ofifo_dict = OrderedDict({'count': get_prod(ofifo_inst) * cycles})
    multiplier_dict = OrderedDict({'count': get_prod(multiplier_inst) * cycles})
    adder_dict = OrderedDict({'count': get_prod(adder_inst) * cycles})
    act_en_reg_dict = OrderedDict({'count': get_prod(act_en_reg_inst) * cycles})
    mult_en_reg_dict = OrderedDict({'count': get_prod(mult_en_reg_inst) * cycles})
    acc_en_reg_dict = OrderedDict({'count': get_prod(acc_en_reg_inst) * cycles})
    sum_en_reg_dict = OrderedDict({'count': get_prod(sum_en_reg_inst) * cycles})
    act_reg_dict = OrderedDict({'count': get_prod(act_reg_inst) * cycles})
    sum_reg_dict = OrderedDict({'count': get_prod(sum_reg_inst) * cycles})
    act_mux_dict = OrderedDict({'count': get_prod(act_mux_inst) * cycles})
    add_mux_dict = OrderedDict({'count': get_prod(add_mux_inst) * cycles})
    sum_mux_dict = OrderedDict({'count': get_prod(sum_mux_inst) * cycles})

    performance_dict['subevent'] = OrderedDict({'ififo': ififo_dict,
                                                'wfifo': wfifo_dict,
                                                'ofifo': ofifo_dict,
                                                'multiplier': multiplier_dict,
                                                'adder': adder_dict,
                                                'act_en_reg': act_en_reg_dict,
                                                'mult_en_reg': mult_en_reg_dict,
                                                'acc_en_reg': acc_en_reg_dict,
                                                'sum_en_reg': sum_en_reg_dict,
                                                'act_reg': act_reg_dict,
                                                'sum_reg': sum_reg_dict,
                                                'act_mux': act_mux_dict,
                                                'add_mux': add_mux_dict,
                                                'sum_mux': sum_mux_dict})
    return performance_dict

def ai_chiplet_weight(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    weight_path_en_reg_inst = architecture_dict['weight_path_en_reg']['instance']
    weight_en_reg_inst = architecture_dict['weight_en_reg']['instance']
    weight_path_reg_inst = architecture_dict['weight_path_reg']['instance']
    weight_reg_inst = architecture_dict['weight_reg']['instance']
    weight_mux_inst = architecture_dict['weight_mux']['instance']

    frequency = architecture_dict['multiplier']['query']['frequency']

    cycles = weight_path_en_reg_inst[1] * weight_path_en_reg_inst[2]
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    weight_path_en_reg_dict = OrderedDict({'count': get_prod(weight_path_en_reg_inst) * cycles})
    weight_en_reg_dict = OrderedDict({'count': get_prod(weight_en_reg_inst) * cycles})
    weight_path_reg_dict = OrderedDict({'count': get_prod(weight_path_reg_inst) * cycles})
    weight_reg_dict = OrderedDict({'count': get_prod(weight_reg_inst) * cycles})
    weight_mux_dict = OrderedDict({'count': get_prod(weight_mux_inst) * cycles})

    performance_dict['subevent'] = OrderedDict({'weight_path_en_reg': weight_path_en_reg_dict,
                                                'weight_en_reg': weight_en_reg_dict,
                                                'weight_path_reg': weight_path_reg_dict,
                                                'weight_reg': weight_reg_dict,
                                                'weight_mux': weight_mux_dict})
    return performance_dict

def accumulator_chiplet(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    accumulator_inst = architecture_dict['accumulator']['instance']
    accumulator_reg_inst = architecture_dict['accumulator_reg']['instance']

    frequency = architecture_dict['accumulator']['query']['frequency']

    cycles = 1
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    accumulator_dict = OrderedDict({'count': get_prod(accumulator_inst) * cycles})
    accumulator_reg_dict = OrderedDict({'count': get_prod(accumulator_reg_inst) * cycles})

    return performance_dict

def vector_unit_chiplet(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    vector_multiplier_inst = architecture_dict['vector_multiplier']['instance']
    vector_adder_inst = architecture_dict['vector_adder']['instance']
    vector_reg_inst = architecture_dict['vector_reg']['instance']
    
    frequency = architecture_dict['vector_multiplier']['query']['frequency']

    cycles = 1
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    vector_multiplier_dict = OrderedDict({'count': get_prod(vector_multiplier_inst) * cycles})
    vector_adder_dict = OrderedDict({'count': get_prod(vector_adder_inst) * cycles})
    vector_reg_dict = OrderedDict({'count': get_prod(vector_reg_inst) * cycles})

    performance_dict['subevent'] = OrderedDict({'vector_multiplier': vector_multiplier_dict,
                                                'vector_adder': vector_adder_dict,
                                                'vector_reg': vector_reg_dict})
    return performance_dict

def memory_weight_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    
    banks = get_prod(architecture_dict['sram']['bank'])

    cycles = banks
    frequency = architecture_dict['sram']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'sram': sram_dict})
    return performance_dict

def memory_compute_reads(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    
    banks = get_prod(architecture_dict['sram']['bank'])

    cycles = banks
    frequency = architecture_dict['sram']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'read'})})
    performance_dict['subevent'] = OrderedDict({'sram': sram_dict})
    return performance_dict

def memory_compute_writes(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    
    banks = get_prod(architecture_dict['sram']['bank'])

    cycles = banks
    frequency = architecture_dict['sram']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    sram_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'write'})})
    performance_dict['subevent'] = OrderedDict({'sram': sram_dict})
    return performance_dict