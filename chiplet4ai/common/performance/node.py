from collections import OrderedDict

def array_input_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    array_height = architecture_dict['pe']['instance'][0]
    array_width = architecture_dict['pe']['instance'][1]
    
    performance_dict['subevent'] = OrderedDict({
        'array_input': {'count': array_width, 'factor': {'cycle_count': 1/(array_width), 'runtime': 1/(array_width)}},
        'sram_input_read': {'count': array_height * array_width, 'factor': {'cycle_count': 0, 'runtime': 0}},
    })

    return performance_dict

def array_weight_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    array_height = architecture_dict['pe']['instance'][0]
    array_width = architecture_dict['pe']['instance'][1]
    
    performance_dict['subevent'] = OrderedDict({
        'array_weight': {'count': array_height, 'factor': {'cycle_count': 1/(array_height), 'runtime': 1/(array_height)}},
        'sram_weight_read': {'count': array_height * array_width, 'factor': {'cycle_count': 0, 'runtime': 0}},
    })

    return performance_dict

def array_compute_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    array_height = architecture_dict['pe']['instance'][0]
    array_width = architecture_dict['pe']['instance'][1]
    
    performance_dict['subevent'] = OrderedDict({
        'array_compute': {'count': array_height * array_width, 'factor': {'cycle_count': 1/(array_height * array_width), 'runtime': 1/(array_height * array_width)}},
        'sram_output_read': {'count': array_height * array_width, 'factor': {'cycle_count': 0, 'runtime': 0}},
        'sram_output_write': {'count': array_height * array_width, 'factor': {'cycle_count': 0, 'runtime': 0}},
    })

    return performance_dict

def sram_input_write_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    isram_bank = architecture_dict['isram']['query']['bank'] / 2
    isram_depth = architecture_dict['isram']['query']['depth']
    
    performance_dict['subevent'] = OrderedDict({
        'sram_input_write': {'count': isram_depth * isram_bank, 'factor': {'cycle_count': 1/(isram_depth * isram_bank), 'runtime': 1/(isram_depth * isram_bank)}},
    })

    return performance_dict

def sram_weight_write_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    wsram_bank = architecture_dict['wsram']['query']['bank'] / 2
    wsram_depth = architecture_dict['wsram']['query']['depth']
    
    performance_dict['subevent'] = OrderedDict({
        'sram_weight_write': {'count': wsram_depth * wsram_bank, 'factor': {'cycle_count': 1/(wsram_depth * wsram_bank), 'runtime': 1/(wsram_depth * wsram_bank)}},
    })

    return performance_dict

def sram_output_write_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    osram_bank = architecture_dict['osram']['query']['bank'] / 2
    osram_depth = architecture_dict['osram']['query']['depth']
    
    performance_dict['subevent'] = OrderedDict({
        'sram_output_write': {'count': osram_depth * osram_bank, 'factor': {'cycle_count': 1/(osram_depth * osram_bank), 'runtime': 1/(osram_depth * osram_bank)}},
    })

    return performance_dict

def sram_output_read_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    osram_bank = architecture_dict['osram']['query']['bank'] / 2
    osram_depth = architecture_dict['osram']['query']['depth']
    
    performance_dict['subevent'] = OrderedDict({
        'sram_output_read': {'count': osram_depth * osram_bank, 'factor': {'cycle_count': 1/(osram_depth * osram_bank), 'runtime': 1/(osram_depth * osram_bank)}},
    })

    return performance_dict
