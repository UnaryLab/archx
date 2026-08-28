from collections import OrderedDict

from chiplet4ai.common.performance.mapping import _buffer_elements


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
        'sram_output_read': {'count': array_height * array_width, 'factor': {'cycle_count': 0, 'runtime': 0, 'dynamic_energy': 0}},
        'sram_output_write': {'count': array_height * array_width, 'factor': {'cycle_count': 0, 'runtime': 0, 'dynamic_energy': 0}},
    })

    return performance_dict

def sram_input_write_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    isram_elements = _active_elements(architecture_dict, 'isram')

    performance_dict['subevent'] = OrderedDict({
        'sram_input_write': {'count': isram_elements, 'factor': {'cycle_count': 1/isram_elements, 'runtime': 1/isram_elements}},
    })
    return performance_dict

def sram_weight_write_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    wsram_elements = _active_elements(architecture_dict, 'wsram')

    performance_dict['subevent'] = OrderedDict({
        'sram_weight_write': {'count': wsram_elements, 'factor': {'cycle_count': 1/wsram_elements, 'runtime': 1/wsram_elements}},
    })
    return performance_dict

def sram_output_write_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    osram_elements = _active_elements(architecture_dict, 'osram')

    performance_dict['subevent'] = OrderedDict({
        'sram_output_write': {'count': osram_elements, 'factor': {'cycle_count': 1/osram_elements, 'runtime': 1/osram_elements}},
    })
    return performance_dict

def sram_output_read_mapping(architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    performance_dict = OrderedDict()

    # architecture parameters
    osram_elements = _active_elements(architecture_dict, 'osram')

    performance_dict['subevent'] = OrderedDict({
        'sram_output_read': {'count': osram_elements, 'factor': {'cycle_count': 1/osram_elements, 'runtime': 1/osram_elements}},
    })
    return performance_dict
