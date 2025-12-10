from collections import OrderedDict
from archx.utils import get_prod

def gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    m = workload_dict['gemm']['configuration']['m']
    k = workload_dict['gemm']['configuration']['k']
    n = workload_dict['gemm']['configuration']['n']
    array_dim = workload_dict['gemm']['configuration']['array_dim']
    bitwidth = workload_dict['gemm']['configuration']['bitwidth']

    input_mappings = (m / array_dim) * (k / array_dim) * (n)
    weight_mappings = (m / array_dim) * (k / array_dim) * (n)
    # weight_mappings = (k / array_dim) * (n)

    multiplier_dim = get_prod(architecture_dict['multiplier']['instance'])
    adder_dim = get_prod(architecture_dict['adder']['instance'])
    act_en_reg_dim = get_prod(architecture_dict['act_en_reg']['instance'])
    mult_en_reg_dim = get_prod(architecture_dict['mult_en_reg']['instance'])
    acc_en_reg_dim = get_prod(architecture_dict['acc_en_reg']['instance'])
    weight_path_en_reg_dim = get_prod(architecture_dict['weight_path_en_reg']['instance'])
    weight_en_reg_dim = get_prod(architecture_dict['weight_en_reg']['instance'])
    sum_en_reg_dim = get_prod(architecture_dict['sum_en_reg']['instance'])
    act_reg_dim = get_prod(architecture_dict['act_reg']['instance'])
    weight_path_reg_dim = get_prod(architecture_dict['weight_path_reg']['instance'])
    sum_reg_dim = get_prod(architecture_dict['sum_reg']['instance'])
    weight_reg_dim = get_prod(architecture_dict['weight_reg']['instance'])
    act_mux_dim = get_prod(architecture_dict['act_mux']['instance'])
    weight_mux_dim = get_prod(architecture_dict['weight_mux']['instance'])
    add_mux_dim = get_prod(architecture_dict['add_mux']['instance'])
    sum_mux_dim = get_prod(architecture_dict['sum_mux']['instance'])

    frequency = architecture_dict['multiplier']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    multiplier_dict = OrderedDict({'count': multiplier_dim})
    adder_dict = OrderedDict({'count': adder_dim})
    act_en_reg_dict = OrderedDict({'count': act_en_reg_dim})
    mult_en_reg_dict = OrderedDict({'count': mult_en_reg_dim})
    acc_en_reg_dict = OrderedDict({'count': acc_en_reg_dim})
    weight_path_en_reg_dict = OrderedDict({'count': weight_path_en_reg_dim})
    weight_en_reg_dict = OrderedDict({'count': weight_en_reg_dim})
    sum_en_reg_dict = OrderedDict({'count': sum_en_reg_dim})
    act_reg_dict = OrderedDict({'count': act_reg_dim})
    weight_path_reg_dict = OrderedDict({'count': weight_path_reg_dim})
    sum_reg_dict = OrderedDict({'count': sum_reg_dim})
    weight_reg_dict = OrderedDict({'count': weight_reg_dim})
    act_mux_dict = OrderedDict({'count': act_mux_dim})
    weight_mux_dict = OrderedDict({'count': weight_mux_dim})
    add_mux_dict = OrderedDict({'count': add_mux_dim})
    sum_mux_dict = OrderedDict({'count': sum_mux_dim})

    performance_dict['subevent'] = OrderedDict({'multiplier': multiplier_dict,
                                                'adder': adder_dict,
                                                'act_en_reg': act_en_reg_dict,
                                                'mult_en_reg': mult_en_reg_dict,
                                                'acc_en_reg': acc_en_reg_dict,
                                                'weight_path_en_reg': weight_path_en_reg_dict,
                                                'weight_en_reg': weight_en_reg_dict,
                                                'sum_en_reg': sum_en_reg_dict,
                                                'act_reg': act_reg_dict,
                                                'weight_path_reg': weight_path_reg_dict,
                                                'sum_reg': sum_reg_dict,
                                                'weight_reg': weight_reg_dict,
                                                'act_mux': act_mux_dict,
                                                'weight_mux': weight_mux_dict,
                                                'add_mux': add_mux_dict,
                                                'sum_mux': sum_mux_dict})

    return performance_dict