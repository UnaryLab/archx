from collections import OrderedDict

def llama_2_7b(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = llama_2(architecture_dict=architecture_dict,
                               workload_dict=workload_dict)
    return performance_dict

def llama_2_13b(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = llama_2(architecture_dict=architecture_dict,
                               workload_dict=workload_dict)
    return performance_dict

def llama_2_70b(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = llama_2(architecture_dict=architecture_dict,
                               workload_dict=workload_dict)
    return performance_dict

def llama_2_70b_GQA(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = llama_2(architecture_dict=architecture_dict,
                               workload_dict=workload_dict)
    return performance_dict

def llama_2(architecture_dict: OrderedDict, workload_dict: OrderedDict=None) -> OrderedDict:
    performance_dict = OrderedDict()

    gemm_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})
    nonlinear_dict = OrderedDict({'count': 1, 'aggregation': 'sequential'})

    performance_dict['subevent'] = OrderedDict({
        'gemm': gemm_dict,
        'nonlinear': nonlinear_dict
    })

    return performance_dict