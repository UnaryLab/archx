from collections import OrderedDict
from archx.utils import get_prod

def gpt2(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    performance_dict['subevent'] = OrderedDict({
        'addmm': OrderedDict({'count': 1}),
        'softmax': OrderedDict({'count': 1}),
        'add': OrderedDict({'count': 1}),
        'tanh': OrderedDict({'count': 1}),
        'bmm': OrderedDict({'count': 1}),
        'where': OrderedDict({'count': 1}),
        'mul': OrderedDict({'count': 1}),
        'div': OrderedDict({'count': 1}),
        'gather': OrderedDict({'count': 1}),
        'cat': OrderedDict({'count': 1}),
        'copy': OrderedDict({'count': 1}),
        'native_layer_norm': OrderedDict({'count': 1}),
        'bitwise_and': OrderedDict({'count': 1}),
        'mm': OrderedDict({'count': 1}),
        'le': OrderedDict({'count': 1}),
        'argmax': OrderedDict({'count': 1}),
        'masked_fill': OrderedDict({'count': 1}),
        'sub': OrderedDict({'count': 1}),
        'gt': OrderedDict({'count': 1}),
        'cumsum': OrderedDict({'count': 1}),
        'index': OrderedDict({'count': 1}),
        'any': OrderedDict({'count': 1}),
        'fill': OrderedDict({'count': 1}),
        'sum': OrderedDict({'count': 1}),
        'eq': OrderedDict({'count': 1}),
        'arange': OrderedDict({'count': 1}),
        'bitwise_not': OrderedDict({'count': 1}),
        'ge': OrderedDict({'count': 1})
    })
    return performance_dict

def addmm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0008': OrderedDict({'count': 1}),
        'k0010': OrderedDict({'count': 1}),
        'k0011': OrderedDict({'count': 1}),
        'k0020': OrderedDict({'count': 1})
    })
    return performance_dict

def softmax(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0004': OrderedDict({'count': 1})
    })
    return performance_dict

def add(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0001': OrderedDict({'count': 1}),
        'k0014': OrderedDict({'count': 1}),
        'k0017': OrderedDict({'count': 1}),
        'k0016': OrderedDict({'count': 1}),
        'k0058': OrderedDict({'count': 1}),
        'k0059': OrderedDict({'count': 1})
    })
    return performance_dict
    
def tanh(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0015': OrderedDict({'count': 1})
    })
    return performance_dict
    
def bmm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0009': OrderedDict({'count': 1}),
        'k0007': OrderedDict({'count': 1})
    })
    return performance_dict
    
def where(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0002': OrderedDict({'count': 1}),
        'k0022': OrderedDict({'count': 1})
    })
    return performance_dict
    
def mul(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0005': OrderedDict({'count': 1}),
        'k0013': OrderedDict({'count': 1}),
        'k0033': OrderedDict({'count': 1}),
        'k0048': OrderedDict({'count': 1})
    })
    return performance_dict
    
def div(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0003': OrderedDict({'count': 1})
    })
    return performance_dict
    
def gather(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0026': OrderedDict({'count': 1}),
        'k0025': OrderedDict({'count': 1})
    })
    return performance_dict
    
def cat(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0006': OrderedDict({'count': 1}),
        'k0035': OrderedDict({'count': 1})
    })
    return performance_dict
    
def copy(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0019': OrderedDict({'count': 1}),
        'k0018': OrderedDict({'count': 1}),
        'k0029': OrderedDict({'count': 1}),
        'k0036': OrderedDict({'count': 1}),
        'k0037': OrderedDict({'count': 1}),
        'k0052': OrderedDict({'count': 1})
    })
    return performance_dict
    
def native_layer_norm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0012': OrderedDict({'count': 1})
    })
    return performance_dict
    
def bitwise_and(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0021': OrderedDict({'count': 1}),
        'k0034': OrderedDict({'count': 1}),
        'k0049': OrderedDict({'count': 1})
    })
    return performance_dict
    
def mm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0023': OrderedDict({'count': 1})
    })
    return performance_dict
    
def le(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0031': OrderedDict({'count': 1})
    })
    return performance_dict
    
def argmax(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0027': OrderedDict({'count': 1})
    })
    return performance_dict
    
def masked_fill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0055': OrderedDict({'count': 1})
    })
    return performance_dict
    
def sub(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0053': OrderedDict({'count': 1}),
        'k0042': OrderedDict({'count': 1}),
        'k0056': OrderedDict({'count': 1})
    })
    return performance_dict
    
def gt(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0024': OrderedDict({'count': 1})
    })
    return performance_dict
    
def cumsum(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0028': OrderedDict({'count': 1}),
        'k0040': OrderedDict({'count': 1}),
        'k0062': OrderedDict({'count': 1})
    })
    return performance_dict
    
def index(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0030': OrderedDict({'count': 1})
    })
    return performance_dict
    
def any(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0032': OrderedDict({'count': 1})
    })
    return performance_dict
    
def fill(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0038': OrderedDict({'count': 1}),
        'k0043': OrderedDict({'count': 1}),
        'k0046': OrderedDict({'count': 1}),
        'k0061': OrderedDict({'count': 1})
    })
    return performance_dict
    
def sum(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0041': OrderedDict({'count': 1})
    })
    return performance_dict
    
def eq(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0044': OrderedDict({'count': 1}),
        'k0050': OrderedDict({'count': 1}),
        'k0051': OrderedDict({'count': 1}),
        'k0054': OrderedDict({'count': 1})
    })
    return performance_dict
    
def arange(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0045': OrderedDict({'count': 1}),
        'k0047': OrderedDict({'count': 1})
    })
    return performance_dict
    
def bitwise_not(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0057': OrderedDict({'count': 1})
    })
    return performance_dict
    
def ge(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict({
        'k0060': OrderedDict({'count': 1})
    })
    return performance_dict
    