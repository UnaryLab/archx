from collections import OrderedDict
from archx.utils import get_prod

def gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    ai_chiplet_compute = 1
    ai_weight_compute = 1
    accumulator_chiplet = 1
    vector_unit_chiplet = 1
    memory_weight_reads = 1
    memory_compute_reads = 1
    memory_weight_reads = 1
    
    performance_dict['subevent'] = {'ai_chiplet_compute': ai_chiplet_compute,
                                    'ai_weight_compute': ai_weight_compute,
                                    'accumulator_chiplet': accumulator_chiplet,
                                    'vector_unit_chiplet': vector_unit_chiplet,
                                    'memory_weight_reads': memory_weight_reads,
                                    'memory_compute_reads': memory_compute_reads,
                                    'memory_weight_reads': memory_weight_reads}
    
    return performance_dict