from collections import OrderedDict
from archx.utils import get_prod
from math import floor

def alexnet(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict()

    conv_0 =  {'layer': 'conv',  'input': [224, 224, 3], 'weight': [11, 11, 3], 'stride': 4, 'pad': 0, 'filter': 96,  'output': [54, 54, 96]}
    pool_0 =  {'layer': 'pool',  'input': [54, 54, 96],  'weight': [3, 3, 96],  'stride': 2, 'pad': 0,                'output': [26, 26, 96]}
    conv_1 =  {'layer': 'conv',  'input': [26, 26, 96],  'weight': [5, 5, 96],  'stride': 1, 'pad': 2, 'filter': 256, 'output': [26, 26, 256]}
    pool_1 =  {'layer': 'pool',  'input': [26, 26, 256], 'weight': [3, 3, 256], 'stride': 2, 'pad': 0,                'output': [12, 12, 256]}
    conv_2 =  {'layer': 'conv',  'input': [12, 12, 256], 'weight': [3, 3, 256], 'stride': 1, 'pad': 1, 'filter': 384, 'output': [12, 12, 384]}
    conv_3 =  {'layer': 'conv',  'input': [12, 12, 384], 'weight': [3, 3, 384], 'stride': 1, 'pad': 1, 'filter': 384, 'output': [12, 12, 384]}
    conv_4 =  {'layer': 'conv',  'input': [12, 12, 384], 'weight': [3, 3, 384], 'stride': 1, 'pad': 1, 'filter': 256, 'output': [12, 12, 256]}
    pool_2 =  {'layer': 'pool',  'input': [12, 12, 256], 'weight': [3, 3, 256], 'stride': 2, 'pad': 0,                'output': [5, 5, 256]}
    dense_0 = {'layer': 'dense', 'input': 6400,          'weight': [6400, 4096],                                      'output': 4096}
    dense_1 = {'layer': 'dense', 'input': 4096,          'weight': [4096, 4096],                                      'output': 4096}
    dense_2 = {'layer': 'dense', 'input': 4096,          'weight': [4096, 1000],                                      'output': 1000}

    layers = [conv_0, pool_0, conv_1, pool_1, conv_2, conv_3, conv_4, pool_2, dense_0, dense_1, dense_2]

    array_shape = architecture_dict['multiplier']['instance']
    array_height = array_shape[0] * array_shape[2]
    array_width = array_shape[1] * array_shape[3]

    for layer in layers:
        if layer['layer'] == 'conv':
            # total mappings
            mappings = get_prod(layer['weight']) * get_prod(layer['output']) / get_prod(array_shape)

            # weight mappings
            weight_mappings = (layer['filter'] * layer['weight']) / get_prod(array_shape)

            # utilization
            row_util = layer['filter'] / array_width
            column_util = layer['weight'][2] / array_height
            total_util = row_util * column_util
            
    performance_dict['subevent']['ai_chiplet_compute'] = OrderedDict({'count': mappings,
                                                                      'factor': {'cycle': total_util,
                                                                                 'runtime': total_util}})
    performance_dict['subevent']['ai_chiplet_weight'] = OrderedDict({'count': weight_mappings,
                                                                     'factor': {'cycle': row_util,
                                                                                 'runtime': row_util}})

    performance_dict['subevent']['accumulator_chiplet'] = OrderedDict({'count': mappings,
                                                                      'factor': {'cycle': total_util,
                                                                                 'runtime': total_util}})

    accumulator_chiplet_dict = OrderedDict()
    vector_unit_chiplet_dict = OrderedDict()
    memory_weight_reads_dict = OrderedDict()
    memory_compute_reads_dict = OrderedDict()
    memory_weight_writes_dict = OrderedDict()

    performance_dict['subevent'] = OrderedDict({'ai_chiplet_compute': ai_chiplet_compute_dict,
                                                'ai_chiplet_weight': ai_chiplet_weight_dict,
                                                'accumulator_chiplet': accumulator_chiplet_dict,
                                                'vector_unit_chiplet': vector_unit_chiplet_dict,
                                                'memory_weight_reads': memory_weight_reads_dict,
                                                'memory_compute_reads': memory_compute_reads_dict,
                                                'memory_weight_reads': memory_weight_writes_dict})