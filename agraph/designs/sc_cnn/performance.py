from collections import OrderedDict
from archx.utils import get_prod

def vgg16(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:

    performance_dict = OrderedDict()
    bitwidth = workload_dict['configuration']['bitwidth']
    conv3_64 = workload_dict['configuration']['conv3-64']
    conv3_128 = workload_dict['configuration']['conv3-128']
    conv3_256 = workload_dict['configuration']['conv3-256']
    conv3_512 = workload_dict['configuration']['conv3-512']

    conv_layers = [conv3_64, conv3_128, conv3_256, conv3_512]

    conv3_of = [workload_dict['configuration']['conv3-64_of'],
                 workload_dict['configuration']['conv3-128_of'],
                 workload_dict['configuration']['conv3-256_of'],
                 workload_dict['configuration']['conv3-512_of']]

    conv_ops = 0

    for output_layer,  weight_layer in zip(conv3_of, conv_layers):
        weight_flattened = weight_layer[0] * weight_layer[1] * weight_layer[2]
        for outpuer in output_layer:
            output_flattened = outpuer[0] * outpuer[1] * outpuer[2]
            layer_ops = weight_flattened * output_flattened
            conv_ops += layer_ops

    mac_dim = get_prod(architecture_dict['mac']['instance'])
    nrdo_dim = get_prod(architecture_dict['nrdo']['instance'])

    value_cycles = 2**bitwidth

    cycles = (conv_ops * value_cycles) / mac_dim

    frequency = architecture_dict['mac']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': (cycles / (frequency * 1e6)), 'unit': 'ms'})

    mac_dict = OrderedDict({'count': conv_ops})
    mac_splitter_dict = OrderedDict({'count': conv_ops})
    weight_splitter_dict = OrderedDict({'count': conv_ops})
    ptl_splitter_dict = OrderedDict({'count': conv_ops})
    nrdo_dict = OrderedDict({'count': conv_ops / (mac_dim / nrdo_dim)})
    input_splitter_dict = OrderedDict({'count': conv_ops})

    performance_dict['subevent'] = OrderedDict({'mac': mac_dict,
                                                'mac_splitter': mac_splitter_dict,
                                                'weight_splitter': weight_splitter_dict,
                                                'input_splitter': input_splitter_dict,
                                                'ptl': ptl_splitter_dict,
                                                'nrdo': nrdo_dict})
    return performance_dict