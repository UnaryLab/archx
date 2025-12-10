from collections import OrderedDict
from archx.utils import get_prod

def vgg16(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:

    performance_dict = OrderedDict()
    bitwidth = workload_dict['vgg16']['configuration']['bitwidth']
    conv3_64 = workload_dict['vgg16']['configuration']['conv3-64']
    conv3_128 = workload_dict['vgg16']['configuration']['conv3-128']
    conv3_256 = workload_dict['vgg16']['configuration']['conv3-256']
    conv3_512 = workload_dict['vgg16']['configuration']['conv3-512']
    # maxpool = workload_dict['vgg16']['configuration']['maxpool']
    # fc1 = workload_dict['vgg16']['configuration']['fc1']
    # fc2 = workload_dict['vgg16']['configuration']['fc2']
    # fc3 = workload_dict['vgg16']['configuration']['fc3']

    conv_layers = [conv3_64, conv3_128, conv3_256, conv3_512]

    conv3_of = [workload_dict['vgg16']['configuration']['conv3-64_of'],
                 workload_dict['vgg16']['configuration']['conv3-128_of'],
                 workload_dict['vgg16']['configuration']['conv3-256_of'],
                 workload_dict['vgg16']['configuration']['conv3-512_of']]

    conv_ops = 0

    for output_layer,  weight_layer in zip(conv3_of, conv_layers):
        weight_flattened = weight_layer[0] * weight_layer[1] * weight_layer[2]
        for outpuer in output_layer:
            output_flattened = outpuer[0] * outpuer[1] * outpuer[2]
            layer_ops = weight_flattened * output_flattened
            conv_ops += layer_ops

    # mult_dim = get_prod(architecture_dict['mult']['instance'])
    # acc_dim = get_prod(architecture_dict['acc']['instance'])
    mac_dim = get_prod(architecture_dict['mac']['instance'])
    nrdo_dim = get_prod(architecture_dict['nrdo']['instance'])

    value_cycles = 2**bitwidth

    cycles = (conv_ops * value_cycles) / mac_dim

    frequency = architecture_dict['mac']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': (cycles / (frequency * 1e6)), 'unit': 'ms'})

    # mult_dict = OrderedDict({'count': conv_ops})
    # acc_dict = OrderedDict({'count': conv_ops})
    mac_dict = OrderedDict({'count': conv_ops})
    mac_splitter_dict = OrderedDict({'count': conv_ops})
    weight_splitter_dict = OrderedDict({'count': conv_ops})
    ptl_splitter_dict = OrderedDict({'count': conv_ops})
    nrdo_dict = OrderedDict({'count': conv_ops / (mac_dim / nrdo_dim)})
    input_splitter_dict = OrderedDict({'count': conv_ops})

    # fc_ops = (fc1[0] * fc1[1]) + (fc2[0] * fc2[1]) + (fc3[0] * fc3[1])

    # convolution_dict = OrderedDict({'count': conv_ops})
    # fc_dict = OrderedDict({'count': fc_ops})

    # performance_dict['subevent'] = OrderedDict({'convolution': convolution_dict,
    #                                             'fully_connected': fc_dict})
    performance_dict['subevent'] = OrderedDict({'mac': mac_dict,
                                                'mac_splitter': mac_splitter_dict,
                                                'weight_splitter': weight_splitter_dict,
                                                'input_splitter': input_splitter_dict,
                                                'ptl': ptl_splitter_dict,
                                                'nrdo': nrdo_dict})
    return performance_dict

def convolution(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()
    
    # mult_dim = get_prod(architecture_dict['mult']['instance'])
    # acc_dim = get_prod(architecture_dict['acc']['instance'])
    pe_dim = get_prod(architecture_dict['pe']['instance'])
    # mac_splitter_dim = get_prod(architecture_dict['mac_splitter']['instance'])
    weight_splitter_dim = get_prod(architecture_dict['weight_splitter']['instance'])

    cycles = 1/pe_dim

    frequency = architecture_dict['mult']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycles / 1000 / frequency, 'unit': 'ms'})

    # mult_dict = OrderedDict({'count': 1})
    # acc_dict = OrderedDict({'count': 1})
    pe_dict = OrderedDict({'count': 1})
    # mac_splitter_dict = OrderedDict({'count': 1})
    weight_splitter_dict = OrderedDict({'count': weight_splitter_dim})

    return performance_dict