from collections import OrderedDict
from math import ceil
from archx.utils import get_prod


def layer_epochs(layer, tile_r, tile_c, tile_m):
    R, C, M, K_h, K_w, Z, H_in, W_in, stride = layer
    # as in the paper, tile the padded ifmap over the stride (same padding, pad = K // 2), not the ofmap
    R_tile = (H_in + 2 * (K_h // 2)) // stride
    C_tile = (W_in + 2 * (K_w // 2)) // stride
    return ceil(R_tile / tile_r) * ceil(C_tile / tile_c) * ceil(M / tile_m) * K_h * K_w * Z


def cnn(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:

    performance_dict = OrderedDict()
    bitwidth = workload_dict['configuration']['bitwidth']
    layers = workload_dict['configuration']['layers']

    # ofmap R x C x M maps onto the mac array x * y * z, output stationary;
    # each PE accumulates K_h * K_w * Z terms, one per epoch of 2^(bitwidth - 1) cycles
    tile_r, tile_c, tile_m = architecture_dict['mac']['instance']

    conv_ops = 0
    epochs = 0
    for layer in layers:
        R, C, M, K_h, K_w, Z = layer[:6]
        conv_ops += R * C * M * K_h * K_w * Z
        epochs += layer_epochs(layer, tile_r, tile_c, tile_m)

    mac_dim = get_prod(architecture_dict['mac']['instance'])
    nrdo_dim = get_prod(architecture_dict['nrdo']['instance'])

    # paper Sec. 3.1.1: t_max = 2^(b - 1) time slots; the sign bit rides on a separate pulse stream
    value_cycles = 2**(bitwidth - 1)

    cycles = epochs * value_cycles

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
