from math import ceil, floor
from collections import OrderedDict

from chiplet4ai.common.performance.utils import _step_config, _step_dims, _fit_2d_tile, _sram_bits

def _architecture(architecture_dict: OrderedDict) -> OrderedDict:
    if 'architecture' in architecture_dict and 'pe' in architecture_dict['architecture']:
        return architecture_dict['architecture']
    if 'architecture' in architecture_dict and 'module' in architecture_dict['architecture']:
        modules = architecture_dict['architecture']['module']
    else:
        modules = architecture_dict

    return OrderedDict({
        'pe': modules['pe']['instance'],
        'isram': modules['isram'],
        'wsram': modules['wsram'],
        'osram': modules['osram'],
    })

def gemm(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    arch = _architecture(architecture_dict)

    step_dim, min_step, max_step, total_steps = _step_config(M, K, N, step_start, step_dim)

    total_steps = max_step - min_step

    # averaged values
    input_count = 0
    weight_count = 0
    compute_count = 0

    input_cycle_count = 0
    weight_cycle_count = 0
    compute_cycle_count = 0

    for step in range(min_step + 1, max_step + 1):

        M_step, K_step, N_step = _step_dims(M, K, N, step, step_dim)

        mt = arch['pe'][0]
        kt = arch['pe'][0]
        nt = arch['pe'][1]

        m_tiles = ceil(M_step / mt)
        k_tiles = ceil(K_step / kt)
        n_tiles = ceil(N_step / nt)

        m_util = M_step / (m_tiles * mt)
        k_util = K_step / (k_tiles * kt)
        n_util = N_step / (n_tiles * nt)

        input_util = m_util * k_util
        weight_util = k_util * n_util
        compute_util = m_util * k_util * n_util

        input_cycle_util = 1/m_util
        weight_cycle_util = 1/n_util
        compute_cycle_util = 1/(m_util * n_util)

        input_tiles = batch * m_tiles * k_tiles * n_tiles * input_util
        weight_tiles = batch * k_tiles * n_tiles * weight_util
        compute_tiles = batch * m_tiles * k_tiles * n_tiles * compute_util

        input_count += input_tiles
        weight_count += weight_tiles
        compute_count += compute_tiles

        input_cycle_count += input_tiles * input_cycle_util
        weight_cycle_count += weight_tiles * weight_cycle_util
        compute_cycle_count += compute_tiles * compute_cycle_util

    performance_dict['subevent'] = OrderedDict({
        'array_input_mapping': {
            'count': input_count / total_steps,
            'factor': {
                'cycle_count': input_cycle_count / input_count if input_count > 0 else 0,
                'runtime': input_cycle_count / input_count if input_count > 0 else 0
                }
        },
        'array_weight_mapping': {
            'count': weight_count / total_steps,
            'factor': {
                'cycle_count': weight_cycle_count / weight_count if weight_count > 0 else 0,
                'runtime': weight_cycle_count / weight_count if weight_count > 0 else 0
            }
        },
        'array_compute_mapping': {
            'count': compute_count / total_steps,
            'factor': {
                'cycle_count': compute_cycle_count / compute_count if compute_count > 0 else 0,
                'runtime': compute_cycle_count / compute_count if compute_count > 0 else 0
            }
        }
    })

    return performance_dict

def sram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    arch = _architecture(architecture_dict)

    step_dim, min_step, max_step, total_steps = _step_config(M, K, N, step_start, step_dim)

    mt = arch['pe'][0]
    kt = arch['pe'][0]
    nt = arch['pe'][1]

    input_width = arch['isram']['query']['width']
    weight_width = arch['wsram']['query']['width']
    output_width = arch['osram']['query']['width']

    isram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'isram') // input_width)
    wsram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'wsram') // weight_width)
    osram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'osram') // output_width)

    input_count = 0
    weight_count = 0
    output_read_count = 0
    output_write_count = 0

    input_cycle_count = 0
    weight_cycle_count = 0
    output_read_cycle_count = 0
    output_write_cycle_count = 0

    for step in range(min_step + 1, max_step + 1):
        step_M, step_K, step_N = _step_dims(M, K, N, step, step_dim)

        input_m, input_k = _fit_2d_tile(step_M, step_K, mt, kt, isram_elements)
        weight_k, weight_n = _fit_2d_tile(step_K, step_N, kt, nt, wsram_elements)
        output_m, output_n = _fit_2d_tile(step_M, step_N, mt, nt, osram_elements)

        k_tile = min(input_k, weight_k)
        
        input_k = k_tile
        weight_k = k_tile
        input_m_tiles = ceil(step_M / input_m)
        output_m_tiles = ceil(step_M / output_m)
        k_tiles = ceil(step_K / k_tile)
        weight_n_tiles = ceil(step_N / weight_n)
        output_n_tiles = ceil(step_N / output_n)

        input_resident = step_M * step_K <= 2 * isram_elements
        weight_resident = step_K * step_N <= 2 * wsram_elements

        input_reloads = 1 if input_resident else weight_n_tiles
        weight_reloads = 1 if weight_resident else 1

        input_events = batch * input_m_tiles * k_tiles * input_reloads
        weight_events = batch * k_tiles * weight_n_tiles * weight_reloads
        output_tiles = output_m_tiles * output_n_tiles

        input_util = (step_M / (input_m_tiles * input_m)) * (step_K / (k_tiles * input_k))
        weight_util = (step_K / (k_tiles * weight_k)) * (step_N / (weight_n_tiles * weight_n))
        output_util = (step_M / (output_m_tiles * output_m)) * (step_N / (output_n_tiles * output_n))

        input_events *= input_util
        weight_events *= weight_util
        output_events = batch * output_tiles * output_util

        output_tile_elements = output_m * output_n
        output_live_tiles = output_tiles
        output_resident_tiles = max(1, floor((2 * osram_elements) / output_tile_elements))
        output_spill_tiles = max(0, output_live_tiles - output_resident_tiles)

        output_final_read_events = output_events
        output_partial_read_events = batch * output_spill_tiles * max(0, k_tiles - 1) * output_util
        output_partial_write_events = batch * output_spill_tiles * max(0, k_tiles - 1) * output_util

        input_count += input_events
        weight_count += weight_events
        output_read_count += output_final_read_events + output_partial_read_events
        output_write_count += output_partial_write_events

        input_cycle_count += input_events
        weight_cycle_count += weight_events
        output_read_cycle_count += output_final_read_events + output_partial_read_events
        output_write_cycle_count += output_partial_write_events

    performance_dict['subevent'] = OrderedDict({
        'sram_input_write_mapping': OrderedDict({
            'count': input_count / total_steps,
            'factor': {
                'cycle_count': input_cycle_count / input_count if input_count > 0 else 0,
                'runtime': input_cycle_count / input_count if input_count > 0 else 0
            }
        }),
        'sram_weight_write_mapping': OrderedDict({
            'count': weight_count / total_steps,
            'factor': {
                'cycle_count': weight_cycle_count / weight_count if weight_count > 0 else 0,
                'runtime': weight_cycle_count / weight_count if weight_count > 0 else 0
            }
        }),
        'sram_output_read_mapping': OrderedDict({
            'count': output_read_count / total_steps,
            'factor': {
                'cycle_count': output_read_cycle_count / output_read_count if output_read_count > 0 else 0,
                'runtime': output_read_cycle_count / output_read_count if output_read_count > 0 else 0
            }
        }),
        'sram_output_write_mapping': OrderedDict({
            'count': output_write_count / total_steps,
            'factor': {
                'cycle_count': output_write_cycle_count / output_write_count if output_write_count > 0 else 0,
                'runtime': output_write_cycle_count / output_write_count if output_write_count > 0 else 0
            }
        })
    })

    return performance_dict

def dram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    arch = _architecture(architecture_dict)

    step_dim, min_step, max_step, total_steps = _step_config(M, K, N, step_start, step_dim)

    mt = arch['pe'][0]
    kt = arch['pe'][0]
    nt = arch['pe'][1]

    input_width = arch['isram']['query']['width']
    weight_width = arch['wsram']['query']['width']
    output_width = arch['osram']['query']['width']

    input_bytes = input_width / 8
    weight_bytes = weight_width / 8
    output_bytes = output_width / 8

    isram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'isram') // input_width)
    wsram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'wsram') // weight_width)
    osram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'osram') // output_width)

    input_read_count = 0
    weight_read_count = 0
    output_read_count = 0
    output_write_count = 0

    for step in range(min_step + 1, max_step + 1):
        step_M, step_K, step_N = _step_dims(M, K, N, step, step_dim)

        input_m, input_k = _fit_2d_tile(step_M, step_K, mt, kt, isram_elements)
        weight_k, weight_n = _fit_2d_tile(step_K, step_N, kt, nt, wsram_elements)
        output_m, output_n = _fit_2d_tile(step_M, step_N, mt, nt, osram_elements)

        k_tile = min(input_k, weight_k)
        input_k = k_tile
        weight_k = k_tile

        input_m_tiles = ceil(step_M / input_m)
        output_m_tiles = ceil(step_M / output_m)
        k_tiles = ceil(step_K / k_tile)
        weight_n_tiles = ceil(step_N / weight_n)
        output_n_tiles = ceil(step_N / output_n)

        input_resident = step_M * step_K <= 2 * isram_elements
        weight_resident = step_K * step_N <= 2 * wsram_elements

        input_reloads = 1 if input_resident else weight_n_tiles
        weight_reloads = 1 if weight_resident else 1

        input_util = (step_M / (input_m_tiles * input_m)) * (step_K / (k_tiles * input_k))
        weight_util = (step_K / (k_tiles * weight_k)) * (step_N / (weight_n_tiles * weight_n))
        output_util = (step_M / (output_m_tiles * output_m)) * (step_N / (output_n_tiles * output_n))

        input_tile_elements = input_m * input_k
        weight_tile_elements = weight_k * weight_n
        output_tile_elements = output_m * output_n

        input_read_count += batch * input_m_tiles * k_tiles * input_reloads * input_tile_elements * input_util * input_bytes
        weight_read_count += batch * k_tiles * weight_n_tiles * weight_reloads * weight_tile_elements * weight_util * weight_bytes

        output_tiles = output_m_tiles * output_n_tiles
        output_resident_tiles = max(1, floor((2 * osram_elements) / output_tile_elements))
        output_spill_tiles = max(0, output_tiles - output_resident_tiles)

        output_final_bytes = batch * output_tiles * output_tile_elements * output_util * output_bytes
        output_partial_bytes = batch * output_spill_tiles * max(0, k_tiles - 1) * output_tile_elements * output_util * output_bytes

        output_read_count += output_partial_bytes
        output_write_count += output_final_bytes + output_partial_bytes

    performance_dict['subevent'] = OrderedDict({
        'dram_input_read': OrderedDict({
            'count': input_read_count / total_steps,
            'factor': {'cycle_count': 1, 'runtime': 1}
        }),
        'dram_weight_read': OrderedDict({
            'count': weight_read_count / total_steps,
            'factor': {'cycle_count': 1, 'runtime': 1}
        }),
        'dram_output_read': OrderedDict({
            'count': output_read_count / total_steps,
            'factor': {'cycle_count': 1, 'runtime': 1}
        }),
        'dram_output_write': OrderedDict({
            'count': output_write_count / total_steps,
            'factor': {'cycle_count': 1, 'runtime': 1}
        })
    })

    return performance_dict
