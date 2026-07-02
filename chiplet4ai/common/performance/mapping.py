from math import ceil
from collections import OrderedDict

from chiplet4ai.common.performance.utils import _step_config, _step_dims, _sram_bits

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
        'dram': modules['dram'],
    })

def _ws_folds(M: int, K: int, N: int, array_rows: int, array_cols: int):
    for k_start in range(0, K, array_rows):
        row_used = min(array_rows, K - k_start)
        for n_start in range(0, N, array_cols):
            col_used = min(array_cols, N - n_start)
            yield row_used, col_used, M + array_rows + array_cols - 2

def _frequency_mhz(arch: OrderedDict) -> float:
    query = arch['dram'].get('query', {})
    if 'frequency' in query:
        return float(query['frequency'])
    return 1000.0

def _dram_bytes_per_cycle(arch: OrderedDict) -> float:
    bandwidth_gbs = float(arch['dram']['query']['bandwidth'])
    frequency_hz = _frequency_mhz(arch) * 1e6
    return max(1e-12, bandwidth_gbs * 2**30 / frequency_hz)

def _cycle_runtime_ms(arch: OrderedDict) -> float:
    return 1 / (1000 * _frequency_mhz(arch))

def _ws_schedule(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    arch = _architecture(architecture_dict)
    step_dim, min_step, max_step, total_steps = _step_config(M, K, N, step_start, step_dim)

    array_rows = arch['pe'][0]
    array_cols = arch['pe'][1]

    input_width = arch['isram']['query']['width']
    weight_width = arch['wsram']['query']['width']
    output_width = arch['osram']['query']['width']

    input_bytes = input_width / 8
    weight_bytes = weight_width / 8
    output_bytes = output_width / 8

    isram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'isram') // input_width)
    wsram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'wsram') // weight_width)
    osram_elements = max(1, _sram_bits(OrderedDict({'architecture': arch}), 'osram') // output_width)

    input_count = 0
    weight_count = 0
    output_read_count = 0
    output_write_count = 0
    input_read_bytes = 0
    weight_read_bytes = 0
    output_read_bytes = 0
    output_write_bytes = 0
    compute_cycles = 0
    read_stall_cycles = 0
    write_stall_cycles = 0
    mapping_efficiency = 0
    compute_utilization_num = 0
    compute_utilization_den = 0
    fold_count = 0
    previous_compute_cycles = 0

    bytes_per_cycle = _dram_bytes_per_cycle(arch)

    for step in range(min_step + 1, max_step + 1):
        step_M, step_K, step_N = _step_dims(M, K, N, step, step_dim)

        for row_used, col_used, cycles_this_fold in _ws_folds(step_M, step_K, step_N, array_rows, array_cols):
            fold_compute_cycles = batch * cycles_this_fold
            input_elements = batch * step_M * row_used
            weight_elements = batch * row_used * col_used
            output_elements = batch * step_M * col_used

            fold_input_bytes = input_elements * input_bytes
            fold_weight_bytes = weight_elements * weight_bytes
            fold_output_bytes = output_elements * output_bytes

            input_count += input_elements / isram_elements
            weight_count += weight_elements / wsram_elements
            output_write_count += output_elements / osram_elements
            input_read_bytes += fold_input_bytes
            weight_read_bytes += fold_weight_bytes
            output_write_bytes += fold_output_bytes
            compute_cycles += fold_compute_cycles

            read_prefetch_cycles = ceil((fold_input_bytes + fold_weight_bytes) / bytes_per_cycle)
            output_drain_cycles = ceil(fold_output_bytes / bytes_per_cycle)
            read_stall_cycles += max(0, read_prefetch_cycles - previous_compute_cycles)
            write_stall_cycles += max(0, output_drain_cycles - fold_compute_cycles)

            mapping_efficiency += (row_used * col_used) / (array_rows * array_cols)
            compute_utilization_num += batch * step_M * row_used * col_used
            compute_utilization_den += fold_compute_cycles * array_rows * array_cols
            fold_count += 1
            previous_compute_cycles = fold_compute_cycles

    return OrderedDict({
        'total_steps': total_steps,
        'input_count': input_count,
        'weight_count': weight_count,
        'output_read_count': output_read_count,
        'output_write_count': output_write_count,
        'input_read_bytes': input_read_bytes,
        'weight_read_bytes': weight_read_bytes,
        'output_read_bytes': output_read_bytes,
        'output_write_bytes': output_write_bytes,
        'compute_cycles': compute_cycles,
        'read_stall_cycles': read_stall_cycles,
        'write_stall_cycles': write_stall_cycles,
        'stall_cycles': read_stall_cycles + write_stall_cycles,
        'mapping_efficiency': mapping_efficiency / fold_count if fold_count > 0 else 0,
        'compute_utilization': compute_utilization_num / compute_utilization_den if compute_utilization_den > 0 else 0,
        'cycle_runtime_ms': _cycle_runtime_ms(arch),
    })

def gemm(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    arch = _architecture(architecture_dict)
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim)

    total_steps = schedule['total_steps']
    input_count = 0
    weight_count = 0
    compute_count = 0

    step_dim, min_step, max_step, _ = _step_config(M, K, N, step_start, step_dim)
    array_rows = arch['pe'][0]
    array_cols = arch['pe'][1]

    for step in range(min_step + 1, max_step + 1):
        M_step, K_step, N_step = _step_dims(M, K, N, step, step_dim)

        m_tiles = ceil(M_step / array_rows)
        k_tiles = ceil(K_step / array_rows)
        n_tiles = ceil(N_step / array_cols)

        m_util = M_step / (m_tiles * array_rows)
        k_util = K_step / (k_tiles * array_rows)
        n_util = N_step / (n_tiles * array_cols)

        input_util = m_util * k_util
        weight_util = k_util * n_util
        compute_util = m_util * k_util * n_util

        input_tiles = batch * m_tiles * k_tiles * n_tiles * input_util
        weight_tiles = batch * k_tiles * n_tiles * weight_util
        compute_tiles = batch * m_tiles * k_tiles * n_tiles * compute_util

        input_count += input_tiles
        weight_count += weight_tiles
        compute_count += compute_tiles

    performance_dict['subevent'] = OrderedDict({
        'array_input_mapping': {
            'count': input_count / total_steps,
            'factor': {
                'cycle_count': schedule['compute_cycles'] / input_count if input_count > 0 else 0,
                'runtime': schedule['compute_cycles'] / input_count if input_count > 0 else 0
                }
        },
        'array_weight_mapping': {
            'count': weight_count / total_steps,
            'factor': {
                'cycle_count': schedule['compute_cycles'] / weight_count if weight_count > 0 else 0,
                'runtime': schedule['compute_cycles'] / weight_count if weight_count > 0 else 0
            }
        },
        'array_compute_mapping': {
            'count': compute_count / total_steps,
            'factor': {
                'cycle_count': schedule['compute_cycles'] / compute_count if compute_count > 0 else 0,
                'runtime': schedule['compute_cycles'] / compute_count if compute_count > 0 else 0
            }
        }
    })
    performance_dict['mapping_efficiency'] = OrderedDict({
        'value': schedule['mapping_efficiency'],
        'unit': 'ratio',
    })
    performance_dict['compute_utilization'] = OrderedDict({
        'value': schedule['compute_utilization'],
        'unit': 'ratio',
    })
    performance_dict['compute_cycle_count'] = OrderedDict({
        'value': schedule['compute_cycles'] / total_steps,
        'unit': 'cycle',
    })
    performance_dict['stall_cycle_count'] = OrderedDict({
        'value': schedule['stall_cycles'] / total_steps,
        'unit': 'cycle',
    })
    performance_dict['read_stall_cycle_count'] = OrderedDict({
        'value': schedule['read_stall_cycles'] / total_steps,
        'unit': 'cycle',
    })
    performance_dict['write_stall_cycle_count'] = OrderedDict({
        'value': schedule['write_stall_cycles'] / total_steps,
        'unit': 'cycle',
    })

    return performance_dict

def sram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim)
    total_steps = schedule['total_steps']

    performance_dict['subevent'] = OrderedDict({
        'sram_input_write_mapping': OrderedDict({
            'count': schedule['input_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0
            }
        }),
        'sram_weight_write_mapping': OrderedDict({
            'count': schedule['weight_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0
            }
        }),
        'sram_output_read_mapping': OrderedDict({
            'count': schedule['output_read_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0
            }
        }),
        'sram_output_write_mapping': OrderedDict({
            'count': schedule['output_write_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0
            }
        })
    })

    return performance_dict

def dram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    arch = _architecture(architecture_dict)
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim)

    total_steps = schedule['total_steps']
    input_read_count = schedule['input_read_bytes']
    weight_read_count = schedule['weight_read_bytes']
    output_read_count = schedule['output_read_bytes']
    output_write_count = schedule['output_write_bytes']
    read_count = input_read_count + weight_read_count
    write_count = output_write_count
    raw_runtime_per_byte = 1000 / (float(arch['dram']['query']['bandwidth']) * 2**30)
    cycle_runtime_ms = schedule['cycle_runtime_ms']

    read_cycle_factor = schedule['read_stall_cycles'] / read_count if read_count > 0 else 0
    write_cycle_factor = schedule['write_stall_cycles'] / write_count if write_count > 0 else 0
    read_runtime_factor = (
        schedule['read_stall_cycles'] * cycle_runtime_ms / (read_count * raw_runtime_per_byte)
        if read_count > 0 else 0
    )
    write_runtime_factor = (
        schedule['write_stall_cycles'] * cycle_runtime_ms / (write_count * raw_runtime_per_byte)
        if write_count > 0 else 0
    )

    performance_dict['subevent'] = OrderedDict({
        'dram_input_read': OrderedDict({
            'count': input_read_count / total_steps,
            'factor': {
                'cycle_count': read_cycle_factor,
                'runtime': read_runtime_factor,
            }
        }),
        'dram_weight_read': OrderedDict({
            'count': weight_read_count / total_steps,
            'factor': {
                'cycle_count': read_cycle_factor,
                'runtime': read_runtime_factor,
            }
        }),
        'dram_output_read': OrderedDict({
            'count': output_read_count / total_steps,
            'factor': {'cycle_count': 0, 'runtime': 0}
        }),
        'dram_output_write': OrderedDict({
            'count': output_write_count / total_steps,
            'factor': {
                'cycle_count': write_cycle_factor,
                'runtime': write_runtime_factor,
            }
        })
    })
    performance_dict['memory_stall_cycle_count'] = OrderedDict({
        'value': schedule['stall_cycles'] / total_steps,
        'unit': 'cycle',
    })
    performance_dict['read_stall_cycle_count'] = OrderedDict({
        'value': schedule['read_stall_cycles'] / total_steps,
        'unit': 'cycle',
    })
    performance_dict['write_stall_cycle_count'] = OrderedDict({
        'value': schedule['write_stall_cycles'] / total_steps,
        'unit': 'cycle',
    })

    return performance_dict
