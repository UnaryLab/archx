from math import ceil, floor
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
    for _, _, row_used, col_used, cycles_this_fold in _ws_fold_infos(M, K, N, array_rows, array_cols):
        yield row_used, col_used, cycles_this_fold

def _ws_fold_infos(M: int, K: int, N: int, array_rows: int, array_cols: int):
    for k_start in range(0, K, array_rows):
        row_used = min(array_rows, K - k_start)
        for n_start in range(0, N, array_cols):
            col_used = min(array_cols, N - n_start)
            yield k_start, n_start, row_used, col_used, M + array_rows + array_cols - 2

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

def _active_fraction(module: OrderedDict) -> float:
    query = module.get('query', {})
    fraction = float(query.get('active_fraction', query.get('active_buffer_fraction', 0.5)))
    return min(1.0, max(0.0, fraction))

def _sram_elements(arch: OrderedDict, sram_name: str) -> int:
    width = int(arch[sram_name]['query']['width'])
    return max(1, _sram_bits(OrderedDict({'architecture': arch}), sram_name) // width)

def _buffer_elements(arch: OrderedDict, sram_name: str) -> tuple[int, int]:
    total = _sram_elements(arch, sram_name)
    active = max(1, min(total, floor(total * _active_fraction(arch[sram_name]))))
    prefetch = max(1, total - active)
    return active, prefetch

def _sram_bytes_per_cycle(arch: OrderedDict, sram_name: str) -> float:
    query = arch[sram_name]['query']
    width_bytes = float(query['width']) / 8
    banks = max(1.0, float(query.get('bank', 1)) / 2)
    return max(1e-12, banks * width_bytes)

def _service_cycles(bytes_count: float, bytes_per_cycle: float) -> int:
    if bytes_count <= 0:
        return 0
    return ceil(bytes_count / max(1e-12, bytes_per_cycle))

def _touch_resident(resident: OrderedDict, key, elements: int, capacity: int) -> bool:
    if elements > capacity:
        return False
    if key in resident:
        resident.move_to_end(key)
        return True
    while sum(resident.values()) + elements > capacity and len(resident) > 0:
        resident.popitem(last=False)
    if elements <= capacity:
        resident[key] = elements
    return False

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

    isram_elements = _sram_elements(arch, 'isram')
    wsram_elements = _sram_elements(arch, 'wsram')
    osram_elements = _sram_elements(arch, 'osram')
    isram_active_elements, isram_prefetch_elements = _buffer_elements(arch, 'isram')
    wsram_active_elements, wsram_prefetch_elements = _buffer_elements(arch, 'wsram')
    osram_active_elements, osram_prefetch_elements = _buffer_elements(arch, 'osram')

    input_count = 0
    weight_count = 0
    output_read_count = 0
    output_write_count = 0
    input_read_bytes = 0
    weight_read_bytes = 0
    output_read_bytes = 0
    output_write_bytes = 0
    input_transfer_window_cycles = 0
    weight_transfer_window_cycles = 0
    compute_cycles = 0
    read_stall_cycles = 0
    write_stall_cycles = 0
    mapping_efficiency = 0
    compute_utilization_num = 0
    compute_utilization_den = 0
    fold_count = 0

    bytes_per_cycle = _dram_bytes_per_cycle(arch)
    isram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'isram')
    wsram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'wsram')
    osram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'osram')
    read_prefetch_bytes_per_cycle = min(
        bytes_per_cycle,
        isram_bytes_per_cycle + wsram_bytes_per_cycle,
    )
    output_drain_bytes_per_cycle = min(bytes_per_cycle, osram_bytes_per_cycle)

    input_resident = OrderedDict()
    weight_resident = OrderedDict()
    output_resident = OrderedDict()
    read_engine_available_cycle = 0
    previous_compute_start_cycle = 0
    previous_compute_end_cycle = 0
    output_buffer_bytes = 0

    for step in range(min_step + 1, max_step + 1):
        step_M, step_K, step_N = _step_dims(M, K, N, step, step_dim)
        k_fold_count = ceil(step_K / array_rows)

        for k_start, n_start, row_used, col_used, cycles_this_fold in _ws_fold_infos(step_M, step_K, step_N, array_rows, array_cols):
            fold_compute_cycles = batch * cycles_this_fold
            input_elements = batch * step_M * row_used
            weight_elements = batch * row_used * col_used
            output_elements = batch * step_M * col_used
            k_fold_idx = k_start // array_rows

            fold_input_bytes = input_elements * input_bytes
            fold_weight_bytes = weight_elements * weight_bytes
            fold_output_bytes = output_elements * output_bytes

            input_key = (step, k_start, row_used, step_M, batch)
            weight_key = (step, k_start, n_start, row_used, col_used, batch)
            output_key = (step, n_start, col_used, step_M, batch)

            input_hit = _touch_resident(input_resident, input_key, input_elements, isram_active_elements)
            weight_hit = _touch_resident(weight_resident, weight_key, weight_elements, wsram_active_elements)
            output_hit = output_key in output_resident
            output_fits = output_elements <= osram_active_elements
            if output_fits:
                if output_hit:
                    output_resident.move_to_end(output_key)
                else:
                    while sum(output_resident.values()) + output_elements > osram_active_elements and len(output_resident) > 0:
                        output_resident.popitem(last=False)
                    output_resident[output_key] = output_elements

            prefetch_input_bytes = 0 if input_hit else fold_input_bytes
            prefetch_weight_bytes = 0 if weight_hit else fold_weight_bytes

            input_count += (0 if input_hit else input_elements) / isram_elements
            weight_count += (0 if weight_hit else weight_elements) / wsram_elements
            input_read_bytes += prefetch_input_bytes
            weight_read_bytes += prefetch_weight_bytes

            if not output_fits and k_fold_idx > 0:
                output_read_count += output_elements / osram_elements
                output_read_bytes += fold_output_bytes

            output_write_count += output_elements / osram_elements
            output_write_bytes += fold_output_bytes
            compute_cycles += fold_compute_cycles

            prefetch_bytes = prefetch_input_bytes + prefetch_weight_bytes
            input_prefetch_chunks = ceil(prefetch_input_bytes / (isram_prefetch_elements * input_bytes)) if prefetch_input_bytes > 0 else 0
            weight_prefetch_chunks = ceil(prefetch_weight_bytes / (wsram_prefetch_elements * weight_bytes)) if prefetch_weight_bytes > 0 else 0
            read_prefetch_cycles = _service_cycles(
                prefetch_bytes,
                read_prefetch_bytes_per_cycle,
            )
            read_prefetch_cycles += max(0, input_prefetch_chunks - 1)
            read_prefetch_cycles += max(0, weight_prefetch_chunks - 1)

            read_prefetch_start_cycle = 0 if fold_count == 0 else previous_compute_start_cycle
            read_prefetch_window_cycles = (
                read_prefetch_cycles if fold_count == 0
                else max(1, previous_compute_end_cycle - read_prefetch_start_cycle)
            )
            if prefetch_bytes > 0:
                input_transfer_window_cycles += read_prefetch_window_cycles * prefetch_input_bytes / prefetch_bytes
                weight_transfer_window_cycles += read_prefetch_window_cycles * prefetch_weight_bytes / prefetch_bytes
            read_engine_available_cycle = max(read_engine_available_cycle, read_prefetch_start_cycle) + read_prefetch_cycles
            compute_start_cycle = max(previous_compute_end_cycle, read_engine_available_cycle)
            read_stall_cycles += max(0, compute_start_cycle - previous_compute_end_cycle)

            sram_stream_cycles = max(
                _service_cycles(fold_input_bytes, isram_bytes_per_cycle),
                _service_cycles(fold_weight_bytes, wsram_bytes_per_cycle),
                _service_cycles(fold_output_bytes, osram_bytes_per_cycle),
            )
            sram_stream_stall_cycles = max(0, sram_stream_cycles - fold_compute_cycles)
            read_stall_cycles += sram_stream_stall_cycles
            compute_duration_cycles = fold_compute_cycles + sram_stream_stall_cycles

            output_buffer_bytes = max(
                0,
                output_buffer_bytes - compute_duration_cycles * output_drain_bytes_per_cycle,
            )
            output_buffer_bytes += fold_output_bytes
            output_buffer_capacity_bytes = osram_prefetch_elements * output_bytes
            if output_buffer_bytes > output_buffer_capacity_bytes:
                write_stall = _service_cycles(
                    output_buffer_bytes - output_buffer_capacity_bytes,
                    output_drain_bytes_per_cycle,
                )
                write_stall_cycles += write_stall
                output_buffer_bytes = max(
                    0,
                    output_buffer_bytes - write_stall * output_drain_bytes_per_cycle,
                )

            if output_fits and k_fold_idx == k_fold_count - 1 and output_key in output_resident:
                del output_resident[output_key]

            mapping_efficiency += (row_used * col_used) / (array_rows * array_cols)
            compute_utilization_num += batch * step_M * row_used * col_used
            compute_utilization_den += fold_compute_cycles * array_rows * array_cols
            fold_count += 1
            previous_compute_start_cycle = compute_start_cycle
            previous_compute_end_cycle = compute_start_cycle + compute_duration_cycles

    final_write_stall_cycles = _service_cycles(output_buffer_bytes, output_drain_bytes_per_cycle)
    write_stall_cycles += final_write_stall_cycles

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
        'input_transfer_window_cycles': input_transfer_window_cycles,
        'weight_transfer_window_cycles': weight_transfer_window_cycles,
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
    performance_dict['input_transfer_window_cycle_count'] = OrderedDict({
        'value': schedule['input_transfer_window_cycles'] / total_steps,
        'unit': 'cycle',
    })
    performance_dict['weight_transfer_window_cycle_count'] = OrderedDict({
        'value': schedule['weight_transfer_window_cycles'] / total_steps,
        'unit': 'cycle',
    })

    return performance_dict
