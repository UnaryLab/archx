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

def _bandwidth_gib_per_second(arch: OrderedDict, bytes_count: float, cycles: float) -> float:
    if bytes_count <= 0 or cycles <= 0:
        return 0
    frequency_hz = _frequency_mhz(arch) * 1e6
    return bytes_count / cycles * frequency_hz / 2**30

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

def _ws_phase_cycles(batch: int, M: int, array_rows: int, array_cols: int) -> tuple[int, int, int]:
    weight_fill_cycles = batch * max(0, array_rows - 1)
    steady_cycles = batch * M
    output_tail_cycles = batch * max(0, array_cols - 1)
    return weight_fill_cycles, steady_cycles, output_tail_cycles

def _phase_stall(phase_cycles: int, *service_cycles: int) -> int:
    if len(service_cycles) == 0:
        return 0
    return max(0, max(service_cycles) - phase_cycles)

def _cycle_event(architecture_dict: OrderedDict) -> OrderedDict:
    arch = _architecture(architecture_dict)
    return OrderedDict({
        'cycle_count': OrderedDict({'value': 1, 'unit': 'cycle'}),
        'runtime': OrderedDict({'value': _cycle_runtime_ms(arch), 'unit': 'ms'}),
        'subevent': OrderedDict({'pe': OrderedDict({'count': 0})}),
    })

def _cycle_reference(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _cycle_event(architecture_dict)

def cycle_reference(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _cycle_event(architecture_dict)

def _stall_event(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return OrderedDict({'subevent': OrderedDict({
        'cycle_reference': OrderedDict({'count': 1, 'aggregation': 'sequential'})
    })})

def dram_input_read_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def dram_weight_read_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def dram_output_write_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def sram_weight_fill_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def sram_steady_state_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

def sram_output_tail_stall(architecture_dict: OrderedDict, workload_dict: OrderedDict = None) -> OrderedDict:
    return _stall_event(architecture_dict, workload_dict)

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

def _touch_chunked_resident(
    resident: OrderedDict,
    key,
    elements: int,
    active_capacity: int,
    prefetch_capacity: int,
) -> tuple[int, int, int, int]:
    if elements <= 0:
        return 0, 0, 0, 0

    chunk_elements = max(1, min(elements, active_capacity, prefetch_capacity))
    active_window_elements = min(elements, active_capacity)
    missed_elements = 0
    startup_missed_elements = 0
    missed_chunks = 0
    startup_missed_chunks = 0
    offset = 0
    chunk_idx = 0

    while offset < elements:
        chunk = min(chunk_elements, elements - offset)
        chunk_key = (key, chunk_idx)
        hit = _touch_resident(resident, chunk_key, chunk, active_capacity)

        if not hit:
            missed_elements += chunk
            missed_chunks += 1
            active_overlap = max(0, min(offset + chunk, active_window_elements) - offset)
            if active_overlap > 0:
                startup_missed_elements += active_overlap
                startup_missed_chunks += 1

        offset += chunk
        chunk_idx += 1

    return missed_elements, startup_missed_elements, missed_chunks, startup_missed_chunks

def _launch_output_drain(
    output_free_bytes: float,
    current_cycle: int,
    output_total_capacity_bytes: float,
    output_drain_capacity_bytes: float,
    output_drain_bytes_per_cycle: float,
) -> tuple[float, int, int]:
    output_used_bytes = output_total_capacity_bytes - output_free_bytes
    drain_bytes = min(output_used_bytes, output_drain_capacity_bytes)
    if drain_bytes <= 0:
        return output_free_bytes, current_cycle, 0

    output_free_bytes += drain_bytes
    drain_cycles = _service_cycles(drain_bytes, output_drain_bytes_per_cycle)
    return output_free_bytes, current_cycle + drain_cycles, drain_cycles

def _service_output_write(
    output_free_bytes: float,
    output_drain_end_cycle: int,
    current_cycle: int,
    output_bytes: float,
    output_active_capacity_bytes: float,
    output_total_capacity_bytes: float,
    output_drain_capacity_bytes: float,
    output_drain_bytes_per_cycle: float,
) -> tuple[float, int, int, int]:
    remaining_bytes = output_bytes
    write_stall_cycles = 0
    output_transfer_window_cycles = 0

    while remaining_bytes > 0:
        if current_cycle >= output_drain_end_cycle and output_free_bytes < output_active_capacity_bytes:
            output_free_bytes, output_drain_end_cycle, drain_cycles = _launch_output_drain(
                output_free_bytes,
                current_cycle,
                output_total_capacity_bytes,
                output_drain_capacity_bytes,
                output_drain_bytes_per_cycle,
            )
            output_transfer_window_cycles += drain_cycles

        if output_free_bytes <= 0:
            stall_cycles = max(0, output_drain_end_cycle - current_cycle)
            write_stall_cycles += stall_cycles
            current_cycle += stall_cycles
            continue

        written_bytes = min(remaining_bytes, output_free_bytes)
        output_free_bytes -= written_bytes
        remaining_bytes -= written_bytes

    if current_cycle >= output_drain_end_cycle and output_free_bytes < output_active_capacity_bytes:
        output_free_bytes, output_drain_end_cycle, drain_cycles = _launch_output_drain(
            output_free_bytes,
            current_cycle,
            output_total_capacity_bytes,
            output_drain_capacity_bytes,
            output_drain_bytes_per_cycle,
        )
        output_transfer_window_cycles += drain_cycles

    return output_free_bytes, output_drain_end_cycle, write_stall_cycles, output_transfer_window_cycles

def _drain_remaining_output(
    output_free_bytes: float,
    output_drain_end_cycle: int,
    current_cycle: int,
    output_total_capacity_bytes: float,
    output_drain_capacity_bytes: float,
    output_drain_bytes_per_cycle: float,
) -> tuple[int, int]:
    final_drain_cycles = 0
    output_transfer_window_cycles = 0

    while output_free_bytes < output_total_capacity_bytes or current_cycle < output_drain_end_cycle:
        if current_cycle < output_drain_end_cycle:
            wait_cycles = output_drain_end_cycle - current_cycle
            final_drain_cycles += wait_cycles
            current_cycle = output_drain_end_cycle

        if output_free_bytes < output_total_capacity_bytes:
            output_free_bytes, output_drain_end_cycle, drain_cycles = _launch_output_drain(
                output_free_bytes,
                current_cycle,
                output_total_capacity_bytes,
                output_drain_capacity_bytes,
                output_drain_bytes_per_cycle,
            )
            output_transfer_window_cycles += drain_cycles

            if output_drain_end_cycle == current_cycle:
                break

    return final_drain_cycles, output_transfer_window_cycles

def _range_sum(first: int, last: int) -> int:
    if last < first:
        return 0
    return (first + last) * (last - first + 1) // 2

def _ceil_sum(first: int, last: int, divisor: int) -> int:
    if last < first:
        return 0
    return _ceil_sum_to(last, divisor) - _ceil_sum_to(first - 1, divisor)

def _ceil_sum_to(n: int, divisor: int) -> int:
    if n <= 0:
        return 0
    q, r = divmod(n, divisor)
    return divisor * q * (q + 1) // 2 + (q + 1) * r

def _ceil_minus_one_sum(first: int, last: int, divisor: int) -> int:
    count = max(0, last - first + 1)
    return _ceil_sum(first, last, divisor) - count

def _fit_fraction(capacity_elements: float, working_set_elements: float) -> float:
    if working_set_elements <= 0:
        return 1.0
    return max(0.0, min(1.0, capacity_elements / working_set_elements))

def _resident_refetch_elements(unique_elements: float, streamed_elements: float, fit_fraction: float) -> float:
    refetch_elements = max(0.0, streamed_elements - unique_elements)
    return unique_elements + refetch_elements * (1.0 - fit_fraction)

def _chunk_count(elements: float, capacity_elements: float) -> int:
    if elements <= 0:
        return 1
    return max(1, ceil(elements / max(1.0, capacity_elements)))

def _window_from_chunks(compute_cycles: float, bytes_count: float, chunks: int) -> float:
    if bytes_count <= 0:
        return 0
    return max(1.0, compute_cycles / max(1, chunks))

def _read_active_span_cycles(
    compute_cycles: float,
    demand_elements: float,
    dram_elements: float,
    startup_buffer_elements: float,
) -> float:
    if dram_elements <= 0:
        return 0
    if compute_cycles <= 0 or demand_elements <= 0:
        return 1

    demand_rate = demand_elements / compute_cycles
    demand_span = min(compute_cycles, dram_elements / max(1e-12, demand_rate))
    startup_span = min(dram_elements, max(1.0, startup_buffer_elements)) / max(1e-12, demand_rate)
    return max(1.0, demand_span + startup_span)

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

    isram_active_elements, isram_prefetch_elements = _buffer_elements(arch, 'isram')
    wsram_active_elements, wsram_prefetch_elements = _buffer_elements(arch, 'wsram')
    osram_active_elements, osram_prefetch_elements = _buffer_elements(arch, 'osram')

    bytes_per_cycle = _dram_bytes_per_cycle(arch)
    isram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'isram')
    wsram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'wsram')
    osram_bytes_per_cycle = _sram_bytes_per_cycle(arch, 'osram')
    input_prefetch_bytes_per_cycle = min(bytes_per_cycle, isram_bytes_per_cycle)
    weight_prefetch_bytes_per_cycle = min(bytes_per_cycle, wsram_bytes_per_cycle)
    output_drain_bytes_per_cycle = min(bytes_per_cycle, osram_bytes_per_cycle)

    first_step = min_step + 1
    last_step = max_step
    if step_dim == 'm':
        sum_M = _range_sum(first_step, last_step)
        sum_K = K * total_steps
        sum_N = N * total_steps
        sum_k_folds = ceil(K / array_rows) * total_steps
        sum_n_folds = ceil(N / array_cols) * total_steps
        fold_count = ceil(K / array_rows) * ceil(N / array_cols) * total_steps
        compute_cycles = batch * ceil(K / array_rows) * ceil(N / array_cols) * (sum_M + total_steps * (array_rows + array_cols - 2))
        input_unique_elements = batch * K * sum_M
        input_sram_elements = input_unique_elements * ceil(N / array_cols)
        weight_elements = batch * K * N * total_steps
        output_write_elements = batch * N * sum_M * ceil(K / array_rows)
        output_read_elements = batch * N * sum_M * max(0, ceil(K / array_rows) - 1)
        max_M, max_K, max_N = last_step, K, N
    elif step_dim == 'k':
        sum_M = M * total_steps
        sum_K = _range_sum(first_step, last_step)
        sum_N = N * total_steps
        sum_k_folds = _ceil_sum(first_step, last_step, array_rows)
        sum_n_folds = ceil(N / array_cols) * total_steps
        fold_count = ceil(N / array_cols) * sum_k_folds
        compute_cycles = batch * ceil(N / array_cols) * sum_k_folds * (M + array_rows + array_cols - 2)
        input_unique_elements = batch * M * sum_K
        input_sram_elements = input_unique_elements * ceil(N / array_cols)
        weight_elements = batch * N * sum_K
        output_write_elements = batch * M * N * sum_k_folds
        output_read_elements = batch * M * N * _ceil_minus_one_sum(first_step, last_step, array_rows)
        max_M, max_K, max_N = M, last_step, N
    elif step_dim == 'n':
        sum_M = M * total_steps
        sum_K = K * total_steps
        sum_N = _range_sum(first_step, last_step)
        sum_k_folds = ceil(K / array_rows) * total_steps
        sum_n_folds = _ceil_sum(first_step, last_step, array_cols)
        fold_count = ceil(K / array_rows) * sum_n_folds
        compute_cycles = batch * ceil(K / array_rows) * sum_n_folds * (M + array_rows + array_cols - 2)
        input_unique_elements = batch * M * K * total_steps
        input_sram_elements = batch * M * K * sum_n_folds
        weight_elements = batch * K * sum_N
        output_write_elements = batch * M * sum_N * ceil(K / array_rows)
        output_read_elements = batch * M * sum_N * max(0, ceil(K / array_rows) - 1)
        max_M, max_K, max_N = M, K, last_step
    else:
        k_folds = ceil(K / array_rows)
        n_folds = ceil(N / array_cols)
        sum_M = M
        sum_K = K
        sum_N = N
        sum_k_folds = k_folds
        sum_n_folds = n_folds
        fold_count = k_folds * n_folds
        compute_cycles = batch * fold_count * (M + array_rows + array_cols - 2)
        input_unique_elements = batch * M * K
        input_sram_elements = input_unique_elements * n_folds
        weight_elements = batch * K * N
        output_write_elements = batch * M * N * k_folds
        output_read_elements = batch * M * N * max(0, k_folds - 1)
        max_M, max_K, max_N = M, K, N

    max_input_tile = batch * max_M * min(array_rows, max_K)
    max_weight_tile = batch * min(array_rows, max_K) * min(array_cols, max_N)
    max_output_tile = batch * max_M * min(array_cols, max_N)
    input_fit = _fit_fraction(isram_active_elements, max_input_tile)
    weight_fit = _fit_fraction(wsram_active_elements, max_weight_tile)
    output_fit = _fit_fraction(osram_active_elements, max_output_tile)
    input_chunks = _chunk_count(max_input_tile, isram_active_elements)
    weight_chunks = _chunk_count(max_weight_tile, wsram_active_elements)
    output_chunks = _chunk_count(max_output_tile, osram_active_elements)

    input_read_elements = _resident_refetch_elements(input_unique_elements, input_sram_elements, input_fit)
    weight_read_elements = weight_elements
    output_spills = max_output_tile > osram_active_elements
    if not output_spills:
        output_read_elements = 0

    input_read_bytes = input_read_elements * input_bytes
    weight_read_bytes = weight_read_elements * weight_bytes
    output_read_bytes = output_read_elements * output_bytes
    output_write_bytes = output_write_elements * output_bytes
    input_sram_bytes = input_sram_elements * input_bytes
    weight_sram_bytes = weight_read_elements * weight_bytes
    output_sram_bytes = output_read_bytes + output_write_bytes

    input_count = input_read_elements / isram_active_elements
    weight_count = weight_read_elements / wsram_active_elements
    output_read_count = output_read_elements / osram_active_elements
    output_write_count = output_write_elements / osram_active_elements

    input_service_cycles = _service_cycles(input_read_bytes, input_prefetch_bytes_per_cycle)
    weight_service_cycles = _service_cycles(weight_read_bytes, weight_prefetch_bytes_per_cycle)
    output_service_cycles = _service_cycles(output_write_bytes, output_drain_bytes_per_cycle)
    input_read_stall_cycles = max(0, input_service_cycles - compute_cycles)
    weight_read_stall_cycles = max(0, weight_service_cycles - compute_cycles)
    write_stall_cycles = max(0, output_service_cycles - compute_cycles)

    weight_fill_window = batch * max(0, array_rows - 1) * fold_count
    steady_window = (
        batch * ceil(K / array_rows) * ceil(N / array_cols) * sum_M
        if step_dim == 'm'
        else batch * M * fold_count
    )
    output_tail_window = batch * max(0, array_cols - 1) * fold_count
    weight_fill_stall_cycles = max(0, _service_cycles(weight_sram_bytes, wsram_bytes_per_cycle) - weight_fill_window)
    steady_state_stall_cycles = max(
        0,
        max(
            _service_cycles(input_sram_bytes, isram_bytes_per_cycle),
            _service_cycles(output_read_bytes, osram_bytes_per_cycle),
        ) - steady_window,
    )
    output_tail_stall_cycles = max(0, _service_cycles(output_write_bytes, osram_bytes_per_cycle) - (steady_window + output_tail_window))
    read_stall_cycles = max(input_read_stall_cycles, weight_read_stall_cycles) + weight_fill_stall_cycles + steady_state_stall_cycles + output_tail_stall_cycles

    input_transfer_window_cycles = _window_from_chunks(compute_cycles, input_read_bytes, input_chunks)
    weight_transfer_window_cycles = _read_active_span_cycles(
        compute_cycles,
        weight_elements,
        weight_read_elements,
        wsram_active_elements,
    )
    output_transfer_window_cycles = compute_cycles if output_write_bytes > 0 else 0
    input_sram_window_cycles = _window_from_chunks(compute_cycles, input_sram_bytes, input_chunks)
    weight_sram_window_cycles = _window_from_chunks(compute_cycles, weight_sram_bytes, weight_chunks)
    output_sram_window_cycles = _window_from_chunks(compute_cycles, output_sram_bytes, output_chunks)

    mapping_efficiency = (sum_K / (sum_k_folds * array_rows)) * (sum_N / (sum_n_folds * array_cols)) if sum_k_folds > 0 and sum_n_folds > 0 else 0
    compute_utilization = (batch * sum_M * sum_K * sum_N / total_steps) / (compute_cycles * array_rows * array_cols) if compute_cycles > 0 else 0

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
        'input_sram_bytes': input_sram_bytes,
        'weight_sram_bytes': weight_sram_bytes,
        'output_sram_bytes': output_sram_bytes,
        'input_transfer_window_cycles': input_transfer_window_cycles,
        'weight_transfer_window_cycles': weight_transfer_window_cycles,
        'output_transfer_window_cycles': output_transfer_window_cycles,
        'input_sram_window_cycles': input_sram_window_cycles,
        'weight_sram_window_cycles': weight_sram_window_cycles,
        'output_sram_window_cycles': output_sram_window_cycles,
        'compute_cycles': compute_cycles,
        'input_read_stall_cycles': input_read_stall_cycles,
        'weight_read_stall_cycles': weight_read_stall_cycles,
        'weight_fill_stall_cycles': weight_fill_stall_cycles,
        'steady_state_stall_cycles': steady_state_stall_cycles,
        'output_tail_stall_cycles': output_tail_stall_cycles,
        'read_stall_cycles': read_stall_cycles,
        'write_stall_cycles': write_stall_cycles,
        'stall_cycles': read_stall_cycles + write_stall_cycles,
        'input_sram_bandwidth': _bandwidth_gib_per_second(arch, input_sram_bytes, input_sram_window_cycles),
        'weight_sram_bandwidth': _bandwidth_gib_per_second(arch, weight_sram_bytes, weight_sram_window_cycles),
        'output_sram_bandwidth': _bandwidth_gib_per_second(arch, output_sram_bytes, output_sram_window_cycles),
        'input_dram_bandwidth': _bandwidth_gib_per_second(arch, input_read_bytes, input_transfer_window_cycles),
        'weight_dram_bandwidth': _bandwidth_gib_per_second(arch, weight_read_bytes, weight_transfer_window_cycles),
        'output_dram_bandwidth': _bandwidth_gib_per_second(arch, output_write_bytes, output_transfer_window_cycles),
        'mapping_efficiency': mapping_efficiency,
        'compute_utilization': compute_utilization,
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
                'runtime': 0,
                'bandwidth': schedule['input_sram_bandwidth'],
            }
        }),
        'sram_weight_write_mapping': OrderedDict({
            'count': schedule['weight_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['weight_sram_bandwidth'],
            }
        }),
        'sram_output_read_mapping': OrderedDict({
            'count': schedule['output_read_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['output_sram_bandwidth'],
            }
        }),
        'sram_output_write_mapping': OrderedDict({
            'count': schedule['output_write_count'] / total_steps,
            'factor': {
                'cycle_count': 0,
                'runtime': 0,
                'bandwidth': schedule['output_sram_bandwidth'],
            }
        }),
        'sram_weight_fill_stall': OrderedDict({
            'count': schedule['weight_fill_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        'sram_steady_state_stall': OrderedDict({
            'count': schedule['steady_state_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        'sram_output_tail_stall': OrderedDict({
            'count': schedule['output_tail_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
    })

    return performance_dict

def dram(architecture_dict: OrderedDict, batch: int, M: int, K: int, N: int, step_start: int, step_dim: str = None) -> OrderedDict:
    performance_dict = OrderedDict()
    schedule = _ws_schedule(architecture_dict, batch, M, K, N, step_start, step_dim)

    total_steps = schedule['total_steps']
    input_read_count = schedule['input_read_bytes']
    weight_read_count = schedule['weight_read_bytes']
    output_read_count = schedule['output_read_bytes']
    output_write_count = schedule['output_write_bytes']

    performance_dict['subevent'] = OrderedDict({
        'dram_input_read': OrderedDict({
            'count': input_read_count / total_steps,
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['input_dram_bandwidth']}
        }),
        'dram_weight_read': OrderedDict({
            'count': weight_read_count / total_steps,
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['weight_dram_bandwidth']}
        }),
        'dram_output_read': OrderedDict({
            'count': output_read_count / total_steps,
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['output_dram_bandwidth']}
        }),
        'dram_output_write': OrderedDict({
            'count': output_write_count / total_steps,
            'aggregation': 'sequential',
            'factor': {'cycle_count': 0, 'runtime': 0, 'bandwidth': schedule['output_dram_bandwidth']}
        }),
        'dram_input_read_stall': OrderedDict({
            'count': schedule['input_read_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        'dram_weight_read_stall': OrderedDict({
            'count': schedule['weight_read_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
        'dram_output_write_stall': OrderedDict({
            'count': schedule['write_stall_cycles'] / total_steps,
            'aggregation': 'sequential',
        }),
    })

    return performance_dict
