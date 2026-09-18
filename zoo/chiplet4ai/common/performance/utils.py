from collections import OrderedDict

def _step_config(M: int, K: int, N: int, step_start: int, step_dim: str):
    if step_dim is not None:
        step_dim = step_dim.lower()
        assert step_dim in ['m', 'k', 'n'], f"step_dim must be one of ['m', 'k', 'n'], but got {step_dim}"
        min_step = step_start
        max_step = M if step_dim == 'm' else K if step_dim == 'k' else N
        assert min_step < max_step, f"step_start must be less than {step_dim}, but got step_start={step_start}, {step_dim}={max_step}"
    else:
        min_step = 0
        max_step = 1

    total_steps = max_step - min_step

    return step_dim, min_step, max_step, total_steps

def _step_dims(M: int, K: int, N: int, step: int, step_dim: str):
    return (
        step if step_dim == 'm' else M,
        step if step_dim == 'k' else K,
        step if step_dim == 'n' else N
    )

# region: sram traffic
def _sram_bits(architecture_dict: OrderedDict, sram_name: str) -> int:
    query = architecture_dict['architecture'][sram_name]['query']
    return int(query['width'] * query['bank'] * query['depth'])
# endregion

# region: nominal sram size
BASE_SRAM_BITS = 10 * 2**23  # 10 MiB, the reference capacity of every SRAM

def nominal_sram_bits(array_m: int, array_n: int, width: int) -> int:
    # The reference capacity, doubled until one whole array_m x array_n weight tile fits
    # the active half of wsram (2 * array_n banks, see mapping._buffer_elements), so
    # mapping._tiling never shrinks Nt below the array. Every array up to 512x512 already
    # fits at 10 MiB and keeps it; this only grows the SRAM of arrays whose PE count
    # outgrows it (2048x2048: 20 MiB, 4096x4096: 80 MiB).
    bits = BASE_SRAM_BITS
    while bits < 2 * array_m * array_n * width:
        bits *= 2
    return bits
# endregion
