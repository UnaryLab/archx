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
def _fit_2d_tile(rows: int, cols: int, min_rows: int, min_cols: int, capacity_elements: int) -> tuple[int, int]:
    rows_t = max(1, min(rows, min_rows))
    cols_t = max(1, min(cols, max(min_cols, capacity_elements // rows_t)))

    if rows_t * cols_t > capacity_elements:
        cols_t = max(1, capacity_elements // rows_t)

    if rows_t * cols_t > capacity_elements:
        rows_t = max(1, capacity_elements // cols_t)

    while rows_t < rows and (rows_t + 1) * cols_t <= capacity_elements:
        rows_t += 1

    while cols_t < cols and rows_t * (cols_t + 1) <= capacity_elements:
        cols_t += 1

    return rows_t, cols_t

def _sram_bits(architecture_dict: OrderedDict, sram_name: str) -> int:
    query = architecture_dict['architecture'][sram_name]['query']
    return int(query['width'] * query['bank'] * query['depth'])
# endregion
