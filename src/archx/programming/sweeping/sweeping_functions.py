from typing import Callable

def condition_sweep(value, funct: Callable, condition: Callable):
    """Sweeping from a lower to upper range using a specified funct."""
    sweep_values = [value]
    new_value = funct(value)
    condition_met = condition(value)
    while condition_met:
        sweep_values.append(new_value)
        new_value = funct(new_value)
        condition_met = condition(new_value)
    return sweep_values

def iteration_sweep(value, iter: int, funct: Callable):
    """Sweeping through an explicit number of iterations using a specified funct."""
    is_single = not isinstance(value, list)
    values = [value] if is_single else list(value)
    sweep_values = []

    for _ in range(iter):
        sweep_values.append(values.copy())
        values = [funct(v) for v in values]

    if is_single:
        return [v[0] for v in sweep_values]
    return sweep_values

def list_sweep(value, funct: Callable):
    """Sweeping through a list of values using a specified funct."""
    sweep_values = []
    for v in value:
        sweep_values.append(funct(v))
    return sweep_values