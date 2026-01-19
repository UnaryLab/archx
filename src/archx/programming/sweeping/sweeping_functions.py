from typing import Callable, Optional, Union
from frontend.sweeping import Sweeping

class SingleSweep(Sweeping):
    """Single value sweeping using a specified funct."""
    def __init__(self, value, funct: Callable, condition: Optional[Callable] = None):
        super().__init__(funct=funct)
        self.value = value

    def _apply(self):
        return [self.funct(self.value)]

class ConditionSweep(Sweeping):
    """Sweeping from a lower to upper range using a specified funct."""
    def __init__(self, value, funct: Callable, condition: Callable):
        super().__init__(funct=funct)
        self.value = value
        self.condition = condition

    # def _apply(self):
    #     sweep_values = []
    #     condition = True
    #     value = self.value if isinstance(self.value, list) else [self.value]
    #     while condition:
    #         values = []
    #         for i in range(len(value)):
    #             values.append(value[i])
    #             value[i] = self.funct(value[i])
    #             if self.condition(value[i]) == False:
    #                 condition = False
    #                 break
    #         sweep_values.append(values)
    #     return sweep_values

    def _apply(self):
        sweep_values = [self.value]
        new_value = self.funct(self.value)
        condition_met = self.condition(self.value)
        while condition_met:
            sweep_values.append(new_value)
            new_value = self.funct(new_value)
            condition_met = self.condition(new_value)
            
        return sweep_values
        
class IterationSweep(Sweeping):
    """Sweeping through an explicit number of iterations using a specified funct."""
    def __init__(self, value, iter, funct: Callable):
        super().__init__(funct=funct)
        self.value = value
        self.iter = iter

    def _apply(self):
        is_single = not isinstance(self.value, list)
        values = [self.value] if is_single else list(self.value)
        sweep_values = []

        for _ in range(self.iter):
            sweep_values.append(values.copy())
            values = [self.funct(v) for v in values]

        if is_single:
            return [v[0] for v in sweep_values]
        return sweep_values

class CombinationalSweep(Sweeping):
    """List of sweeping functions, including ConditionSweep and NumSweep"""
    def __init__(self, sweeps: list[Sweeping]):
        super().__init__(funct=None)
        self.sweeps = sweeps

    def _apply(self):
        sweep_results = [sweep._apply() for sweep in self.sweeps]

        
        lengths = [len(r) for r in sweep_results]
        assert len(set(lengths)) == 1, "All sweeps must have the same length."

        normalized = []
        for sweep in sweep_results:
            if not isinstance(sweep[0], list):
                normalized.append([[value] for value in sweep])
            else:
                normalized.append(sweep)

        combined = [
            [value for group in step for value in group]
            for step in zip(*normalized)
        ]
        return combined

# class ListSweep(Sweeping):
#     def __init__(self, sweeping ):
#         super().__init__(funct=None)
#         self.sweeping = sweeping
#         self.default = isinstance(sweeping, Sweeping)

#     def _apply(self, values):
#         sweeping_values = []
#         if self.default:
#             for v in values:
#                 sweeping_values.extend(self.sweeping._apply(v))
#         else:
#             for sweep, v in zip(self.sweeping, values):
#                 sweeping_values.extend(sweep._apply(v))
#         return sweeping_values



# class ListSweep(Sweeping):
#     def __init__(self, funct: Callable, condition: Optional[Callable] = None):
#         super().__init__(funct=funct)
#         self.condition = condition

#     def _apply(self, input_values: list):
#         output_values = []
#         for value in input_values:
#             condition_met = self.condition(value) if self.condition else True
#             if condition_met:
#                 output_values.append(self.funct(value))
#         return output_values