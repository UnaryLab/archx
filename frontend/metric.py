from __future__ import annotations
from copy import deepcopy
from yaml import safe_dump
from typing import Optional, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from frontend.design import Design

class Metric():
    def __init__(self, design: Design, name: str, yaml_path: Optional[str] = None):
        assert isinstance(name, str), "Parameter 'name' must be of type 'str'."
        assert isinstance(yaml_path, (str, type(None))), "'yaml_path' parameter must be of type 'str'."
        
        self.design = design
        self.name = name
        self.yaml_path = yaml_path if yaml_path.endswith('/') else yaml_path + '/'

        self.metrics = dict()
        if self.yaml_path and not os.path.exists(self.yaml_path):
            os.makedirs(os.path.normpath(self.yaml_path))
        self.yaml_path = self.yaml_path + 'metric.yaml' if self.yaml_path else None

    def new_metric(self, name: str, unit: str, aggregation: str):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert isinstance(unit, str), "'unit' parameter must be of type 'str'."
        assert isinstance(aggregation, str), "'aggregation' parameter must be of type 'str'."

        self.metrics[name] = {
            "unit": unit,
            "aggregation": aggregation
        }

    def to_yaml(self):
        with open(self.yaml_path, 'w') as file:
            safe_dump({'metric': deepcopy(self.metrics)}, file)

# def remove_metric(self, name: str):
#     assert isinstance(name, str), "'name' parameter must be of type 'str'."
#     assert name in self.metrics, f"Metric '{name}' not found in metric."

#     del self.metrics[name]

# def update_metric(self, name: str, unit: Optional[str] = None, aggregation: Optional[str] = None):
#     assert isinstance(name, str), "'name' parameter must be of type 'str'."
#     assert name in self.metrics, f"Metric '{name}' not found in metric. Maybe you meant to call 'add_metric()'."
#     assert unit or aggregation, "At least one of 'unit' or 'aggregation' parameters must be provided."

#     if unit:
#         assert isinstance(unit, str), "'unit' parameter must be of type 'str'."
#         self.metrics[name]['unit'] = unit
    
#     if aggregation:
#         assert isinstance(aggregation, str), "'aggregation' parameter must be of type 'str'."
#         self.metrics[name]['aggregation'] = aggregation