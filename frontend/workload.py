from __future__ import annotations
from yaml import safe_dump
from typing import Optional, TYPE_CHECKING
from copy import deepcopy
import os
import graph_tool.all as gt

from frontend.parameter import Parameter
from frontend.sweeping import Sweeping

if TYPE_CHECKING:
    from frontend.design import Design

class WorkloadConfiguration:
    def __init__(self, workload: Workload):
        self.parameters = dict()
        self.workload = workload

class Workload:
    def __init__(self, design: Design, name: str, yaml_path: Optional[str] = None):
        assert isinstance(name, str), "Parameter 'name' must be of type 'str'."
        assert isinstance(yaml_path, (str, type(None))), "'yaml_path' parameter must be of type 'str'. or not provided."
        
        self.design = design
        self.name = name
        self.yaml_path = yaml_path if yaml_path.endswith('/') else yaml_path + '/'

        self.configurations = dict()

        if self.yaml_path and not os.path.exists(self.yaml_path):
            os.makedirs(os.path.normpath(self.yaml_path))

    def new_configuration(self, name: str) -> WorkloadConfiguration:
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert name not in self.configurations, f"Configuration '{name}' already exists in workload. Maybe you meant to call 'update_configuration()'."

        self.configurations[name] = WorkloadConfiguration(workload=self)
        return self.configurations[name]

    def new_parameter(self, configuration: str, parameter_name: str, parameter_value, sweep: bool = None):
        assert isinstance(configuration, str), "'configuration' parameter must be of type 'str'."
        assert configuration in self.configurations, f"Configuration '{configuration}' not found in workload. Maybe you meant to call 'add_configuration()'."
        assert isinstance(parameter_name, str), "'parameter_name' parameter must be of type 'str'."
        assert parameter_name not in self.configurations[configuration].parameters, f"Parameter '{parameter_name}' already exists in configuration. Maybe you meant to call 'update_parameter()'."
        assert isinstance(sweep, (bool, type(None))), "'sweep' parameter must be of type 'bool'."
        assert (isinstance(parameter_value, list) and isinstance(sweep, bool)) or (not isinstance(parameter_value, list)) , "If 'parameter_value' is a list, 'sweep' must be defined as either True or False."

        if isinstance(parameter_value, Sweeping) or sweep is True:
            sweep = True
        else:
            sweep = False

        if isinstance(parameter_value, Sweeping):
            parameter_value = parameter_value._apply()

        self.configuration_parameter_node(configuration_name=configuration, parameter_name=parameter_name, parameter=parameter_value, sweep=sweep)

    def get_configuration(self, name: str) -> WorkloadConfiguration:
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert name in self.configurations, f"Configuration '{name}' not found in workload."

        return self.configurations[name]
    
    def configuration_parameter_node(self, configuration_name: str, parameter_name: str, parameter, sweep: bool):
        new_parameter = Parameter(value=parameter, sweep=sweep)
        parameter_vertex = self.design.constraint_graph.add_node(name=configuration_name, node_type=parameter_name, node=new_parameter, description='workload')
        new_parameter.set_vertex(parameter_vertex)
        self.configurations[configuration_name].parameters[parameter_name] = new_parameter