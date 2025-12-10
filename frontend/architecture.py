from __future__ import annotations
from copy import deepcopy
from typing import Optional, Union, TYPE_CHECKING
from yaml import safe_dump
import os
import graph_tool.all as gt
import inspect

from frontend.parameter import Parameter
from frontend.sweeping import Sweeping

if TYPE_CHECKING:
    from frontend.design import Design

class Architecture:
    def __init__(self, design: Design, name: str, yaml_path: Optional[str] = None):
        assert isinstance(name, str), "Parameter 'name' must be of type 'str'."
        assert isinstance(yaml_path, (str, type(None))), "'yaml_path' parameter must be of type 'str' or not provided"
        
        self.design = design
        self.name = name
        self.yaml_path = yaml_path if yaml_path.endswith('/') else yaml_path + '/'

        self.modules = dict()
        self.attributes = dict()

        if self.yaml_path and not os.path.exists(self.yaml_path):
            os.makedirs(os.path.normpath(self.yaml_path))

    def add_attributes(self, attributes: Optional[dict] = dict(), **kwargs):
        assert attributes or kwargs, "At least 'attributes' must be provided or values passed as kwargs."
        assert isinstance(attributes, dict), "'attributes' parameter must be of type 'dict'."
        for key, value in kwargs.items():
            assert key not in attributes, f"Attribute '{key}' already exists in attributes."
            self.attributes[key] = value

    def new_module(self, name: Union[str, list], instance: Union[list, Sweeping], tag: list, query: Optional[dict] = dict(), **kwargs):
        assert isinstance(name, (str, list)), "'name' parameter must be of type 'str' or 'list'."
        assert isinstance(tag, (str, list)), "'tag' parameter must be of type 'str' or 'list'."
        assert isinstance(query, dict), "'query' parameter must be of type 'dict'."

        # extract query parameters from kwargs and apply sweeping
        for key, value in kwargs.items():
            if isinstance(value, Sweeping):
                value = value._apply()
            query[key] = value

        # apply remaining sweeping
        if isinstance(instance, Sweeping):
            instance = instance._apply()
        for key, value in query.items():
            if isinstance(value, Sweeping):
                query[key] = value._apply()

        if isinstance(name, str):
            name = [name]
        if isinstance(tag, str):    
            tag = [tag]

        for n in name:
            assert n not in self.modules, f"Module '{n}' already exists in architecture."
            self.modules[n] = dict()

            instance_sweep = isinstance(instance[0], list)
            self.module_parameter_node(module_name=n, parameter_name='instance', parameter=instance, sweep=instance_sweep)
            self.module_parameter_node(module_name=n, parameter_name='tag', parameter=tag, sweep=False)

            for key, value in query.items():
                query_sweep = isinstance(value, list)
                self.module_parameter_node(module_name=n, parameter_name=key, parameter=value, sweep=query_sweep)

    def module_parameter_node(self, module_name: str, parameter_name: str, parameter, sweep: bool):
        new_parameter = Parameter(value=parameter, sweep=sweep)
        parameter_vertex = self.design.constraint_graph.add_node(name=module_name, node_type=parameter_name, node=new_parameter, description='architecture')
        new_parameter.set_vertex(parameter_vertex)
        self.modules[module_name][parameter_name] = new_parameter