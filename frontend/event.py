from __future__ import annotations
from copy import deepcopy
from typing import Optional, TYPE_CHECKING
from yaml import safe_dump
import os

if TYPE_CHECKING:
    from frontend.design import Design

class Node:
    def __init__(self, subevent: list, performance: Optional[str] = None):
        self.subevent = subevent
        self.performance = performance

class Event:
    def __init__(self, design: Design, name: str, yaml_path: Optional[str] = None, performance_path: Optional[str] = None):
        assert isinstance(name, str), "Parameter 'name' must be of type 'str'."
        assert isinstance(yaml_path, (str, type(None))), "'yaml_path' parameter must be of type 'str' or not provided."
        assert isinstance(performance_path, (str, type(None))), "'performance_path' parameter must be of type 'str' or not provided."

        self.design = design
        self.name = name
        self.yaml_path = yaml_path if yaml_path.endswith('/') else yaml_path + '/'
        self.performance_path = performance_path

        self.nodes = dict()

        if self.yaml_path and not os.path.exists(self.yaml_path):
            os.makedirs(os.path.normpath(self.yaml_path))
        self.yaml_path = self.yaml_path + 'event.yaml' if self.yaml_path else None

    def new_event(self, name: str, subevent: list, performance: Optional[str] = None):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert name not in self.nodes, f"Event '{name}' already exists in event. Maybe you meant to call 'update_event()'."
        assert isinstance(subevent, list), "'subevent' parameter must be of type 'list'."
        assert performance is None or isinstance(performance, str), "'performance' parameter must be of type 'str' or not provided."

        self.nodes[name] = Node(subevent, performance)

    def add_event(self, name: str, node: Node):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert name not in self.nodes, f"Event '{name}' already exists in event. Maybe you meant to call 'update_event()'."
        assert isinstance(node, Node), "'node' parameter must be of type 'Node'."

        self.nodes[name] = node

    def to_yaml(self):
        event_dict = dict()
        event_dict['event'] = dict()

        for name, node in self.nodes.items():
            event_dict['event'][name] = dict()
            event_dict['event'][name]['subevent'] = deepcopy(node.subevent)
            if node.performance:
                event_dict['event'][name]['performance'] = node.performance
        
        with open(self.yaml_path, 'w') as file:
            safe_dump(event_dict, file)

# def remove_event(self, name: str):
#     assert isinstance(name, str), "'name' parameter must be of type 'str'."
#     assert name in self.nodes, f"Event '{name}' not found in event."

#     del self.nodes[name]

# def update_event(self, name: str, subevent: Optional[list] = None, performance: Optional[str] = None):
#     assert isinstance(name, str), "'name' parameter must be of type 'str'."
#     assert name in self.nodes, f"Event '{name}' not found in event. Maybe you meant to call 'add_event()'."
#     assert subevent or performance, "At least one of 'subevent' or 'performance' parameters must be provided."

#     if subevent:
#         assert isinstance(subevent, list), "'subevent' parameter must be of type 'list'."
#         self.nodes[name].subevent = subevent
    
#     if performance:
#         assert isinstance(performance, str), "'performance' parameter must be of type 'str'."
#         self.nodes[name].performance = performance

# def replace_event(self, name: str, node: Node):
#     assert isinstance(name, str), "'name' parameter must be of type 'str'."
#     assert name in self.nodes, f"Event '{name}' not found in event. Maybe you meant to call 'add_event()'."
#     assert isinstance(node, Node), "'node' parameter must be of type 'Node'."

#     self.nodes[name] = node