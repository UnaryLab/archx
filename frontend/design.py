import graph_tool.all as gt
from typing import Optional, Union
from itertools import product
import os
import yaml

from frontend.architecture import Architecture
from archx.programming.object.event import Event
from frontend.metric import Metric
from frontend.workload import Workload
from frontend.constraint import ConstraintGraph

class Design:
    def __init__(self, name: str, yaml_path: str):
        self.name = name
        self.yaml_path = yaml_path
        self.constraint_graph = ConstraintGraph(design=self)
        self.architecture = Architecture(design=self, name='architecture', yaml_path=yaml_path + '/architecture/')
        self.event = Event(design=self, name='event', yaml_path=yaml_path + '/event/', performance_path=yaml_path + '/performance/')
        self.metric = Metric(design=self, name='metric', yaml_path=yaml_path + '/metric/')
        self.workload = Workload(design=self, name='workload', yaml_path=yaml_path + '/workload/')
        self.run_path = yaml_path + '/runs.txt'

        if not os.path.exists(self.yaml_path):
            os.makedirs(os.path.normpath(self.yaml_path))

        if not os.path.exists(self.yaml_path + '/runs/'):
            os.makedirs(os.path.normpath(self.yaml_path + '/runs/'))
            
        self.architecture_configurations = []

        if self.yaml_path and not os.path.exists(self.yaml_path):
            os.makedirs(os.path.normpath(self.yaml_path))

    # def new_architecture_description(self, name: str, yaml_path: Optional[str] = None):
    #     yaml_path = self.yaml_path if yaml_path is None else yaml_path
    #     architecture_description = Architecture(design=self, name=name, yaml_path=yaml_path)
    #     self.architecture = architecture_description
    #     return architecture_description
    
    # def new_event_description(self, name: str, yaml_path: Optional[str] = None, performance_path: Optional[str] = None):
    #     yaml_path = self.yaml_path if yaml_path is None else yaml_path
    #     event_description = Event(design=self, name=name, yaml_path=yaml_path, performance_path=performance_path)
    #     self.event = event_description
    #     return event_description
    
    # def new_metric_description(self, name: str, yaml_path: Optional[str] = None):
    #     yaml_path = self.yaml_path if yaml_path is None else yaml_path
    #     metric_description = Metric(design=self, name=name, yaml_path=yaml_path)
    #     self.metric = metric_description
    #     return metric_description
    
    # def new_workload_description(self, name: str, yaml_path: Optional[str] = None):
    #     yaml_path = self.yaml_path if yaml_path is None else yaml_path
    #     workload_description = Workload(design=self, name=name, yaml_path=yaml_path)
    #     self.workload = workload_description
    #     return workload_description

    def get_constraint_graph(self):
        return self.constraint_graph

    def sweep(self):
        architecture_configurations = self.architecture_sweep()
        workload_configurations = self.workload_sweep()

        for i, arch_config in enumerate(architecture_configurations):
            print(f"Architecture Configuration {i}")
            if not os.path.exists(self.architecture.yaml_path + f'architecture/architecture_config_{i}/'):
                os.makedirs(self.architecture.yaml_path + f'architecture/architecture_config_{i}/')
            with open(self.architecture.yaml_path + f'architecture/architecture_config_{i}/architecture.yaml', 'w') as f:
                yaml.dump(arch_config, f)

        for i, work_config in enumerate(workload_configurations):
            print(f"Workload Configuration {i}")
            if not os.path.exists(self.workload.yaml_path + f'workload/workload_config_{i}/'):
                os.makedirs(self.workload.yaml_path + f'workload/workload_config_{i}/')
            with open(self.workload.yaml_path + f'workload/workload_config_{i}/workload.yaml', 'w') as f:
                yaml.dump(work_config, f)

    def architecture_sweep(self):
        modules = self.architecture.modules

        architecture_configurations = []
        for name, module in modules.items():
            instance = module.instance
            query = module.query
            tag = module.tag

            instances = [{"instance": x} if isinstance(x[0], list) else {"instance": [x]} for x in instance]
            queries = [
                dict(zip(query.keys(), v))
                for v in product(*(val if isinstance(val, list) else [val] for val in query.values()))
            ]
            architecture_configurations.append([{name: {**i, **j, 'tag': tag}} for i, j in product(instances, queries)])

        architecture_yamls = [
            {
                "architecture": {
                    "attribute": self.architecture.attributes,
                    "module": {k: v for config in combo for k, v in config.items()}
                }
            }
            for combo in product(*architecture_configurations)
        ]

        return architecture_yamls
        
    def workload_sweep(self):
        configurations = self.workload.configurations
        configuration_configs = []
        for config_name, configuration in configurations.items():
            parameters = configuration.parameters
            configuration_configs.append([
                {config_name: {'configuration':dict(zip(parameters.keys(), combo))}}
                for combo in product(*(param.value if param.sweep else [param.value] for param in parameters.values()))
            ])
        
        workload_yaml = [
            {
                "workload": {
                    k: v for config in combo for k, v in config.items()  
                }
            } for combo in product(*configuration_configs)
        ]
        
        return workload_yaml

    def to_yaml(self):
        for architecture in self.architecture.values():
            architecture.to_yaml()
        for event in self.event.values():
            event.to_yaml()
        for metric in self.metric.values():
            metric.to_yaml()
        for workload in self.workload.values():
            workload.to_yaml()