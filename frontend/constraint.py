import graph_tool.all as gt
from typing import Union, TypedDict, Optional
from collections import namedtuple, defaultdict
from copy import deepcopy
from tqdm import tqdm
import math
import yaml
import os

from frontend.parameter import Parameter, ArchitectureParameter, WorkloadParameter, ArchitectureWorkloadParameter
from itertools import product, combinations, zip_longest, combinations

class ConstraintGraph:
    def __init__(self, design):
        self.graph = self.new_graph()
        self.subgraphs = []

        self.design = design

        self.name_to_vp = dict()

        self.architecture_idx = 0

    def add_node(self, name: str, node_type: str, node: Union[Parameter], description: str):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert isinstance(node_type, str), "'node_type' parameter must be a string"
        assert name not in self.graph.vp, f"Node '{name}' already exists in constraint graph. Maybe you meant to call 'update_node()'."
        assert isinstance(node, (Parameter)), "'node' parameter must be of type 'Parameter'."

        v = self.graph.add_vertex()
        self.graph.vp.name[v] = name
        self.graph.vp.node[v] = node
        self.graph.vp.node_type[v] = node_type
        self.graph.vp.desc[v] = description

        if name in self.name_to_vp:
            assert node_type not in self.name_to_vp[name], f"Node '{name}' with type '{node_type}' already exists in constraint graph mapping."
            self.name_to_vp[name][node_type] = v
        else:
            self.name_to_vp[name] = {node_type: v}

        return v

    def add_constraint(self, a_name: str = None, a_parameters: list = None, constraint: Optional[str] = 'direct', b_name: Optional[str] = None, b_parameters: Optional[list] = None, condition: Optional[callable] = None, **kwargs):
        assert isinstance(a_name, (str, type(None))), "'a_name' parameter must be of type 'str'."
        assert (b_name is not None) == (b_parameters is not None), "Parameters 'b_name' and 'b_parameters' must be all defined or all None."
        assert isinstance(b_name, (str, type(None))), "'b_name' parameter must be of type 'str'."
        assert isinstance(a_parameters, (list, type(None))), "'a_parameters' parameter must be of type 'list'."
        assert isinstance(b_parameters, (list, type(None))), "'b_parameters' parameter must be of type 'list'."
        assert a_parameters is None or len(a_parameters) > 0, "'a_parameters' parameter must contain at least one parameter."
        assert b_parameters is None or len(b_parameters) > 0, "'b_parameters' parameter must contain at least one parameter."
        assert condition is None or callable(condition), "'condition' parameter must be a callable function or None."
        assert constraint in ['direct', 'anti', 'conditional'], "'constraint' parameter must be either 'direct', 'anti', or 'conditional'."

        vertices = []

        if kwargs is not None:
            for key, value in kwargs.items():
                assert key not in ['a_name', 'a_parameters', 'b_name', 'b_parameters', 'constraint', 'condition'], f"Keyword argument '{key}' is reserved and cannot be used in 'add_constraint()'."
                assert key in self.name_to_vp, f"Node '{key}' not found in constraint graph."

                for param in value:
                    assert param in self.name_to_vp[key], f"Parameter '{param}' not found in node '{key}' in constraint graph."
                    vertices.append(self.name_to_vp[key][param])

                for a in vertices:
                    for b in vertices:
                        if a != b:
                            e = self.graph.edge(a, b)
                            if e is None:
                                e = self.graph.add_edge(a, b)
                            self.graph.ep.constraint[e] = constraint
                            self.graph.ep.condition[e] = condition
        else:
            if b_name is None:
                assert a_name in self.name_to_vp, f"Node '{a_name}' not found in constraint graph."

                for parameter in a_parameters:
                    assert parameter in self.name_to_vp[a_name], f"Parameter '{parameter}' not found in node '{a_name}' in constraint graph."
                    vertices.append(self.name_to_vp[a_name][parameter])

                for i in range(len(vertices) - 1):
                    e = self.graph.edge(vertices[i], vertices[i + 1])
                    if e is None:
                        e = self.graph.add_edge(vertices[i], vertices[i + 1])
                    self.graph.ep.constraint[e] = constraint
                    self.graph.ep.condition[e] = condition
                
            else:
                assert a_name in self.name_to_vp, f"Node '{a_name}' not found in constraint graph."
                assert b_name in self.name_to_vp, f"Node '{b_name}' not found in constraint graph."

                for a_parameter in a_parameters:
                    assert a_parameter in self.name_to_vp[a_name], f"Parameter '{a_parameter}' not found in node '{a_name}' in constraint graph."
                    vertex_a = self.name_to_vp[a_name][a_parameter]

                    for b_parameter in b_parameters:
                        assert b_parameter in self.name_to_vp[b_name], f"Parameter '{b_parameter}' not found in node '{b_name}' in constraint graph."
                        vertex_b = self.name_to_vp[b_name][b_parameter]

                        e = self.graph.edge(vertex_a, vertex_b)
                        if e is None:
                            e = self.graph.add_edge(vertex_a, vertex_b)
                        self.graph.ep.constraint[e] = constraint
                        self.graph.ep.condition[e] = condition

    def generate(self):
        edge_set = set()

        expanded_architecture_desc = dict()
        expanded_workload_desc = dict()

        architecture_dependencies = {}
        workload_dependencies = {}
        architecture_sweeps = []
        workload_sweeps = []

        arch_work_dependencies = {}
        arch_work_sweeps = []

        # process all conditions
        for ojb in self.name_to_vp.values():
            for src_vertex in ojb.values():

                src_dict = dict()
                src_dict['name'] = self.graph.vp.name[src_vertex]
                src_dict['type'] = self.graph.vp.node_type[src_vertex]
                src_dict['parameter'] = self.graph.vp.node[src_vertex]
                src_dict['value'] = src_dict['parameter'].get_value()
                src_dict['sweep'] = src_dict['parameter'].get_sweep()
                src_dict['desc'] = self.graph.vp.desc[src_vertex]
                src_dict['key'] = (src_dict['desc'], src_dict['name'], src_dict['type'])

                sweep_list = []

                if src_vertex.out_degree() == 0 and src_vertex.in_degree() == 0:
                    
                    if src_dict['sweep']:
                        for sv in src_dict['value']:
                            sweep_list.append({'desc': src_dict['desc'], 'name': src_dict['name'], 'type': src_dict['type'], 'value': sv})
                           
                    else:
                        sweep_list.append({'desc': src_dict['desc'], 'name': src_dict['name'], 'type': src_dict['type'], 'value': src_dict['value']})
                    
                    if src_dict['desc'] == 'architecture':
                        architecture_sweeps.append(sweep_list)
                    else:
                        workload_sweeps.append(sweep_list)
                else:
                    for e in src_vertex.out_edges():
                        trg_vertex = e.target()
                        constraint_dict = {}
                        
                        # Edge direction is source -> target
                        param_order = False

                        trg_dict = dict()
                        trg_dict['name'] = self.graph.vp.name[trg_vertex]
                        trg_dict['type'] = self.graph.vp.node_type[trg_vertex]
                        trg_dict['parameter'] = self.graph.vp.node[trg_vertex]
                        trg_dict['value'] = trg_dict['parameter'].get_value()
                        trg_dict['sweep'] = trg_dict['parameter'].get_sweep()
                        trg_dict['desc'] = self.graph.vp.desc[trg_vertex]
                        trg_dict['key'] = (trg_dict['desc'], trg_dict['name'], trg_dict['type'])

                        constraint = self.graph.ep.constraint[e]
                        condition = self.graph.ep.condition[e]

                        if src_dict['desc'] == 'architecture' and trg_dict['desc'] == 'architecture':
                            src_constraint_dict = architecture_dependencies
                        elif src_dict['desc'] == 'workload' and trg_dict['desc'] == 'workload':
                            src_constraint_dict = workload_dependencies
                        else:
                            src_constraint_dict = arch_work_dependencies

                        if src_dict['desc'] not in src_constraint_dict:
                            src_constraint_dict[src_dict['desc']] = dict()
                        if src_dict['name'] not in src_constraint_dict[src_dict['desc']]:
                            src_constraint_dict[src_dict['desc']][src_dict['name']] = dict()
                        if src_dict['type'] not in src_constraint_dict[src_dict['desc']][src_dict['name']]:
                            src_constraint_dict[src_dict['desc']][src_dict['name']][src_dict['type']] = dict()

                        src_key_dict = src_constraint_dict[src_dict['desc']][src_dict['name']][src_dict['type']]

                        condition_type = None if condition is None else 'iter' if condition.__code__.co_argcount == 1 else 'comparison'

                        assert src_dict['sweep'] and trg_dict['sweep'], "Dependencies can only be defined between sweeping parameters."

                        if constraint == 'direct':
                            constraint_dict, constraint_sweep_list = self.direct_dependencies(src_dict, trg_dict, constraint, condition, condition_type, param_order)  
                        elif constraint == 'anti':
                            constraint_sweep_list = self.anti_dependencies(src_dict, trg_dict, constraint, condition, condition_type, param_order)
                        elif constraint == 'conditional':
                            constraint_dict, constraint_sweep_list = self.conditional_dependencies(src_dict, trg_dict, constraint, condition, condition_type, param_order)
    
                        if src_dict['key'] not in src_key_dict:
                            src_key_dict[src_dict['key']] = deepcopy(constraint_dict)
                        else:
                            for key, value in constraint_dict.items():
                                if key not in src_key_dict[src_dict['key']]:
                                    src_key_dict[src_dict['key']][key] = deepcopy(value)
                                else:
                                    for dep in value:
                                        if dep not in src_key_dict[src_dict['key']][key]:
                                            src_key_dict[src_dict['key']][key].append(deepcopy(dep))

                    for key, dep in src_key_dict[src_dict['key']].items():
                        for value in src_dict['value']:
                            value_dict = {'desc': src_dict['desc'], 'name': src_dict['name'], 'type': src_dict['type'],'value': value}
                            if not any(value_dict in d for d in src_key_dict[src_dict['key']].values()) and value_dict not in sweep_list:
                                sweep_list.append(value_dict)

                    if src_dict['desc'] == 'architecture' and trg_dict['desc'] == 'architecture' and sweep_list:
                        architecture_sweeps.append(sweep_list)
                    elif src_dict['desc'] == 'workload' and trg_dict['desc'] == 'workload' and sweep_list:
                        workload_sweeps.append(sweep_list)
                    elif sweep_list:
                        arch_work_sweeps.append(sweep_list)

        architecture_dependencies = self.connect_dependencies(architecture_dependencies)
        workload_dependencies = self.connect_dependencies(workload_dependencies)
        arch_work_dependencies = self.connect_dependencies(arch_work_dependencies)

        architecture_combinations = self.sweep(architecture_dependencies, architecture_sweeps)
        workload_combinations = self.sweep(workload_dependencies, workload_sweeps)
        arch_work_combinations = self.sweep(arch_work_dependencies, arch_work_sweeps)

        for i, arch in enumerate(architecture_combinations):
            arch_yaml = {'architecture': {'module': {}, 'attribute': {}}}
            for key, value in arch['architecture'].items():
                if key not in arch_yaml['architecture']['module']:
                    arch_yaml['architecture']['module'][key] = dict()
                for sub_key, sub_value in value.items():
                    if sub_key not in ['instance', 'tag']:
                        if 'query' not in arch_yaml['architecture']['module'][key]:
                            arch_yaml['architecture']['module'][key]['query'] = dict()
                        arch_yaml['architecture']['module'][key]['query'][sub_key] = deepcopy(sub_value)
                    else:
                        arch_yaml['architecture']['module'][key][sub_key] = deepcopy(sub_value)
            arch_yaml['architecture']['attribute'] = deepcopy(self.design.architecture.attributes)
                
            with open(self.design.architecture.yaml_path + f'config_{i}.architecture.yaml', 'w') as y:
                yaml.dump(arch_yaml, y)
            arch['path'] = self.design.architecture.yaml_path + f'config_{i}.architecture.yaml'

        for i, work in enumerate(workload_combinations):
            work_yaml = {'workload': {}}
            for key, value in work['workload'].items():
                work_yaml['workload'][key] = {'configuration': deepcopy(value)}

            with open(self.design.workload.yaml_path + f'config_{i}.workload.yaml', 'w') as y:
                yaml.dump(work_yaml, y)
            work['path'] = self.design.workload.yaml_path + f'config_{i}.workload.yaml'

        with open(self.design.run_path, 'w') as f:
            f.write('')

        event_path = self.design.event.yaml_path
        metric_path = self.design.metric.yaml_path
        
        i = 0
        if len(arch_work_combinations) > 0 and arch_work_combinations[0]:
            for arch_work in arch_work_combinations:
                arch_path = None
                work_path = None
                for arch in architecture_combinations:
                    arch_path = None
                    in_dict = self.dict_exists_in(arch_work['architecture'], arch['architecture'])
                    if in_dict:
                        arch_path = arch['path']
                    for work in workload_combinations:
                        work_path = None
                        in_dict = self.dict_exists_in(arch_work['workload'], work['workload'])
                        if in_dict:
                            work_path = work['path']
                        if arch_path is not None and work_path is not None:
                            run_path = self.design.yaml_path + f'/runs/config_{i}'
                            i = i+1
                            with open(self.design.run_path, 'a') as f:
                                f.write(f'-a {arch_path} -e {event_path} -m {metric_path} -r {run_path} -w {work_path} -c {run_path}/checkpoint.gt -s\n')
        else:
            with open(self.design.run_path, 'a') as f:
                print(self.design.run_path)
                f.write(f'-a {self.design.architecture.yaml_path}config_0.architecture.yaml -e {event_path} -m {metric_path} -r {self.design.yaml_path}/runs/config_0/ -w {self.design.workload.yaml_path}config_0.workload.yaml -c {self.design.yaml_path}/runs/config_0/checkpoint.gt -s\n')


        self.design.event.to_yaml()
        
        self.design.metric.to_yaml()

    def sweep(self, constraint_list, sweep_list):
        constraint_comb = list(product(*constraint_list))
        sweep_comb = list(product(*sweep_list))

        depedency_comb_flat = []
        for dep in constraint_comb:
            dep_list = []
            for item in dep:
                dep_list.extend(item)
            depedency_comb_flat.append(dep_list)

        sweep_dict_comb = []
        for comb in sweep_comb:
            sweep_dict = {}
            for item in comb:
                if item['desc'] not in sweep_dict:
                    sweep_dict[item['desc']] = {}
                if item['name'] not in sweep_dict[item['desc']]:
                    sweep_dict[item['desc']][item['name']] = {}
                if item['type'] not in sweep_dict[item['desc']][item['name']]:
                    sweep_dict[item['desc']][item['name']][item['type']] = item['value']
                else:
                    raise ValueError(f"Duplicate entry found for desc: {item['desc']}, name: {item['name']}, type: {item['type']}.")
            sweep_dict_comb.append(sweep_dict)

        total_sweeps = []
        for sweep in sweep_dict_comb:
            for dep in depedency_comb_flat:
                dep_combination = deepcopy(sweep)
                base_combination = deepcopy(sweep)
                for value in dep:
                    exists = self.nested_key_exists(base_combination, [value['desc'], value['name'], value['type']])
                    if exists is False:
                        self.nested_set(dep_combination, [value['desc'], value['name'], value['type']], value['value'])
                        self.nested_set(base_combination, [value['desc'], value['name'], value['type']], value['value'])
                    else:
                        self.nested_set(dep_combination, [value['desc'], value['name'], value['type']], value['value'])
                total_sweeps.append(dep_combination)
                total_sweeps.append(base_combination)

        total_combinations = self.remove_duplicate_dicts(total_sweeps)

        return total_combinations

    def list_to_dict(self, lst: list):
        dct = {}
        for item in lst:
            if item['desc'] not in dct:
                dct[item['desc']] = {}
            if item['name'] not in dct[item['desc']]:
                dct[item['desc']][item['name']] = {}
            if item['type'] not in dct[item['desc']][item['name']]:
                dct[item['desc']][item['name']][item['type']] = item['value']
            else:
                raise ValueError(f"Duplicate entry found for desc: {item['desc']}, name: {item['name']}, type: {item['type']}.")

    def print_dict(self, d, indent=0):
        tab = "\t" * indent

        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    print(f"{tab}{key}:")
                    self.print_dict(value, indent + 1)
                else:
                    print(f"{tab}{key}: {value}")

        elif isinstance(d, list):
            for item in d:
                self.print_dict(item, indent)
        else:
            print(f"{tab}{d}")

    def dicts_equal(self, d1, d2):
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            # If both are not dicts, compare directly
            return d1 == d2
        
        # Check if both have the same keys
        if d1.keys() != d2.keys():
            return False
        
        # Recursively check each key
        for key in d1:
            if not self.dicts_equal(d1[key], d2[key]):
                return False
        
        return True

    def dict_exists_in(self, d1: dict, d2: dict, debug: bool = False, path: str = "") -> bool:
        """
        Check if all nested keys and values from d1 exist in d2.
        This is a one-way check - verifies that d1 is a subset of d2.
        
        Args:
            d1: The dictionary to check for (subset)
            d2: The dictionary to check in (superset)
            debug: If True, prints debug information about mismatches
            path: Current path in the nested structure (for debugging)
            
        Returns:
            True if all keys and values from d1 exist in d2, False otherwise
        """
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            # If d1 is not a dict, compare values directly
            match = d1 == d2
            if debug and not match:
                print(f"VALUE MISMATCH at '{path}': d1={d1} (type: {type(d1).__name__}) != d2={d2} (type: {type(d2).__name__})")
            return match
        
        # Check if all keys from d1 exist in d2
        for key in d1:
            current_path = f"{path}.{key}" if path else str(key)
            
            if key not in d2:
                if debug:
                    print(f"KEY MISSING at '{current_path}': key '{key}' exists in d1 but not in d2")
                return False
            
            # Recursively check nested structures
            if isinstance(d1[key], dict):
                if not isinstance(d2[key], dict):
                    if debug:
                        print(f"TYPE MISMATCH at '{current_path}': d1['{key}'] is dict but d2['{key}'] is {type(d2[key]).__name__}")
                    return False
                if not self.dict_exists_in(d1[key], d2[key], debug, current_path):
                    return False
            else:
                # For non-dict values, check equality
                if d1[key] != d2[key]:
                    if debug:
                        print(f"VALUE MISMATCH at '{current_path}': d1['{key}']={d1[key]} (type: {type(d1[key]).__name__}) != d2['{key}']={d2[key]} (type: {type(d2[key]).__name__})")
                    return False
        
        return True

    def remove_duplicate_dicts(self, dict_list):
        unique_dicts = []
        for d in dict_list:
            if not any(self.dicts_equal(d, u) for u in unique_dicts):
                unique_dicts.append(d)
        return unique_dicts

    def nested_key_exists(self, d, keys):
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return False
        return True

    def nested_set(self, d, keys, value):
        if not keys:
            return
        
        key = keys[0]
        if len(keys) == 1:
            d[key] = value
            return
        
        # If the next level doesn't exist or is not a dict, create a dict
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        
        # Recurse into the next level
        self.nested_set(d[key], keys[1:], value)

    def connect_dependencies(self, constraint_dict):

        constraint_list = []
        for desc_key, desc_value in constraint_dict.items():
            for name_key, name_value in desc_value.items():
                for type_key, type_value in name_value.items():
                    for param_key, param_value in type_value.items():
                        for value in param_value.values():
                            constraint_list.append(value)

        combined_dep_list = []
        item_all = False         
        for dep in constraint_list:
            for compare_dep in constraint_list:
                dep = self.combine_shared_value(dep, compare_dep)
            for value in combined_dep_list:
                if all(item in dep for item in value):
                    item_all = True
            if not item_all:
                combined_dep_list.append(dep)

        structured_combined_dep_list = []
        used_indices = set()

        for i, dep in enumerate(combined_dep_list):
            if i in used_indices:
                continue

            # Use only desc, name, type for grouping
            dep_set = set((item['desc'], item['name'], item['type']) for item in dep)
            dep_group = [dep]
            used_indices.add(i)

            changed = True
            while changed:
                changed = False
                for j, compare_dep in enumerate(combined_dep_list):
                    if j in used_indices:
                        continue
                    compare_set = set((item['desc'], item['name'], item['type']) for item in compare_dep)
                    if dep_set & compare_set:  # intersection exists
                        dep_group.append(compare_dep)
                        used_indices.add(j)
                        dep_set |= compare_set  # expand the set to include new keys
                        changed = True

            structured_combined_dep_list.append(dep_group)

        return structured_combined_dep_list

    def combine_shared_value(self, list1, list2):
        shared = any(item in list2 for item in list1)

        combined = []
        seen = set()
        if shared:
            for item in list1 + list2:
                key = self.make_hashable(item)
                if key not in seen:
                    combined.append(item.copy())
                    seen.add(key)
            return combined
        else:
            return list1

    def direct_dependencies(self, src: dict, trg: dict, constraint: str, condition: callable, condition_type: str, param_order: bool):

        constraint_dict = {}
        sweep_list = []

        if constraint == 'direct':
            if condition is None:
                assert len(src['value']) == len(trg['value']), f"Direct constraint between parameter '{src['type']}' in node '{src['name']}' and parameter '{trg['type']}' in node '{trg['name']}' requires both parameters to have the same sweep length."
                for sv, tv in zip(src['value'], trg['value']):
                    constraint_dict[src['key'] + self.make_hashable((sv,))] = [
                        {'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv},
                        {'desc': trg['desc'], 'name': trg['name'], 'type': trg['type'], 'value': tv}
                    ]
            else:
                if condition_type == 'iter':
                    for i, (sv, tv) in enumerate(zip_longest(src['value'], trg['value'], fillvalue=None)):
                        if condition(i):
                            assert sv is not None and tv is not None, "Direct constraint with iteration condition is not valid for parameters with different sweep lengths."
                            constraint_dict[src['key'] + self.make_hashable((sv,))] = [
                                {'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv},
                                {'desc': trg['desc'], 'name': trg['name'], 'type': trg['type'], 'value': tv}
                            ]
                        else:
                            if sv is not None:
                                sweep_list.append({'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv})
                                
                elif condition_type == 'comparison':
                    for sv in src['value']:
                        for tv in trg['value']:
                            # Apply condition with correct parameter order
                            if param_order:
                                condition_result = condition(tv, sv)
                            else:
                                condition_result = condition(sv, tv)
                            if condition_result:
                                constraint_dict[src['key'] + self.make_hashable((sv,))] = [
                                    {'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv},
                                    {'desc': trg['desc'], 'name': trg['name'], 'type': trg['type'], 'value': tv}
                                ]
                            else:
                                sweep_list.append({'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv})
        return constraint_dict, sweep_list
                                            
    def anti_dependencies(self, src: dict, trg: dict, constraint: str, condition: callable, condition_type: str, param_order: bool):

        sweep_list = []

        if constraint == 'anti':
            if condition is None:
                raise ValueError("Anti constraint without condition is not supported.")
            else:
                if condition_type == 'iter':
                    for i, (sv, tv) in enumerate(zip_longest(src['value'], trg['value'], fillvalue=None)):
                        if condition(i):
                            assert sv is not None or tv is not None, "Iteration Anti constraint cannot exist between parameters that produce different sweep lengths."
                            continue
                        else:
                            if sv is not None:
                                sweep_list.append([{'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv}])
                elif condition_type == 'comparison':
                    for sv in src['value']:
                        for tv in trg['value']:
                            # Apply condition with correct parameter order
                            if param_order:
                                condition_result = condition(tv, sv)
                            else:
                                condition_result = condition(sv, tv)
                            
                            if condition_result:
                                continue
                            else:
                                sweep_list.append([{'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv}])
        return sweep_list

    def conditional_dependencies(self, src: dict, trg: dict, constraint: str, condition: callable, condition_type: str, param_order: bool):
        # Placeholder for future implementation
        raise NotImplementedError("Conditional dependencies are not yet implemented.")

    def add_to_constraint_dict(self, src, trg, sv, tv, generated_dict, constraint_dict):

        src_key = (src['desc'], src['name'], src['type'], self.make_hashable(sv))
        trg_key = (trg['desc'], trg['name'], trg['type'], self.make_hashable(tv))

        if generated_dict is None:
            constraint_dict[trg_key] = [
                {'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv},
                {'desc': trg['desc'], 'name': trg['name'], 'type': trg['type'], 'value': tv}
            ]
        if generated_dict is not None:
            if src_key in generated_dict:
                constraint_dict[src_key] = generated_dict[src_key].append(
                    {'desc': trg['desc'], 'name': trg['name'], 'type': trg['type'], 'value': tv}
                )
            else:
                constraint_dict[trg_key] = [
                    {'desc': src['desc'], 'name': src['name'], 'type': src['type'], 'value': sv},
                    {'desc': trg['desc'], 'name': trg['name'], 'type': trg['type'], 'value': tv}
                ]

    def add_to_dict_as_set(self, d: dict, desc: str, name: str, type: str, value):
        if desc not in d:
            d[desc] = dict()
        if name not in d[desc]:
            d[desc][name] = dict()
        if type not in d[desc][name]:
            d[desc][name][type] = set()
        d[desc][name][type].add(self.make_hashable(value))

    def sweep_combinations(self, sweep_list: list, constraint_list: list, path: str):
        
        sweeps = []
        if constraint_list:
            for dep_entry in constraint_list:
                for _, dependent_dicts in dep_entry.items():
                    sweeps.append(dependent_dicts)
        
        for sweep in sweep_list:
            sweeps.append([sweep])

        sweeps_dict = {}

        def unique_key(item):
            return (item['desc'], item['name'], item['type'])

        valid_combinations = []
        seen = set()

        for i, sweep_static in enumerate(tqdm(sweeps, desc="Processing sweeps", unit="sweep")):
            sweep_dict = dict()
            for item_static in sweep_static:
                if item_static['desc'] not in sweep_dict:
                    sweep_dict[item_static['desc']] = dict()
                if item_static['name'] not in sweep_dict[item_static['desc']]:
                    sweep_dict[item_static['desc']][item_static['name']] = dict()
                sweep_dict[item_static['desc']][item_static['name']][item_static['type']] = item_static['value']

            for sweep_dynamic in sweeps:
                for item_dynamic in sweep_dynamic:
                    if item_dynamic['desc'] not in sweep_dict:
                        sweep_dict[item_dynamic['desc']] = dict()
                    if item_dynamic['name'] not in sweep_dict[item_dynamic['desc']]:
                        sweep_dict[item_dynamic['desc']][item_dynamic['name']] = dict()
                    if item_dynamic['type'] not in sweep_dict[item_dynamic['desc']][item_dynamic['name']]:
                        sweep_dict[item_dynamic['desc']][item_dynamic['name']][item_dynamic['type']] = item_dynamic['value']
                    
                
            
            with open(os.path.normpath(f"{path}/config{i}.yaml"), "w") as f:
                yaml.dump(sweep_dict, f)
            
    def make_hashable(self, obj):
        if isinstance(obj, dict):
            return tuple((self.make_hashable(k), self.make_hashable(v)) for k, v in sorted(obj.items()))
        elif isinstance(obj, (list, tuple, set)):
            return tuple(self.make_hashable(x) for x in obj)
        else:
            return obj
        
    # TODO: working changes
    # Flow
    # split nodes where dependencies differ of edge
    #  - loop through each edge, check dependencies on all nodes remake constraint
    #  - put trg conditional constraint after generated on edge (only target value)
    #  - loop through values and remove values not in edge constraint
    # split graph into subgraphs on dependencies
    # sweep subgraphs independently
    # sweep subgraphs globally
    # write to yaml

    # change to directed maybe to maintain condition (a, b)

    def new_graph(self, directed: bool = True):
        graph = gt.Graph(directed=directed)

        # vertex properties
        graph.vp.name = graph.new_vertex_property("string")
        graph.vp.node = graph.new_vertex_property("object")
        graph.vp.value = graph.new_vertex_property("object")
        graph.vp.node_type = graph.new_vertex_property("string")
        graph.vp.desc = graph.new_vertex_property("string")
        graph.vp.constraint_dict = graph.new_vertex_property("object")
        graph.vp.sweep_list = graph.new_vertex_property("object")

        # edge properties
        graph.ep.constraint = graph.new_edge_property("string")       
        graph.ep.condition = graph.new_edge_property("object")
        graph.ep.processed = graph.new_edge_property("bool")
        graph.ep.src_constraints = graph.new_edge_property("object")
        graph.ep.trg_constraints = graph.new_edge_property("object")
        graph.ep.src_sweep = graph.new_edge_property("object")
        graph.ep.trg_sweep = graph.new_edge_property("object")

        return graph
    
    def generate_test(self):
        self.group_graph()

        for subgraph in self.subgraphs:
            edges = list(subgraph.edges())
            for e in edges:
                self.process_constraints(e, subgraph)

            nodes = list(subgraph.vertices())
            expanded_subgraph = self.new_graph(directed=False)
            for n in nodes:
                self.split_node(n, subgraph, expanded_subgraph)


    def group_graph(self):
        '''
        Splits graph into subgraphs where each subgraph contains nodes with constraints.
        Nodes that are not connected by any constraints create separate subgraphs.
        '''
        comp, hist = gt.label_components(self.graph)
        for c in range(len(hist)):
            vfilt = self.graph.new_vertex_property("bool")
            vfilt.a = (comp.a == c)  # True for vertices in this component
            self.subgraphs.append(gt.GraphView(self.graph, vfilt=vfilt))

    def process_constraints(self, edge, subgraph):
        edge_dict = self.get_edge_dict(edge, subgraph)
        if not subgraph.ep.processed[edge]:
            src_vertex = edge.source()
            trg_vertex = edge.target()
            src_dict = self.get_vertex_dict(src_vertex, subgraph)
            trg_dict = self.get_vertex_dict(trg_vertex, subgraph)
            
            if edge_dict['constraint'] == 'direct':
                src_dependencies, trg_dependencies, src_sweep, trg_sweep = self.direct_constraint_new(src_dict, trg_dict, edge_dict)
            elif edge_dict['constraint'] == 'anti':
                pass
            elif edge_dict['constraint'] == 'conditional':
                pass

            subgraph.ep.processed[edge] = True
            subgraph.ep.src_constraints[edge] = src_dependencies
            subgraph.ep.trg_constraints[edge] = trg_dependencies
            subgraph.ep.src_sweep[edge] = src_sweep
            subgraph.ep.trg_sweep[edge] = trg_sweep

    def split_node(self, vertex, subgraph, expanded_subgraph):
        src_dict = self.get_vertex_dict(vertex, subgraph)

        if src_dict['sweep']:
            print(src_dict['sweep'])
            for value in src_dict['value']:
                print(value)
                pass
                self.copy_node(vertex, value, subgraph, expanded_subgraph)
        else:
            self.copy_node(vertex, src_dict['value'], subgraph, expanded_subgraph)
            # add to node
                
    def copy_node(self, vertex, value, subgraph, expanded_subgraph):
        new_vertex = expanded_subgraph.add_vertex()
        expanded_subgraph.vp.name[new_vertex] = subgraph.vp.name[vertex]
        expanded_subgraph.vp.node_type[new_vertex] = subgraph.vp.node_type[vertex]
        expanded_subgraph.vp.node[new_vertex] = subgraph.vp.node[vertex]
        expanded_subgraph.vp.value[new_vertex] = value
        expanded_subgraph.vp.desc[new_vertex] = subgraph.vp.desc[vertex]
        print('value', expanded_subgraph.vp.name[new_vertex], expanded_subgraph.vp.value[new_vertex])
        return new_vertex

            

    def get_vertex_dict(self, vertex, graph):
        vertex_dict = dict()
        vertex_dict['name'] = graph.vp.name[vertex]
        vertex_dict['type'] = graph.vp.node_type[vertex]
        vertex_dict['parameter'] = graph.vp.node[vertex]
        vertex_dict['value'] = vertex_dict['parameter'].get_value()
        vertex_dict['sweep'] = vertex_dict['parameter'].get_sweep()
        vertex_dict['desc'] = graph.vp.desc[vertex]
        return vertex_dict
    
    def get_edge_dict(self, edge, graph):
        edge_dict = dict()
        edge_dict['constraint'] = graph.ep.constraint[edge]
        edge_dict['condition'] = graph.ep.condition[edge]
        edge_dict['processed'] = graph.ep.processed[edge]
        edge_dict['src_constraints'] = graph.ep.src_constraints[edge]
        edge_dict['trg_constraints'] = graph.ep.trg_constraints[edge]
        edge_dict['src_sweep'] = graph.ep.src_sweep[edge]
        edge_dict['trg_sweep'] = graph.ep.trg_sweep[edge]
        condition = edge_dict['condition']
        edge_dict['condition_type'] = None if condition is None else 'iter' if condition.__code__.co_argcount == 1 else 'comparison'
        return edge_dict
    
    def direct_constraint_new(self, src_dict, trg_dict, edge_dict):
        '''
        Processes direct constraints between source and target parameters.
        Direct constraints directly link parameters from source to target.
        Parameters not defined by the constraint are considered sweeping.
        '''
        src_dependencies = []
        trg_dependencies = []
        src_sweep = []
        trg_sweep = []

        # No condition: direct one-to-one mapping
        if edge_dict['condition_type'] is None:
            assert len(src_dict['value']) == len(trg_dict['value']), f"Direct constraint between parameter '{src_dict['type']}' in node '{src_dict['name']}' and parameter '{trg_dict['type']}' in node '{trg_dict['name']}' requires both parameters to have the same length."
            for src_value, trg_value in zip(src_dict['value'], trg_dict['value']):
                src_dependencies.append(src_value)
                trg_dependencies.append(trg_value)
        # Iteration condition
        elif edge_dict['condition_type'] == 'iter':
            for i, (src_value, trg_value) in enumerate(zip_longest(src_dict['value'], trg_dict['value'], fillvalue=None)):
                if edge_dict['condition'](i):
                    assert src_value is not None and trg_value is not None, "Direct constraint with iteration condition is not valid for parameters with different sweep lengths."
                    src_dependencies.append(src_value)
                    trg_dependencies.append(trg_value)
                else:
                    if src_value is not None:
                        src_sweep.append(src_value)
                    if trg_value is not None:
                        trg_sweep.append(trg_value)
        # Comparison condition
        elif edge_dict['condition_type'] == 'comparison':
            for src_value in src_dict['value']:
                for trg_value in trg_dict['value']:
                    # Directed edge: src -> trg, so condition is always (src, trg)
                    condition_result = edge_dict['condition'](src_value, trg_value)
                    if condition_result:
                        src_dependencies.append(src_value)
                        trg_dependencies.append(trg_value)
                    else:
                        src_sweep.append(src_value)
                        trg_sweep.append(trg_value)
        return src_dependencies, trg_dependencies, src_sweep, trg_sweep

    def anti_constraint_new(self, src_dict, trg_dict, edge_dict):
        pass

    def conditional_constraint_new(self, src_dict, trg_dict, edge_dict):
        pass