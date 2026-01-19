import graph_tool.all as gt
from itertools import product, zip_longest
from yaml import safe_dump
from copy import deepcopy
import os
import tqdm

# Helper functions
# create graph
# add vertex (in each frontend components)
# create edge
# split vertex
#   takes a vertex and separates it into n vertices, and then reconnects the edges

# TODO:
#  - Multiple ways to add constraints
#  - add in anti and conditional constraints
#  - how to pass constraints to select correct vertex?
#  - What if you want to name a parameter and module the same name.
#  - edges get processed twice in split vertices for the same vertex
#  - currently naivly copies arch, workload dicts. Optimize this.
#  - value conllisions when trying to do direct list constraints (i.e. inst=[1,2,3] and inst=[1,1,1]), the [1,1,1] has collisions.

from frontend.architecture import Architecture
from archx.programming.object.event import Event
from frontend.metric import Metric
from frontend.workload import Workload

class AGraph:
    def __init__(self, path):
        self.path = path
        self._create_graph()
        self.subgraphs = []
        self.architecture = Architecture(self.graph)
        self.event = Event()
        self.metric = Metric()
        self.workload = Workload(self.graph)

        self.constr_set = set()

    def add_constraint(self, **kwargs):

        for key_a, value_a in kwargs.items():
            for param_a in value_a:
                vertex_a = self._get_vertex(name=key_a, param_name=param_a)
                for key_b, value_b in kwargs.items():
                    for param_b in value_b:
                        vertex_b = self._get_vertex(name=key_b, param_name=param_b)
                        if vertex_a != vertex_b:
                            # v_tuple = (int(vertex_a), int(vertex_b))
                            # v_tuple_rev = (int(vertex_b), int(vertex_a))
                            # if v_tuple not in self.constr_set and v_tuple_rev not in self.constr_set:
                            #     self.constr_set.add(v_tuple)
                            # else:
                            #     continue
                            self._add_edge(vertex_a, vertex_b, type='direct')
                            
    def generate(self):
        print("Generating configuration constraints...")
        self._compute_constraint()
        print("Splitting sweeping vertices...")
        self._split_vertices()
        print("Grouping graph...")
        self._group_graph()
        print("Grouping direct constraints...")
        self._group_direct_constraints()
        print("Collapsing direct constraints...")
        self._collapse_direct_constraints()
        print("Reducing anti constraints...")
        self._reduce_anti_constraints()
        print(f"Generated {len(self.configurations)} configurations.")
        self._to_yaml()
        print('Done!')
        

    def _create_graph(self):
        self.graph = gt.Graph(directed=False)

        # vertex properties
        self.graph.vp.name = self.graph.new_vertex_property("string")
        self.graph.vp.param_name = self.graph.new_vertex_property("string")
        self.graph.vp.value = self.graph.new_vertex_property("object")
        self.graph.vp.type = self.graph.new_vertex_property("string")
        self.graph.vp.sweep = self.graph.new_vertex_property("bool")
        self.graph.vp.desc = self.graph.new_vertex_property("string")
        self.graph.vp.idx = self.graph.new_vertex_property("int")

        # edge properties
        self.graph.ep.type = self.graph.new_edge_property("string")       
        self.graph.ep.condition = self.graph.new_edge_property("object")
        self.graph.ep.condition_type = self.graph.new_edge_property("string")
        self.graph.ep.direction = self.graph.new_edge_property("int")
        self.graph.ep.constraint = self.graph.new_edge_property("object")

    def _get_vertex(self, name, param_name):
        for v in self.graph.vertices():
            if self.graph.vp.name[v] == name and self.graph.vp.param_name[v] == param_name:
                return v
        assert False, f"vertex with name '{name}' and type '{param_name}' not found."

    def _add_edge(self, vertex_a, vertex_b, type, condition=None, condition_type='', direction=-1, constraint=None):
        edge = self.graph.add_edge(vertex_a, vertex_b)
        self.graph.ep.type[edge] = type
        self.graph.ep.condition[edge] = condition
        self.graph.ep.condition_type[edge] = condition_type
        self.graph.ep.direction[edge] = direction
        self.graph.ep.constraint[edge] = constraint
        return edge

    def _compute_constraint(self):
        edges = self.graph.get_edges()
        for e in edges:
            edge_dict = self._get_edge_dict(e)
            if edge_dict['direction'] != -1:
                src_v = self.graph.vertex(edge_dict['direction'])
                trg_v = e[1] if e[0] == src_v else e[0]
            else:
                src_v, trg_v = e[0], e[1]

            src_dict = self._get_vertex_dict(src_v)
            trg_dict = self._get_vertex_dict(trg_v)

            assert src_dict['sweep'] and trg_dict['sweep'], f"Both source parameter '{src_dict['param_name']}' in vertex '{src_dict['name']}' and target parameter '{trg_dict['param_name']}' in vertex '{trg_dict['name']}' must be sweeping to apply constraints."

            if edge_dict['type'] == 'direct':
                src_constraints, trg_constraints = self.direct_constraint(src_dict, trg_dict, edge_dict)
            else:
                raise NotImplementedError(f"Constraint type '{edge_dict['type']}' not implemented yet.")
            
            if src_constraints != [] and trg_constraints != []:
                self.graph.ep.constraint[e] = {
                    (src_dict['name'], src_dict['param_name'], src_dict['type'], src_dict['desc']): src_constraints,
                    (trg_dict['name'], trg_dict['param_name'], trg_dict['type'], trg_dict['desc']): trg_constraints
                }

    def _get_edge_dict(self, e):
        edge_dict = {
            'type': self.graph.ep.type[e],
            'condition': self.graph.ep.condition[e],
            'condition_type': self.graph.ep.condition_type[e],
            'direction': self.graph.ep.direction[e],
            'constraint': self.graph.ep.constraint[e]
        }
        return edge_dict
    
    def _get_vertex_dict(self, v):
        vertex_dict = {
            'name': self.graph.vp.name[v],
            'param_name': self.graph.vp.param_name[v],
            'value': self.graph.vp.value[v],
            'type': self.graph.vp.type[v],
            'sweep': self.graph.vp.sweep[v],
            'desc': self.graph.vp.desc[v],
            'idx': self.graph.vp.idx[v]
        }
        return vertex_dict

    def _split_vertices(self):
        sweep_vertices_data = [(int(v), self._get_vertex_dict(v)) for v in self.graph.vertices() if self.graph.vp.sweep[v]]

        for src_v_idx, src_dict in sweep_vertices_data:
            if not src_dict['sweep']:
                continue
            
            src_v = self.graph.vertex(src_v_idx)
            edges = self.graph.get_out_edges(src_v)

            new_vertices = []
            for value in src_dict['value']:

                new_vertex = self._add_node(name=src_dict['name'],
                                            param_name=src_dict['param_name'],
                                            value=value,
                                            type=src_dict['type'],
                                            sweep=False,
                                            desc=src_dict['desc'])
                new_vertices.append(new_vertex)

                processed_edges = set()
                for e in edges:
                    if (e[0], e[1]) in processed_edges:
                        continue
                    processed_edges.add((e[0], e[1]))
                    edge_dict = self._get_edge_dict(e)
                    trg_v = e[1] if e[0] == src_v else e[0]
                    trg_dict = self._get_vertex_dict(trg_v)

                    src_constraints = edge_dict['constraint'][(src_dict['name'], src_dict['param_name'], src_dict['type'], src_dict['desc'])]
                    trg_constraints = edge_dict['constraint'][(trg_dict['name'], trg_dict['param_name'], trg_dict['type'], trg_dict['desc'])]

                    if isinstance(src_constraints, list) and isinstance(trg_constraints, list):
                        constraints = zip(src_constraints, trg_constraints)
                        for src_c, trg_c in constraints:
                            if src_c == value:

                                new_edge = self._add_edge(new_vertex, trg_v,
                                                          type=edge_dict['type'],
                                                          condition=edge_dict['condition'],
                                                          condition_type=edge_dict['condition_type'])
                                
                                self.graph.ep.constraint[new_edge] = {
                                    ((src_dict['name'], src_dict['param_name'], src_dict['type'], src_dict['desc'])): src_c,
                                    ((trg_dict['name'], trg_dict['param_name'], trg_dict['type'], trg_dict['desc'])): trg_c
                                }
                                # Debug: print when direct edges are created
                                #print(f"  Created direct-edge {new_edge}: {src_dict['name']}.{src_dict['param_name']} {value} -> {trg_dict['name']}.{trg_dict['param_name']} {trg_dict['value']}")
                                break

                    elif src_constraints == value:
                        new_edge = self._add_edge(new_vertex, trg_v,
                                    type=edge_dict['type'],
                                    condition=edge_dict['condition'],
                                    condition_type=edge_dict['condition_type'])
                        # Debug: print when direct edges are created
                        #print(f"  Created direct-edge {new_edge}: {src_dict['name']}.{src_dict['param_name']} {value} -> {trg_dict['name']}.{trg_dict['param_name']} {trg_dict['value']}")

            for new_src_v in new_vertices:
                for new_trg_v in new_vertices:
                    if new_src_v != new_trg_v:
                        existing_edge = self.graph.edge(new_src_v, new_trg_v)
                        if existing_edge is None:
                            existing_edge = self.graph.edge(new_trg_v, new_src_v)
                            if existing_edge is None:
                                anti_edge = self._add_edge(new_src_v, new_trg_v, type='anti')
                                # Debug: print when anti-edges are created
                                src_val = self.graph.vp.value[new_src_v]
                                trg_val = self.graph.vp.value[new_trg_v]
                                #print(f"  Created anti-edge {anti_edge}: {src_dict['name']}.{src_dict['param_name']} {src_val} <-> {trg_val}")
            for e in edges:
                edge = self.graph.edge(e[0], e[1])
                #print(f"  Removed edge {edge} between sweeping vertex '{src_dict['name']}.{src_dict['param_name']}' and vertex '{self.graph.vp.name[trg_v]}.{self.graph.vp.param_name[trg_v]}'")
                self.graph.remove_edge(edge)
                

        for src_v_idx, src_dict in reversed(sweep_vertices_data):
            if src_v_idx < self.graph.num_vertices():
                src_v = self.graph.vertex(src_v_idx)
                self.graph.remove_vertex(src_v)

    def _group_graph(self):
        comp, hist = gt.label_components(self.graph)
    
        for c in range(len(hist)):
            vfilt = self.graph.new_vertex_property("bool")
            vfilt.a = (comp.a == c)  # True for vertices in this component
            subgraph = gt.GraphView(self.graph, vfilt=vfilt)
            self.subgraphs.append(subgraph)
        
        # print(f"Total subgraphs: {len(self.subgraphs)}")
        # for i, subgraph in enumerate(self.subgraphs):
        #     print(f"  Subgraph {i}: {subgraph.num_vertices()} nodes")

    def _add_node(self, name: str, param_name: str, value, type: str, sweep: bool, desc: str, idx: int = 0):
        vertex = self.graph.add_vertex()
        self.graph.vp.name[vertex] = name
        self.graph.vp.param_name[vertex] = param_name
        self.graph.vp.value[vertex] = value
        self.graph.vp.type[vertex] = type
        self.graph.vp.sweep[vertex] = sweep
        self.graph.vp.desc[vertex] = desc
        self.graph.vp.idx[vertex] = idx
        return vertex

    def direct_constraint(self, src_dict, trg_dict, edge_dict):
        '''
        Processes direct constraints between source and target parameters.
        Direct constraints directly link parameters from source to target.
        Parameters not defined by the constraint are considered sweeping.
        '''
        src_dependencies = []
        trg_dependencies = []
        # No condition: direct one-to-one mapping
        if not edge_dict['condition_type']:
            assert len(src_dict['value']) == len(trg_dict['value']), f"Direct constraint between parameter '{src_dict['type']}' in vertex '{src_dict['name']}' and parameter '{trg_dict['type']}' in vertex '{trg_dict['name']}' requires both parameters to have the same length."
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
        # Comparison condition
        elif edge_dict['condition_type'] == 'comparison':
            for src_value in src_dict['value']:
                for trg_value in trg_dict['value']:
                    # Directed edge: src -> trg, so condition is always (src, trg)
                    condition_result = edge_dict['condition'](src_value, trg_value)
                    if condition_result:
                        src_dependencies.append(src_value)
                        trg_dependencies.append(trg_value)

        return src_dependencies, trg_dependencies
    
    def _group_direct_constraints(self):
        new_subgraphs = []
        
        for subgraph in self.subgraphs:
            # Create edge filter that only includes 'direct' type edges
            efilt = subgraph.new_edge_property("bool")
            for e in subgraph.edges():
                efilt[e] = (subgraph.ep.type[e] == 'direct')
            
            direct_view = gt.GraphView(subgraph, efilt=efilt)
            
            comp, hist = gt.label_components(direct_view)
            
            # Create list of components for this subgraph
            subgraph_components = []
            for c in range(len(hist)):
                vfilt = subgraph.new_vertex_property("bool")
                vfilt.a = (comp.a == c)
                component_subgraph = gt.GraphView(subgraph, vfilt=vfilt)
                subgraph_components.append(component_subgraph)
            
            new_subgraphs.append(subgraph_components)
        
        self.subgraphs = new_subgraphs
        
        # total_subgraphs = sum(len(sg) for sg in self.subgraphs)
        # # print(f"Total subgraphs after direct constraint grouping: {total_subgraphs}")
        # # for i, subgraph_list in enumerate(self.subgraphs):
        # #     print(f"  Group {i}: {len(subgraph_list)} subgraphs")
        # #     for j, subgraph in enumerate(subgraph_list):
        # #         print(f"    Subgraph {j}: {subgraph.num_vertices()} nodes")

    def _collapse_direct_constraints(self):
        
        self.grouped_dicts = []

        for subgraph in self.subgraphs:
            collapsed_dicts = []
            for split_subgraph in subgraph:
                arch_dict = {'architecture': {}}
                workload_dict = {'workload': {}}

                for v in split_subgraph.vertices():
                    v_dict = self._get_vertex_dict(v)
                    if v_dict['desc'] == 'architecture':
                        if v_dict['type'] not in arch_dict['architecture']:
                            arch_dict['architecture'][v_dict['type']] = {}
                        # Attribute
                        if v_dict['type'] == 'attribute':
                            arch_dict['architecture'][v_dict['type']][v_dict['param_name']] = deepcopy(v_dict['value'])
                        # Module
                        elif v_dict['type'] == 'module':
                            if v_dict['name'] not in arch_dict['architecture'][v_dict['type']]:
                                arch_dict['architecture'][v_dict['type']][v_dict['name']] = {}
                            if v_dict['param_name'] in ['instance', 'tag']:
                                arch_dict['architecture'][v_dict['type']][v_dict['name']][v_dict['param_name']] = deepcopy(v_dict['value'])
                            else:
                                if 'query' not in arch_dict['architecture']['module'][v_dict['name']]:
                                    arch_dict['architecture']['module'][v_dict['name']]['query'] = {}
                                arch_dict['architecture']['module'][v_dict['name']]['query'][v_dict['param_name']] = deepcopy(v_dict['value'])

                    elif v_dict['desc'] == 'workload':
                        if v_dict['name'] not in workload_dict['workload']:
                            workload_dict['workload'][v_dict['name']] = {}
                        if 'configuration' not in workload_dict['workload'][v_dict['name']]:
                            workload_dict['workload'][v_dict['name']]['configuration'] = {}
                        workload_dict['workload'][v_dict['name']]['configuration'][v_dict['param_name']] = deepcopy(v_dict['value'])

                constraint_dict = {
                    'arch': arch_dict,
                    'work': workload_dict,
                }
                collapsed_dicts.append(constraint_dict)
            self.grouped_dicts.append(collapsed_dicts)

    def _reduce_anti_constraints(self):
        self.configurations = []

        # for group in self.grouped_dicts:
        #     if len(group) > 1:
        #         print(len(group))


        total_configs = list(product(*self.grouped_dicts))
        for group in tqdm.tqdm(total_configs, desc="Merging configurations"):
            self.configurations.append(self._combine_dicts(group))

    def _combine_dicts(self, dict_list):
        if not dict_list:
            return {}
        
        if len(dict_list) == 1:
            return dict_list[0]
        
        def merge_recursive(d1, d2, path=""):
            merged = {}
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key in d1 and key in d2:
                    v1, v2 = d1[key], d2[key]
                    
                    if isinstance(v1, dict) and isinstance(v2, dict):
                        merged[key] = merge_recursive(v1, v2, current_path)
                    elif v1 == v2:
                        merged[key] = v1
                    else:
                        raise ValueError(f"Collision at '{current_path}': value1={repr(v1)}, value2={repr(v2)}")
                elif key in d1:
                    merged[key] = d1[key]
                else:
                    merged[key] = d2[key]
            
            return merged
        
        # Merge all dictionaries sequentially
        result = dict_list[0]
        for i, next_dict in enumerate(dict_list[1:], start=1):
            try:
                result = merge_recursive(result, next_dict)
            except ValueError as e:
                raise ValueError(f"Error merging dict at index {i}: {e}")
        
        return result

    def _to_yaml(self):

        runs = []
        runs_txt = self.path + 'runs.txt'
        event_path = self.path + 'event/event.yaml'
        metric_path = self.path + f'metric/metric.yaml'

        if not os.path.exists(self.path + 'event/'):
            os.makedirs(self.path + 'event/')
        if not os.path.exists(self.path + 'metric/'):
            os.makedirs(self.path + 'metric/')
        if not os.path.exists(self.path + 'architecture/'):
            os.makedirs(self.path + 'architecture/')
        if not os.path.exists(self.path + 'workload/'):
            os.makedirs(self.path + 'workload/')

        with open(event_path, 'w') as f:
            safe_dump(
                self.event.events,
                f,
                sort_keys=False
            )

        with open(metric_path, 'w') as f:
            safe_dump(
                self.metric.metrics,
                f,
                sort_keys=False
            )

        for i, config in enumerate(self.configurations):
            arch_yaml = config['arch']
            work_yaml = config['work']

            arch_path = self.path + f'architecture/config_{i}.architecture.yaml'
            work_path = self.path + f'workload/config_{i}.workload.yaml'
            run_path = self.path + f'runs/config_{i}'
            checkpoint_path = self.path + f'runs/config_{i}/checkpoint.gt'

            with open(arch_path, 'w') as f:
                safe_dump(arch_yaml,
                        f,
                        sort_keys=False)
            
            with open(work_path, 'w') as f:
                safe_dump(work_yaml,
                        f,
                        sort_keys=False)
            
            runs.append({
                'arch_path': arch_path,
                'work_path': work_path,
                'event_path': event_path,
                'metric_path': metric_path,
                'run_path': run_path,
                'checkpoint_path': checkpoint_path
            })

        with open(runs_txt, 'w') as f:
            f.write('')

        for run in runs:
            with open(runs_txt, 'a') as f:
                run_line = f"-a {run['arch_path']}"
                run_line += f" -e {run['event_path']}"
                run_line += f" -m {run['metric_path']}"
                run_line += f" -r {run['run_path']}"
                run_line += f" -w {run['work_path']}"
                run_line += f" -c {run['checkpoint_path']}"
                run_line += ' -s\n'
                f.write(run_line)
            
