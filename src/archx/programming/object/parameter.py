class Parameter:
    def __init__(self, graph):
        self.graph = graph

    def _add_node(self, name: str, param_name: str, value, type: str, sweep: bool, desc: str):
        vertex = self.graph.add_vertex()
        self.graph.vp.name[vertex] = name
        self.graph.vp.param_name[vertex] = param_name
        self.graph.vp.value[vertex] = value
        self.graph.vp.type[vertex] = type
        self.graph.vp.sweep[vertex] = sweep
        self.graph.vp.desc[vertex] = desc
        return vertex