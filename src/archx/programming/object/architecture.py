from archx.programming.object.parameter import Parameter

# TODO:
#   - Finalize method signitures
#   - Finalize assertion messages
#   - Allow sweeping objects to be passed.
#   - Allow add_module to pass query as **kwargs

class Architecture(Parameter):
    def __init__(self, graph):
        super().__init__(graph)
        self.attributes = []
        self.modules = []
        self.tags = []

    def add_attributes(self, **kwargs):
        assert kwargs, "At least one attribute must be provided."

        for key, value in kwargs.items():
            sweep = isinstance(value, list)
            attr_vertex = self._add_node(name='attribute',
                                         param_name=key,
                                         value=value,
                                         type='attribute',
                                         sweep=sweep,
                                         desc='architecture')
            self.attributes.append(attr_vertex)

    def add_module(self, name, instance, tag, query):
        assert isinstance(name, str) or isinstance(name, list), "Module name must be a string or list of strings."
        assert isinstance(instance, list), "Module instance must be a list."
        assert isinstance(tag, list), "Module tag must be a list."
        assert isinstance(query, dict), "Module query must be a dictionary."

        if isinstance(name, list):
            for module_name in name:
                self._module(module_name, instance, tag, query)
        else:
            self._module(name, instance, tag, query)

    def _module(self, name, instance, tag, query):
        assert isinstance(name, str), "Module name must be a string."

        # check if instance is a sweep (list of lists) or not
        inst_sweep = None
        for inst in instance:
            sweep_check = isinstance(inst, list)
            if inst_sweep is None:
                inst_sweep = sweep_check
            else:
                assert inst_sweep == sweep_check, "All instance entries must be of the same type (sweep or non-sweep)."
                
        inst_vertex = self._add_node(name=name,
                                     param_name='instance',
                                     value=instance,
                                     type='module',
                                     sweep=inst_sweep,
                                     desc='architecture')
        
        self.modules.append(inst_vertex)
        
        tag_sweep = None
        tag_vertex = self._add_node(name=name,
                                    param_name='tag',
                                    value=tag,
                                    type='module',
                                    sweep=tag_sweep,
                                    desc='architecture')

        self.tags.append(tag_vertex)
        
        for key, value in query.items():
            sweep = isinstance(value, list)
            query_vertex = self._add_node(name=name, 
                                          param_name=key,
                                          value=value,
                                          type='module',
                                          sweep=sweep,
                                          desc='architecture')
            
            self.modules.append(query_vertex)