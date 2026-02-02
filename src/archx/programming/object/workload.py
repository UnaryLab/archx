from archx.programming.object.parameter import Parameter
from loguru import logger

class Workload(Parameter):
    def __init__(self, graph):
        super().__init__(graph)
        self.configurations = dict()

    def add_configuration(self, name: str):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert name not in self.configurations, f"Configuration '{name}' already exists in workload."

        self.configurations[name] = []
        logger.info(f"Added configuration '{name}'.")

    def add_parameter(self, configuration: str, parameter_name: str, parameter_value, sweep: bool = None):
        assert isinstance(configuration, str), "'configuration' parameter must be of type 'str'."
        assert configuration in self.configurations, f"Configuration '{configuration}' not found in workload. Maybe you meant to call 'add_configuration()'."
        assert isinstance(parameter_name, str), "'parameter_name' parameter must be of type 'str'."
        assert isinstance(sweep, (bool, type(None))), "'sweep' parameter must be of type 'bool'."
        assert (isinstance(parameter_value, list) and isinstance(sweep, bool)) or (not isinstance(parameter_value, list)) , "If 'parameter_value' is a list, 'sweep' must be defined as either True or False."

        sweep = sweep if sweep is not None else False

        parameter_vertex = self._add_node(name=configuration,
                                          param_name=parameter_name,
                                          value=parameter_value,
                                          type='parameter',
                                          sweep=sweep,
                                          desc='workload')
        
        self.configurations[configuration].append(parameter_vertex)

        logger.info(f"Added parameter '{parameter_name}' to configuration '{configuration}'.")
        logger.debug(f"\tValue: {parameter_value}")
        logger.debug(f"\tSweep: {sweep}")