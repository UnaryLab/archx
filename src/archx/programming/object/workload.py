from loguru import logger

class Workload():
    def __init__(self, parameter_enumerator):
        self.parameter_enumerator = parameter_enumerator
        self.configurations = []

    def add_configuration(self, name: str):
        assert isinstance(name, str), "'name' parameter must be of type 'str'."
        assert all(config.name != name for config in self.configurations), f"Configuration '{name}' already exists in workload."

        configuration = Configuration(name, self.parameter_enumerator)
        self.configurations.append(configuration)
        logger.info(f"Added configuration '{name}'.")
        return configuration
    
    def to_yaml(self, config):
        workload_dict = {'workload': {}}

        for var_name, value in config.items():
            param_info = self.parameter_enumerator.get_parameters_from_name(var_name)
            config_name = param_info['name']
            param_name = param_info['param_name']
            
            if config_name not in workload_dict['workload']:
                workload_dict['workload'][config_name] = {'configuration': {}}
            workload_dict['workload'][config_name]['configuration'][param_name] = value
        
        return workload_dict
            

class Configuration():
    def __init__(self, name: str, parameter_enumerator):
        self.name = name
        self.parameter_enumerator = parameter_enumerator

    def add_parameter(self, parameter_name: str, parameter_value, sweep: bool = None):
        assert isinstance(parameter_name, str), "'parameter_name' parameter must be of type 'str'."
        assert isinstance(sweep, (bool, type(None))), "'sweep' parameter must be of type 'bool'."
        assert (isinstance(parameter_value, list) and isinstance(sweep, bool)) or (not isinstance(parameter_value, list)) , "If 'parameter_value' is a list, 'sweep' must be defined as either True or False."

        sweep = sweep if sweep is not None else False

        parameter_param = self.parameter_enumerator.add_parameter(name=self.name,
                                                                   param_name=parameter_name,
                                                                   value=parameter_value,
                                                                   type='parameter',
                                                                   sweep=sweep,
                                                                   desc='workload')

        param_dict = {
            parameter_name: parameter_param
        }

        logger.info(f"Added parameter '{parameter_name}' to configuration '{self.name}'.")
        logger.debug(f"\tValue: {parameter_value}")
        logger.debug(f"\tSweep: {sweep}")

        return param_dict