from archx.utils import read_yaml, write_yaml
from collections import OrderedDict
import os, shutil, copy, itertools
from loguru import logger


def clean_dir(base_path: str):
    base_path = base_path if base_path.endswith('/') else base_path + '/'
    generated_path = f'{base_path}generated/'
    shutil.rmtree(generated_path, ignore_errors=True)

def generate_architecture(base_path: str):
    """
    Generate architecture yamls from template
    """

    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = base_path + 'template/'
    generated_path = base_path + 'generated/'

    architecture_file = template_path + 'architecture.yaml'
    configuration_file = template_path + 'configuration.yaml'

    architecture_dict = read_yaml(architecture_file)
    configuration_dict = read_yaml(configuration_file)

    network_config_dict = configuration_dict.get('network')
    tag_config_dict = configuration_dict.get('tag')
    array_config_dict = configuration_dict.get('array')

    subarchitecture_config_dict = configuration_dict.get('subarchitecture', [None])
    
    # Pass through each network, subarchitecture, and array configuration
    # - Network: single/multinode, node dimensions, ex. 4x4, 8x8
    # - subarchitecture: ex. mac, figna, etc.
    # - array: array dimensions, ex. mugi 128x8, 256x8, systolic 8x8, 16x16, etc.

    for network_key, network_value in network_config_dict.items():
        for network_dimension in network_value['dimension']:
            for subarchitecture in subarchitecture_config_dict:
                for array_dim, array_value in array_config_dict.items():
                    if network_dimension == []:
                        network_path = f'{network_key}/'
                    else:
                        network_path = f'{network_key}_{network_dimension[0]}x{network_dimension[1]}/'

                    # can be dictionary of instance, width, depth labels or path to yaml containing those labels
                    if 'path' in array_value:
                        array_value = read_yaml(array_value['path'])
                        
                    generated_architecture_dict = copy.deepcopy(architecture_dict)

                    if 'path' in network_value:
                        network_architecture_dict = read_yaml(network_value['path'])
                        generated_architecture_dict['architecture']['module'].update(network_architecture_dict['architecture']['module'])

                    if subarchitecture:
                        subarchitecture_dict = read_yaml(subarchitecture_config_dict[subarchitecture]['path'])
                        generated_architecture_dict['architecture']['module'].update(subarchitecture_dict['architecture']['module'])
                        subarchitecture_path = f'{subarchitecture}/'
                    else:
                        subarchitecture_path = '/'

                    generated_architecture_component_dict = OrderedDict({'architecture': {'module': {}}})
                    for component_key, component_value in generated_architecture_dict['architecture']['module'].items():
                        if 'path' in component_value:
                            component_value = read_yaml(component_value['path'])['architecture']['module']
                            generated_architecture_component_dict['architecture']['module'].update(component_value)
                        else:
                            generated_architecture_component_dict['architecture']['module'][component_key] = component_value

                    for component_key, component_value in generated_architecture_component_dict['architecture']['module'].items():
                        if isinstance(component_value.get('tag'), str):
                            component_value['tag'] = copy.deepcopy(tag_config_dict.get(component_value['tag'], None))

                        if isinstance(component_value.get('instance'), str):
                            component_value['instance'] = network_dimension + array_value['instance'].get(component_value['instance'], None)

                        if isinstance(component_value['query'].get('width'), str):
                            component_value['query']['width'] = array_value['width'].get(component_value['query']['width'], None)
                        if isinstance(component_value['query'].get('depth'), str):
                            component_value['query']['depth'] = array_value['depth'].get(component_value['query']['depth'], None)
                    generated_architecture_component_dict['architecture']['attribute'] = generated_architecture_dict['architecture']['attribute']

                    output_path = f'{generated_path}{subarchitecture_path}{network_path}{array_dim}/architecture.yaml'
                    write_yaml(file=output_path, content=generated_architecture_component_dict)

def generate_event(base_path: str):
    """
    Generate architecture yamls from template
    """

    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = base_path + 'template/'
    generated_path = base_path + 'generated/'

    configuration_dict = read_yaml(template_path + 'configuration.event.yaml')
    network_config_dict = configuration_dict.get('network')
    hardware_config_dict = configuration_dict.get('hardware')
    default_config_dict = configuration_dict.get('default')
    memory_config_dict = default_config_dict.get('memory')

    for network_key, network_value in network_config_dict.items():
        for subarchitecture_key, subarchitecture_value in hardware_config_dict.items():
            generated_event_dict = OrderedDict({'event': {}})

            for default_event_value in default_config_dict.values():
                if 'path' in default_event_value:
                    default_event_dict = read_yaml(default_event_value['path'])
                generated_event_dict['event'].update(default_event_dict['event'])
                
                if network_value:
                    if 'path' in network_value:
                        network_event_dict = read_yaml(network_value['path'])
                    generated_event_dict['event'].update(network_event_dict['event'])

                for hardware_key, hardware_value in subarchitecture_value.items():
                    if 'path' in hardware_value:
                        hardware_event_dict = read_yaml(hardware_value['path'])
                    generated_event_dict['event'].update(hardware_event_dict['event'])

                for hardware_key, hardware_value in subarchitecture_value.items():
                    if 'path' in hardware_value:
                        hardware_event_dict = read_yaml(hardware_value['path'])
                    for event_key, event_value in generated_event_dict['event'].items():
                        if not isinstance(event_value['subevent'], list) and event_value['subevent'] == hardware_key:
                            hardware_subevent_list = list(hardware_event_dict['event'].keys())
                            if 'path' in memory_config_dict:
                                memory_config_dict = read_yaml(memory_config_dict['path'])
                            memory_subevent_list = list(memory_config_dict['event'].keys())
                            network_subevent_list = list(network_event_dict['event'].keys()) if 'network_event_dict' in locals() else []
                            event_value['subevent'] = hardware_subevent_list + memory_subevent_list + network_subevent_list

                generated_event_path = f'{generated_path}{network_key}/{subarchitecture_key}/event.yaml'
                write_yaml(generated_event_path, generated_event_dict)

def generate_workload(base_path: str):
    """
    Generate architecture yamls from template
    """

    base_path = base_path if base_path.endswith('/') else base_path + '/'
    template_path = base_path + 'template/'
    generated_path = base_path + 'generated/'

    workload_dict = read_yaml(template_path + 'workload.yaml')
    configuration_dict = read_yaml(template_path + 'configuration.yaml')
    config_dict = configuration_dict.get('configuration')
    architecture_dict = configuration_dict.get('architecture', {'workload': None})

    for workload_key, workload_value in workload_dict['workload'].items():
        for architecture_key, architecture_value in architecture_dict.items():
            if architecture_key == 'workload':
                architecture_key = ''
            config_combinations = [dict(zip(config_dict.keys(), combination)) for combination in itertools.product(*config_dict.values())]
            for combination in config_combinations:
                generated_workload_dict = OrderedDict({'workload': {workload_key: {'configuration': {}}}})
                generated_workload_dict['workload'][workload_key]['configuration'].update(workload_value['configuration'])
                if architecture_value:
                    generated_workload_dict['workload'][workload_key]['configuration'].update(architecture_value)
                generated_workload_dict['workload'][workload_key]['configuration'].update(combination)

                workload_config_path = ''
                for key, value in combination.items():
                    if len(config_dict[key]) > 1:
                        workload_config_path += '/' + key + '_' + str(value)

                workload_path = os.path.normpath(f'{generated_path}{workload_key}/{architecture_key}/{workload_config_path}/workload.yaml')
                write_yaml(workload_path, generated_workload_dict)

def main():
    logger.remove()
    base_dir = 'zoo/llm/'
    architecture_list = read_yaml(base_dir + 'scripts/architecture_configuration.yaml')
    for arch in architecture_list['architecture']:
        base_path = os.path.join(base_dir, arch)
        if os.path.isdir(base_path):
            architecture_path = os.path.join(base_path, 'architecture/')
            event_path = os.path.join(base_path, 'event/')
            workload_path = os.path.join(base_path, 'workload/')

            clean_dir(architecture_path)
            clean_dir(event_path)
            clean_dir(workload_path)

            generate_architecture(architecture_path)
            generate_event(event_path)
            generate_workload(workload_path)

if __name__ == "__main__":
    main()