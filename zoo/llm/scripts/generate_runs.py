from archx.utils import read_yaml
import itertools, os

def main():
    base_path = 'zoo/llm/'
    scripts_path = base_path + 'scripts/'
    architecture_list = read_yaml(scripts_path + 'architecture_configuration.yaml')

    with open(scripts_path + 'runs.txt', 'w') as file:
        for arch in architecture_list['architecture']:
            arch_path = arch + '/'
            architecture_config_dict = read_yaml(base_path + arch_path + 'architecture/template/configuration.yaml')
            network_dict = architecture_config_dict['network']
            array_dict = architecture_config_dict['array']
            subarch_dict = architecture_config_dict.get('subarchitecture', {'': {}})
            for network_key, network_value in network_dict.items():
                network_path = network_key + '/'
                for network_dim in network_value['dimension']:
                    network_dim_path = network_key + '_' + str(network_dim[0]) + 'x' + str(network_dim[1]) + '/' if len(network_dim) == 2 else network_key + '/'
                    for array_key in array_dict.keys():
                        array_path = array_key + '/'
                        for subarch_key in subarch_dict.keys():
                            subarch_path = subarch_key + '/'
                            workload_config_dict = read_yaml(base_path + arch_path + 'workload/template/configuration.yaml')
                            workload_dict = read_yaml(base_path + arch_path + 'workload/template/workload.yaml')
                            config_combinations = [dict(zip(workload_config_dict['configuration'].keys(), combination)) for combination in itertools.product(*workload_config_dict['configuration'].values())]
                            for model in workload_dict['workload'].keys():
                                model_path = model + '/'
                                for config_combination in config_combinations:
                                    workload_config_path = ''
                                    for key, value in config_combination.items():
                                        if len(workload_config_dict['configuration'][key]) > 1:
                                            workload_config_path += '/' + key + '_' + str(value)
                                    workload_config_path += '/'

                                    for termination in ['full_termination', 'early_termination']:
                                        if arch != 'mugi' and termination == 'early_termination':
                                            continue
                                        elif arch == 'mugi' and termination == 'full_termination':
                                            workload_termination_path = 'full_termination/'
                                            runs_termination_path = 'full_termination/'
                                        elif arch == 'mugi' and termination == 'early_termination':
                                            workload_termination_path = 'early_termination_' + array_path
                                            runs_termination_path = 'early_termination/'
                                        else:
                                            workload_termination_path = ''
                                            runs_termination_path = ''
                                    
                                        if network_key == 'multi_node' and subarch_key in ['pwl', 'taylor']:
                                            continue
                                        if network_key == 'multi_node' and config_combination['batch_size'] !=8:
                                            continue
                                        if network_key == 'multi_node' and config_combination['max_seq_len'] != 4096:
                                            continue
                                        if network_key == 'multi_node' and array_key in ['32x32', '64x64']:
                                            continue
                                        if array_key in ['32x8', '4x4']:
                                            continue
                                        if array_key != '16x16' and subarch_key in ['pwl', 'taylor']:
                                            continue

                                        event_subarch = 'vlp/' if subarch_path == '/' else subarch_path

                                        architecture_path = os.path.normpath('./' + base_path + arch_path + '/architecture/generated/' + subarch_path + network_dim_path + array_path + 'architecture.yaml')
                                        event_path = os.path.normpath('./' + base_path + arch_path + 'event/generated/' + network_path + event_subarch + 'event.yaml')
                                        metric_path = os.path.normpath('./' + base_path + 'metric/metric.yaml')
                                        runs_path = os.path.normpath('./' + base_path + 'runs/' + arch_path + network_dim_path + subarch_path + array_path + model_path + workload_config_path + runs_termination_path)
                                        workload_path = os.path.normpath('./' + base_path + arch_path + 'workload/generated/' + model_path + workload_termination_path + workload_config_path + 'workload.yaml')
                                        checkpoint_path = os.path.normpath(runs_path + '/checkpoint.gt/')
                                        save_arg = 'true'

                                        file.write('-a ' + architecture_path + ' -e ' + event_path + ' -m ' + metric_path + ' -r ' + runs_path +  ' -w ' + workload_path  + ' -c ' + checkpoint_path + ' -s ' + save_arg + '\n')
    
if '__main__' == __name__:
    main()