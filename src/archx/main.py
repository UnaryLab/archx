import pyfiglet, argparse, time

from loguru import logger

from archx.architecture import create_architecture_dict, save_architecture_dict
from archx.event import create_event_graph, save_event_graph
from archx.metric import create_metric_dict, create_event_metrics, save_metric_dict
from archx.workload import create_workload_dict, save_workload_dict
from archx.performance import simulate_performance_all_events
from archx.utils import bcolors, write_yaml, read_yaml

def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='Archx is a framework to explore the design space of hardware architecture.')
    parser.add_argument('-r', '--run_dir', type=str, default=None,
                        help = 'Run directory to store outputs.')
    parser.add_argument('-a', '--architecture_yaml', type=str, default=None,
                        help = 'Path to architecture yaml.')
    parser.add_argument('-m', '--metric_yaml', type=str, default=None,
                        help = 'Path to metric yaml.')
    parser.add_argument('-w', '--workload_yaml', type=str, default=None,
                        help = 'Path to workload yaml.')
    parser.add_argument('-e', '--event_yaml', type=str, default=None,
                        help = 'Path to event yaml.')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help = 'Path to checkpoint, which requires <.gt> format.')
    parser.add_argument('-l', '--log_level', type=str, default='INFO',
                        help = 'Level of logger.')
    parser.add_argument('-s', '--save_yaml', type=str, default='false', help = 'To save yaml files in run directory.')

    return parser.parse_args()


def main():
    args = parse_commandline_args()
    
    # set up output log
    logger.remove()
    output_log = args.run_dir + '/archx' + '-' + str(time.time()) + '.log'
    logger.add(output_log, level=args.log_level)

    # set up banner
    ascii_banner = pyfiglet.figlet_format('Archx')
    print(bcolors.Magenta + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format('UnaryLab')
    print(bcolors.Yellow + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format('https://github.com/UnaryLab/archx', font='term')
    print(bcolors.UNDERLINE + bcolors.Green + ascii_banner + bcolors.ENDC)

    # configure yaml save path
    save_yaml = args.save_yaml.lower()

    # validate checkpoint
    assert args.checkpoint.endswith('.gt'), logger.error('Invalid event checkpoint format; requires <.gt>.')

    logger.success(f'\n----------------------------------------------\nStep 1: Create architectue dict\n----------------------------------------------')
    architecture_dict = create_architecture_dict(args.architecture_yaml)

    logger.success(f'\n----------------------------------------------\nStep 2: Create metric dict\n----------------------------------------------')
    metric_dict = create_metric_dict(args.metric_yaml)

    logger.success(f'\n----------------------------------------------\nStep 3: Creat workload dict\n----------------------------------------------')
    workload_dict = create_workload_dict(args.workload_yaml)
    
    logger.success(f'\n----------------------------------------------\nStep 4: Creat event graph\n----------------------------------------------')
    event_graph = create_event_graph(args.event_yaml)

    logger.success(f'\n----------------------------------------------\nStep 5: Create metrics for all events and modules\n----------------------------------------------')
    event_graph = create_event_metrics(event_graph, architecture_dict, metric_dict, run_dir=args.run_dir)
    
    logger.success(f'\n----------------------------------------------\nStep 6: Simulate performance\n----------------------------------------------')
    event_graph = simulate_performance_all_events(event_graph, architecture_dict, workload_dict)
    
    logger.success(f'\n----------------------------------------------\nStep 7: Save event graph and log\n----------------------------------------------')
    save_event_graph(event_graph=event_graph, save_path=args.checkpoint)

    if save_yaml == 'true':
        save_architecture_dict(architecture_dict=architecture_dict, save_path=args.run_dir + '/architecture.yaml')
        save_metric_dict(metric_dict=metric_dict, save_path=args.run_dir + '/metric.yaml')
        save_workload_dict(workload_dict=workload_dict, save_path=args.run_dir + '/workload.yaml')
        write_yaml(args.run_dir + '/event.yaml', read_yaml(args.event_yaml))
        logger.success(f'Save dictionaries to <{args.run_dir}>.')

    logger.success(f'Save log to <{output_log}>.')

if __name__ == '__main__':
    main()

