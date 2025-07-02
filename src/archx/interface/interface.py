import shutil, os, sys, copy
import importlib.util

from collections import OrderedDict
from loguru import logger

from archx.utils import get_path


key_interface = 'interface'


def query_interface(module: str, query: OrderedDict, input_dir=None, output_dir=None) -> OrderedDict:
    """
    query is a dictionary with query configurations
    """
    query = copy.deepcopy(query)
    assert key_interface in query, logger.error(f'Invalid query: <{query}>. Must contain <{key_interface}>. Possible undefined attribute in archtitecture dictionary.')
    q_interface = query[key_interface]
    dst_file = os.path.join(os.path.dirname(__file__), q_interface, q_interface + '.py')

    # find proper query interface code
    spec = importlib.util.spec_from_file_location('query_' + str(module) + '_' + str(q_interface), dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)
    
    # actual query
    del query[key_interface]
    query_result = module_py.query(module, q_interface, query, input_dir, output_dir)

    return query_result
    

def register_interface(name: str, interface_dir: str) -> None:
    assert name != key_interface, logger.error(f'Invalid interface name: <{name}>.')
    archx.dir = get_path(interface_dir)
    dst_dir = os.path.join(os.path.dirname(__file__), name)
    if os.path.isdir(dst_dir):
        logger.warning(f'Interface <{name}> exists at <{dst_dir}>.')
    else:
        shutil.copytree(archx.dir, dst_dir)
        logger.success(f'Register interface <{name}> from <{archx.dir}> to <{dst_dir}>.')


def unregister_interface(name: str) -> None:
    assert name != key_interface, logger.error(f'Invalid interface name: <{name}>.')
    dst_dir = os.path.join(os.path.dirname(__file__), name)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
        logger.success(f'Unregister interface <{name}> to <{dst_dir}>.')
    else:
        logger.warning(f'Interface <{name}> does not exist at <{dst_dir}>.')


def copy_interface(name: str, interface_dir: str) -> None:
    assert name != key_interface, logger.error(f'Invalid interface name: <{name}>.')
    archx.dir = os.path.join(os.path.dirname(__file__), name)
    dst_dir = os.path.join(os.getcwd(), interface_dir)
    if os.path.isdir(archx.dir) and not os.path.isdir(dst_dir):
        shutil.copytree(archx.dir, dst_dir)
        logger.success(f'Copy interface: <{name}> from <{archx.dir}> to: <{dst_dir}>.')
    elif not os.path.isdir(archx.dir):
        logger.warning(f'Interface <{name}> does not exist at <{archx.dir}>.')
    elif os.path.isdir(archx.dir):
        logger.warning(f'Interface <{name}> exists at <{dst_dir}>.')

