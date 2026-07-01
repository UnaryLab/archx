import shutil, os, sys, copy
import importlib.util

from collections import OrderedDict
from loguru import logger

from archx.utils import get_path


key_interface = 'interface'
_interface_module_cache = {}
_interface_output_cache = {}


def _freeze_cache_value(value):
    if isinstance(value, dict):
        return tuple((k, _freeze_cache_value(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_cache_value(v) for v in value)
    return value


def _interface_output_cache_key(module: str, interface: str, query: OrderedDict, input_dir=None):
    if input_dir is not None:
        input_dir = os.path.realpath(input_dir)
    return module, interface, _freeze_cache_value(query), input_dir


def _load_interface_module(interface: str, dst_file: str):
    cache_key = os.path.realpath(dst_file)
    if cache_key not in _interface_module_cache:
        spec = importlib.util.spec_from_file_location(
            'query_' + str(interface),
            dst_file,
        )
        module_py = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module_py
        spec.loader.exec_module(module_py)
        _interface_module_cache[cache_key] = module_py
    return _interface_module_cache[cache_key]


def query_interface(module: str, query: OrderedDict, input_dir=None, output_dir=None) -> OrderedDict:
    """
    query is a dictionary with query configurations
    """
    query = copy.deepcopy(query)
    assert key_interface in query, logger.error(f'Invalid query: <{query}>. Must contain <{key_interface}>. Possible undefined attribute in archtitecture dictionary.')
    q_interface = query[key_interface]
    dst_file = os.path.join(os.path.dirname(__file__), q_interface, q_interface + '.py')

    # find proper query interface code
    module_py = _load_interface_module(q_interface, dst_file)
    
    # actual query
    del query[key_interface]
    cache_key = _interface_output_cache_key(module, q_interface, query, input_dir)
    if cache_key in _interface_output_cache:
        return copy.deepcopy(_interface_output_cache[cache_key])

    query_result = module_py.query(module, q_interface, query, input_dir, output_dir)
    _interface_output_cache[cache_key] = copy.deepcopy(query_result)

    return copy.deepcopy(query_result)
    

def register_interface(name: str, interface_dir: str) -> None:
    assert name != key_interface, logger.error(f'Invalid interface name: <{name}>.')
    src_dir = get_path(interface_dir)
    dst_dir = os.path.join(os.path.dirname(__file__), name)
    if os.path.isdir(dst_dir):
        logger.warning(f'Interface <{name}> exists at <{dst_dir}>.')
    else:
        shutil.copytree(src_dir, dst_dir)
        _interface_module_cache.pop(os.path.realpath(os.path.join(dst_dir, name + '.py')), None)
        _interface_output_cache.clear()
        logger.success(f'Register interface <{name}> from <{src_dir}> to <{dst_dir}>.')


def unregister_interface(name: str) -> None:
    assert name != key_interface, logger.error(f'Invalid interface name: <{name}>.')
    dst_dir = os.path.join(os.path.dirname(__file__), name)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
        _interface_module_cache.pop(os.path.realpath(os.path.join(dst_dir, name + '.py')), None)
        _interface_output_cache.clear()
        logger.success(f'Unregister interface <{name}> to <{dst_dir}>.')
    else:
        logger.warning(f'Interface <{name}> does not exist at <{dst_dir}>.')


def copy_interface(name: str, interface_dir: str) -> None:
    assert name != key_interface, logger.error(f'Invalid interface name: <{name}>.')
    src_dir = os.path.join(os.path.dirname(__file__), name)
    dst_dir = os.path.join(os.getcwd(), interface_dir)
    if os.path.isdir(src_dir) and not os.path.isdir(dst_dir):
        shutil.copytree(src_dir, dst_dir)
        logger.success(f'Copy interface: <{name}> from <{src_dir}> to: <{dst_dir}>.')
    elif not os.path.isdir(src_dir):
        logger.warning(f'Interface <{name}> does not exist at <{src_dir}>.')
    elif os.path.isdir(src_dir):
        logger.warning(f'Interface <{name}> exists at <{dst_dir}>.')
