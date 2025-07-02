# following two lines are used in testing
import sys, os, shutil

from loguru import logger

from archx.interface import query_interface, register_interface, unregister_interface, copy_interface
from archx.utils import get_path, create_dir


def test_query_interface_csv_cmos():
    module = 'ireg'
    query = {
        'class': 'multiplier_reg',
        'interface': 'csv_cmos',
        'technology': 32,
        'frequency': 400,
        'bitwidth': 32
    }
    output = query_interface(module, query)
    logger.info('test_query_interface_csv_cmos: ', output)


def test_query_interface_cacti7_sram():
    module = 'isram'
    query = {
        'class': 'sram',
        'interface': 'cacti7',
        'technology': 32,
        'frequency': 400,
        'width': 16,
        'depth': 512,
        'bank': 32
    }
    path = get_path('tests')
    path = path + '/test_interface/'
    create_dir(path)
    output = query_interface(module, query, output_dir=path)
    logger.info('test_query_interface_cacti7_sram: ', output)


def test_query_interface_cacti7_dram():
    module = 'ocdram'
    query = {
        'class': 'dram',
        'interface': 'cacti7',
        'technology': 32,
        'frequency': 400,
        'bandwidth': 25.6,
        'size': 1073741824 # 1GB in bytes
    }
    path = get_path('tests')
    path = path + '/test_interface/'
    create_dir(path)
    output = query_interface(module, query, output_dir=path)
    logger.info('test_query_interface_cacti7_dram: ', output)


def test_copy_interface():
    name = 'csv_cmos'
    path = 'tests/test_interface/dummy_csv_cmos'
    copy_interface(name, path)


def test_register_interface():
    name = 'dummy_csv_cmos'
    path = 'tests/test_interface/dummy_csv_cmos'
    register_interface(name, path)


def test_unregister_interface():
    name = 'dummy_csv_cmos'
    unregister_interface(name)


def test_cleanup():
    path = 'tests/test_interface/dummy_csv_cmos'
    shutil.rmtree(path)
    path = get_path('tests/test_interface/')
    shutil.rmtree(path)


if __name__ == "__main__":
    test_query_interface_csv_cmos()
    test_query_interface_cacti7_sram()
    test_query_interface_cacti7_dram()
    test_copy_interface()
    test_register_interface()
    test_unregister_interface()
    test_cleanup()

