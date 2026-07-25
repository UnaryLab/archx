# following two lines are used in testing
import shutil
import importlib

from loguru import logger

from archx.interface import query_interface, register_interface, unregister_interface, copy_interface
from archx.interface.cacti7.cacti7 import cacti7_run
from archx.utils import get_path, create_dir


def test_query_interface_csv_cmos():
    module = 'ireg'
    query = {
        'class': 'multiplier',
        'interface': 'csv_cmos',
        'technology': 45,
        'frequency': 400,
        'width': 32
    }
    output = query_interface(module, query)
    logger.info('test_query_interface_csv_cmos: ', output)


def test_query_interface_output_cache(monkeypatch, tmp_path):
    interface_module = importlib.import_module('archx.interface.interface')
    interface_module._interface_output_cache.clear()
    monkeypatch.setenv('ARCHX_INTERFACE_CACHE_DIR', str(tmp_path))

    class DummyInterface:
        calls = 0

        @staticmethod
        def query(name, interface, query, input_dir=None, output_dir=None):
            DummyInterface.calls += 1
            return {'area': {'value': DummyInterface.calls, 'unit': 'mm^2'}}

    monkeypatch.setattr(
        interface_module,
        '_load_interface_module',
        lambda interface, dst_file: DummyInterface,
    )

    query = {'class': 'mac', 'interface': 'dummy_cache', 'width': 32}
    first = query_interface('mac0', query, output_dir='/tmp/run0')
    first['area']['value'] = 999

    second = query_interface('mac1', query, output_dir='/tmp/run1')

    assert DummyInterface.calls == 1
    assert second == {'area': {'value': 1, 'unit': 'mm^2'}}

    interface_module._interface_output_cache.clear()


def test_query_interface_persistent_output_cache(monkeypatch, tmp_path):
    interface_module = importlib.import_module('archx.interface.interface')
    interface_module._interface_output_cache.clear()
    monkeypatch.setenv('ARCHX_INTERFACE_CACHE_DIR', str(tmp_path))

    class DummyInterface:
        calls = 0

        @staticmethod
        def query(name, interface, query, input_dir=None, output_dir=None):
            DummyInterface.calls += 1
            return {'area': {'value': DummyInterface.calls, 'unit': 'mm^2'}}

    monkeypatch.setattr(
        interface_module,
        '_load_interface_module',
        lambda interface, dst_file: DummyInterface,
    )

    query = {'class': 'mac', 'interface': 'dummy_persistent_cache', 'width': 32}
    first = query_interface('mac0', query, output_dir='/tmp/run0')
    interface_module._interface_output_cache.clear()
    second = query_interface('mac1', query, output_dir='/tmp/run1')

    assert DummyInterface.calls == 1
    assert first == second == {'area': {'value': 1, 'unit': 'mm^2'}}

    interface_module._interface_output_cache.clear()


def test_cacti7_run_uses_platform_binary_directly(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, cwd=None, stdout=None, check=None):
        calls.append({'cmd': cmd, 'cwd': cwd, 'check': check})
        stdout.write('fake cacti report')

    def fail_call(*args, **kwargs):
        raise AssertionError('cacti7_run should not use shell copy/remove commands')

    monkeypatch.setattr('archx.interface.cacti7.cacti7.platform.system', lambda: 'Linux')
    monkeypatch.setattr('archx.interface.cacti7.cacti7.platform.machine', lambda: 'x86_64')
    monkeypatch.setattr('archx.interface.cacti7.cacti7.subprocess.run', fake_run)
    monkeypatch.setattr('archx.interface.cacti7.cacti7.subprocess.call', fail_call)

    origin_cfg = tmp_path / 'origin.cfg'
    target_cfg = tmp_path / 'target.cfg'
    result_file = tmp_path / 'result.rpt'
    origin_cfg.write_text('-dummy option 1\n')

    cacti7_run(
        'sram',
        45,
        {'width': 16, 'depth': 512, 'bank': 32},
        origin_cfg_file=str(origin_cfg),
        target_cfg_file=str(target_cfg),
        result_file=str(result_file),
    )

    assert len(calls) == 1
    assert calls[0]['cmd'][0] == './cacti-Linux-x86_64'
    assert calls[0]['cmd'][1:] == ['-infile', str(target_cfg)]
    assert calls[0]['check'] is True
    assert result_file.read_text() == 'fake cacti report'


def test_query_interface_cacti7_sram():
    module = 'isram'
    query = {
        'class': 'sram',
        'interface': 'cacti7',
        'technology': 45,
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
        'technology': 45,
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
