import importlib
from collections import OrderedDict

from archx.performance import simulate_performance_one_event


class DummyEventGraph:
    def __init__(self, performance_path):
        self.performance_path = performance_path

    def has_node(self, event_name):
        return event_name == 'kernel'

    def get_performance_path(self, event_name):
        return self.performance_path

    def is_leaf(self, event_name):
        return True

    def get_out_neighbors(self, event_name):
        return []


def test_performance_cache_uses_traced_dependencies(monkeypatch, tmp_path):
    performance_module = importlib.import_module('archx.performance.performance')
    performance_module._performance_output_cache.clear()
    performance_module._performance_dependency_cache.clear()
    monkeypatch.setenv('ARCHX_PERFORMANCE_CACHE_DIR', str(tmp_path / 'cache'))

    count_file = tmp_path / 'calls.txt'
    count_file.write_text('0')
    monkeypatch.setenv('ARCHX_TEST_COUNT_FILE', str(count_file))

    performance_file = tmp_path / 'kernel.performance.py'
    performance_file.write_text(
        "from collections import OrderedDict\n"
        "import os\n"
        "\n"
        "def kernel(architecture_dict, workload_dict=None):\n"
        "    count_file = os.environ['ARCHX_TEST_COUNT_FILE']\n"
        "    calls = int(open(count_file).read()) + 1\n"
        "    open(count_file, 'w').write(str(calls))\n"
        "    _ = architecture_dict['used']['value'] * workload_dict['configuration']['m']\n"
        "    return OrderedDict({'subevent': OrderedDict()})\n"
    )

    graph = DummyEventGraph(str(performance_file))
    first_arch = OrderedDict({
        'used': OrderedDict({'value': 4}),
        'irrelevant': OrderedDict({'value': 1}),
    })
    second_arch = OrderedDict({
        'used': OrderedDict({'value': 4}),
        'irrelevant': OrderedDict({'value': 999}),
    })
    workload = OrderedDict({'configuration': OrderedDict({'m': 8, 'unused': 1})})

    simulate_performance_one_event(graph, first_arch, workload, 'kernel')
    simulate_performance_one_event(graph, second_arch, workload, 'kernel')

    assert count_file.read_text() == '1'

    changed_arch = OrderedDict({
        'used': OrderedDict({'value': 5}),
        'irrelevant': OrderedDict({'value': 999}),
    })
    simulate_performance_one_event(graph, changed_arch, workload, 'kernel')

    assert count_file.read_text() == '2'

    performance_module._performance_output_cache.clear()
    performance_module._performance_dependency_cache.clear()
