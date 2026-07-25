from archx.programming.graph.agraph import AGraph

def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    ##############################################
    ################ Architecture ################
    ##############################################
    architecture.add_attributes(interface='csv_h200')

    k0001 = architecture.add_module(name='k0001', instance=[1], tag=['kernel'], query={'class': 'k0001_gpt2', 'n_kernels': 12})
    k0002 = architecture.add_module(name='k0002', instance=[1], tag=['kernel'], query={'class': 'k0002_gpt2', 'n_kernels': 12})
    k0003 = architecture.add_module(name='k0003', instance=[1], tag=['kernel'], query={'class': 'k0003_gpt2', 'n_kernels': 12})
    k0004 = architecture.add_module(name='k0004', instance=[1], tag=['kernel'], query={'class': 'k0004_gpt2', 'n_kernels': 12})
    k0005 = architecture.add_module(name='k0005', instance=[1], tag=['kernel'], query={'class': 'k0005_gpt2', 'n_kernels': 48})
    k0006 = architecture.add_module(name='k0006', instance=[1], tag=['kernel'], query={'class': 'k0006_gpt2', 'n_kernels': 24})
    k0007 = architecture.add_module(name='k0007', instance=[1], tag=['kernel'], query={'class': 'k0007_gpt2', 'n_kernels': 12})
    k0008 = architecture.add_module(name='k0008', instance=[1], tag=['kernel'], query={'class': 'k0008_gpt2', 'n_kernels': 12})
    k0009 = architecture.add_module(name='k0009', instance=[1], tag=['kernel'], query={'class': 'k0009_gpt2', 'n_kernels': 12})
    k0010 = architecture.add_module(name='k0010', instance=[1], tag=['kernel'], query={'class': 'k0010_gpt2', 'n_kernels': 12})
    k0011 = architecture.add_module(name='k0011', instance=[1], tag=['kernel'], query={'class': 'k0011_gpt2', 'n_kernels': 12})
    k0012 = architecture.add_module(name='k0012', instance=[1], tag=['kernel'], query={'class': 'k0012_gpt2', 'n_kernels': 25})
    k0013 = architecture.add_module(name='k0013', instance=[1], tag=['kernel'], query={'class': 'k0013_gpt2', 'n_kernels': 12})
    k0014 = architecture.add_module(name='k0014', instance=[1], tag=['kernel'], query={'class': 'k0014_gpt2', 'n_kernels': 12})
    k0015 = architecture.add_module(name='k0015', instance=[1], tag=['kernel'], query={'class': 'k0015_gpt2', 'n_kernels': 12})
    k0016 = architecture.add_module(name='k0016', instance=[1], tag=['kernel'], query={'class': 'k0016_gpt2', 'n_kernels': 12})
    k0017 = architecture.add_module(name='k0017', instance=[1], tag=['kernel'], query={'class': 'k0017_gpt2', 'n_kernels': 25})
    k0018 = architecture.add_module(name='k0018', instance=[1], tag=['kernel'], query={'class': 'k0018_gpt2', 'n_kernels': 12})
    k0019 = architecture.add_module(name='k0019', instance=[1], tag=['kernel'], query={'class': 'k0019_gpt2', 'n_kernels': 12})
    k0020 = architecture.add_module(name='k0020', instance=[1], tag=['kernel'], query={'class': 'k0020_gpt2', 'n_kernels': 12})
    k0021 = architecture.add_module(name='k0021', instance=[1], tag=['kernel'], query={'class': 'k0021_gpt2', 'n_kernels': 1})
    k0022 = architecture.add_module(name='k0022', instance=[1], tag=['kernel'], query={'class': 'k0022_gpt2', 'n_kernels': 1})
    k0023 = architecture.add_module(name='k0023', instance=[1], tag=['kernel'], query={'class': 'k0023_gpt2', 'n_kernels': 1})
    k0024 = architecture.add_module(name='k0024', instance=[1], tag=['kernel'], query={'class': 'k0024_gpt2', 'n_kernels': 27})
    k0025 = architecture.add_module(name='k0025', instance=[1], tag=['kernel'], query={'class': 'k0025_gpt2', 'n_kernels': 1})
    k0026 = architecture.add_module(name='k0026', instance=[1], tag=['kernel'], query={'class': 'k0026_gpt2', 'n_kernels': 1})
    k0027 = architecture.add_module(name='k0027', instance=[1], tag=['kernel'], query={'class': 'k0027_gpt2', 'n_kernels': 1})
    k0028 = architecture.add_module(name='k0028', instance=[1], tag=['kernel'], query={'class': 'k0028_gpt2', 'n_kernels': 1})
    k0029 = architecture.add_module(name='k0029', instance=[1], tag=['kernel'], query={'class': 'k0029_gpt2', 'n_kernels': 1})
    k0030 = architecture.add_module(name='k0030', instance=[1], tag=['kernel'], query={'class': 'k0030_gpt2', 'n_kernels': 1})
    k0031 = architecture.add_module(name='k0031', instance=[1], tag=['kernel'], query={'class': 'k0031_gpt2', 'n_kernels': 1})
    k0032 = architecture.add_module(name='k0032', instance=[1], tag=['kernel'], query={'class': 'k0032_gpt2', 'n_kernels': 1})
    k0033 = architecture.add_module(name='k0033', instance=[1], tag=['kernel'], query={'class': 'k0033_gpt2', 'n_kernels': 3})
    k0034 = architecture.add_module(name='k0034', instance=[1], tag=['kernel'], query={'class': 'k0034_gpt2', 'n_kernels': 1})
    k0035 = architecture.add_module(name='k0035', instance=[1], tag=['kernel'], query={'class': 'k0035_gpt2', 'n_kernels': 2})
    k0036 = architecture.add_module(name='k0036', instance=[1], tag=['kernel'], query={'class': 'k0036_gpt2', 'n_kernels': 1})
    k0037 = architecture.add_module(name='k0037', instance=[1], tag=['kernel'], query={'class': 'k0037_gpt2', 'n_kernels': 2})
    k0038 = architecture.add_module(name='k0038', instance=[1], tag=['kernel'], query={'class': 'k0038_gpt2', 'n_kernels': 4})
    k0040 = architecture.add_module(name='k0040', instance=[1], tag=['kernel'], query={'class': 'k0040_gpt2', 'n_kernels': 1})
    k0041 = architecture.add_module(name='k0041', instance=[1], tag=['kernel'], query={'class': 'k0041_gpt2', 'n_kernels': 1})
    k0042 = architecture.add_module(name='k0042', instance=[1], tag=['kernel'], query={'class': 'k0042_gpt2', 'n_kernels': 2})
    k0043 = architecture.add_module(name='k0043', instance=[1], tag=['kernel'], query={'class': 'k0043_gpt2', 'n_kernels': 2})
    k0044 = architecture.add_module(name='k0044', instance=[1], tag=['kernel'], query={'class': 'k0044_gpt2', 'n_kernels': 1})
    k0045 = architecture.add_module(name='k0045', instance=[1], tag=['kernel'], query={'class': 'k0045_gpt2', 'n_kernels': 1})
    k0046 = architecture.add_module(name='k0046', instance=[1], tag=['kernel'], query={'class': 'k0046_gpt2', 'n_kernels': 2})
    k0047 = architecture.add_module(name='k0047', instance=[1], tag=['kernel'], query={'class': 'k0047_gpt2', 'n_kernels': 2})
    k0048 = architecture.add_module(name='k0048', instance=[1], tag=['kernel'], query={'class': 'k0048_gpt2', 'n_kernels': 1})
    k0049 = architecture.add_module(name='k0049', instance=[1], tag=['kernel'], query={'class': 'k0049_gpt2', 'n_kernels': 1})
    k0050 = architecture.add_module(name='k0050', instance=[1], tag=['kernel'], query={'class': 'k0050_gpt2', 'n_kernels': 1})
    k0051 = architecture.add_module(name='k0051', instance=[1], tag=['kernel'], query={'class': 'k0051_gpt2', 'n_kernels': 1})
    k0052 = architecture.add_module(name='k0052', instance=[1], tag=['kernel'], query={'class': 'k0052_gpt2', 'n_kernels': 1})
    k0053 = architecture.add_module(name='k0053', instance=[1], tag=['kernel'], query={'class': 'k0053_gpt2', 'n_kernels': 1})
    k0054 = architecture.add_module(name='k0054', instance=[1], tag=['kernel'], query={'class': 'k0054_gpt2', 'n_kernels': 1})
    k0055 = architecture.add_module(name='k0055', instance=[1], tag=['kernel'], query={'class': 'k0055_gpt2', 'n_kernels': 1})
    k0056 = architecture.add_module(name='k0056', instance=[1], tag=['kernel'], query={'class': 'k0056_gpt2', 'n_kernels': 1})
    k0057 = architecture.add_module(name='k0057', instance=[1], tag=['kernel'], query={'class': 'k0057_gpt2', 'n_kernels': 1})
    k0058 = architecture.add_module(name='k0058', instance=[1], tag=['kernel'], query={'class': 'k0058_gpt2', 'n_kernels': 1})
    k0059 = architecture.add_module(name='k0059', instance=[1], tag=['kernel'], query={'class': 'k0059_gpt2', 'n_kernels': 1})
    k0060 = architecture.add_module(name='k0060', instance=[1], tag=['kernel'], query={'class': 'k0060_gpt2', 'n_kernels': 1})
    k0061 = architecture.add_module(name='k0061', instance=[1], tag=['kernel'], query={'class': 'k0061_gpt2', 'n_kernels': 1})
    k0062 = architecture.add_module(name='k0062', instance=[1], tag=['kernel'], query={'class': 'k0062_gpt2', 'n_kernels': 1})

    ##############################################
    ###############    Event    ##################
    ##############################################
    event.add_event(name='gpt2', subevent=['addmm', 'softmax', 'add', 'tanh', 'bmm', 'where', 'mul', 'div', 'gather', 'cat', 'copy',
                                           'native_layer_norm', 'bitwise_and', 'mm', 'le', 'argmax', 'masked_fill', 'sub', 'gt', 'cumsum',
                                           'index', 'any', 'fill', 'sum', 'eq', 'arange', 'bitwise_not', 'ge'],
                                           performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='addmm', subevent=['k0008', 'k0010', 'k0011', 'k0020'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='softmax', subevent=['k0004'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='add', subevent=['k0001', 'k0014', 'k0017', 'k0016', 'k0058', 'k0059'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='tanh', subevent=['k0015'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='bmm', subevent=['k0009', 'k0007'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='where', subevent=['k0002', 'k0022'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='mul', subevent=['k0005', 'k0013', 'k0033', 'k0048'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='div', subevent=['k0003'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='gather', subevent=['k0026', 'k0025'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='cat', subevent=['k0006', 'k0035'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='copy', subevent=['k0019', 'k0018', 'k0029', 'k0036', 'k0037', 'k0052'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='native_layer_norm', subevent=['k0012'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='bitwise_and', subevent=['k0021', 'k0034', 'k0049'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='mm', subevent=['k0023'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='le', subevent=['k0031'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='argmax', subevent=['k0027'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='masked_fill', subevent=['k0055'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='sub', subevent=['k0053', 'k0042', 'k0056'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='gt', subevent=['k0024'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='cumsum', subevent=['k0028', 'k0040', 'k0062'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='index', subevent=['k0030'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='any', subevent=['k0032'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='fill', subevent=['k0038', 'k0043', 'k0046', 'k0061'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='sum', subevent=['k0041'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='eq', subevent=['k0044', 'k0050', 'k0051', 'k0054'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='arange', subevent=['k0045', 'k0047'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='bitwise_not', subevent=['k0057'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    event.add_event(name='ge', subevent=['k0060'], performance='zoo/agraph/designs/gpu_gpt2/performance.py')
    ##############################################
    ###############    Metric    #################
    ##############################################
    metric.add_metric(name='power',   unit='W',  aggregation='summation')
    metric.add_metric(name='energy',  unit='uJ', aggregation='summation')
    metric.add_metric(name='runtime', unit='ms', aggregation='summation')

    ##############################################
    ###############   Workload   #################
    ##############################################
    gemm = workload.add_configuration(name='gemm')
    mapping = gemm.add_parameter(parameter_name='mapping', parameter_value=1)

    return agraph.generate()