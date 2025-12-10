
def init_metric(metric: object):
    metric.new_metric(name='area',           unit='mm^2',   aggregation='module')
    metric.new_metric(name='leakage_power',  unit='mW',     aggregation='module')
    metric.new_metric(name='dynamic_energy', unit='nJ',     aggregation='summation')
    metric.new_metric(name='cycle_count',    unit='cycles', aggregation='specified')
    metric.new_metric(name='runtime',        unit='ms',     aggregation='specified')