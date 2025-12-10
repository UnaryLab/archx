from collections import OrderedDict
from archx.utils import get_prod
import math

def tnn(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:

    performance_dict = OrderedDict()

    edge2pulse_dim = architecture_dict['edge2pulse']['instance']
    pulse2edge_dim = architecture_dict['pulse2edge']['instance']
    wta_dim = architecture_dict['WTA']['instance']
    top_reg_dim = architecture_dict['top_reg']['instance']
    neuron_dim = architecture_dict['neuron']['instance']
    stdp_dim = architecture_dict['stdp']['instance']
    fsm_synapse_dim = architecture_dict['fsm_synapse']['instance']
    fsm_simple_dim = architecture_dict['fsm_simple']['instance']
    

    frequency = architecture_dict['edge2pulse']['query']['frequency']
    cycles = 1
    performance_dict['cycle_count'] = OrderedDict({'value': cycles, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': cycles / (frequency * 1000), 'unit': 'ms'})

    threshold = 13
    theshold_bits = math.ceil(math.log2(threshold))


    edge2pulse_dict = {'count': get_prod(edge2pulse_dim)}
    pulse2edge_dict = {'count': get_prod(pulse2edge_dim)}
    wta_dict = {'count': get_prod(wta_dim)}
    top_reg_dict = {'count': get_prod(top_reg_dim)}
    neuron_dict = {'count': get_prod(neuron_dim)}
    stdp_dict = {'count': get_prod(stdp_dim)}
    fsm_synapse_dict = {'count': get_prod(fsm_synapse_dim)}
    fsm_simple_dict = {'count': get_prod(fsm_simple_dim)}

    performance_dict['subevent'] = OrderedDict({'edge2pulse': edge2pulse_dict,
                                               'pulse2edge': pulse2edge_dict,
                                               'WTA': wta_dict,
                                               'top_reg': top_reg_dict,
                                               'neuron': neuron_dict,
                                               'stdp': stdp_dict,
                                               'fsm_synapse': fsm_synapse_dict,
                                                  'fsm_simple': fsm_simple_dict
                                               })
    return performance_dict