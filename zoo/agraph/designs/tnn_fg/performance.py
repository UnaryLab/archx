from collections import OrderedDict
from archx.utils import get_prod

def tnn(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    edge2pulse_dim = architecture_dict['edge2pulse']['instance']
    pulse2edge_dim = architecture_dict['pulse2edge']['instance']
    wta_dim = architecture_dict['WTA']['instance']
    stdp_case_gen_dim = architecture_dict['stdp_case_gen']['instance']
    # incdec_dim = architecture_dict['incdec']['instance']
    # stabilize_func_dim = architecture_dict['stabilize_func']['instance']
    syn_readout_dim = architecture_dict['syn_readout']['instance']
    syn_weight_update_dim = architecture_dict['syn_weight_update']['instance']
    spike_gen_dim = architecture_dict['spike_gen']['instance']
    top_reg_dim = architecture_dict['top_reg']['instance']
    fsm_reg_1_dim = architecture_dict['fsm_reg_1']['instance']
    fsm_reg_2_dim = architecture_dict['fsm_reg_2']['instance']
    adder_dim = architecture_dict['adder']['instance']
    adder_output_dim = architecture_dict['adder_output']['instance']
    muxout_reg_dim = architecture_dict['muxout_reg']['instance']
    wta_and_dim = architecture_dict['wta_and']['instance']
    pac_and_dim = architecture_dict['pac_and']['instance']  

    frequency = architecture_dict['edge2pulse']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / (frequency * 1000), 'unit': 'ms'})

    edge2pulse_dict = {'count': get_prod(edge2pulse_dim)}
    pulse2edge_dict = {'count': get_prod(pulse2edge_dim)}
    wta_dict = {'count': get_prod(wta_dim)}
    stdp_case_gen_dict = {'count': get_prod(stdp_case_gen_dim)}
    # incdec_dict = {'count': get_prod(incdec_dim)}
    # stabilize_func_dict = {'count': get_prod(stabilize_func_dim)}
    syn_readout_dict = {'count': get_prod(syn_readout_dim)}
    syn_weight_update_dict = {'count': get_prod(syn_weight_update_dim)}
    spike_gen_dict = {'count': get_prod(spike_gen_dim)}
    top_reg_dict = {'count': get_prod(top_reg_dim)}
    fsm_reg_1_dict = {'count': get_prod(fsm_reg_1_dim)}
    fsm_reg_2_dict = {'count': get_prod(fsm_reg_2_dim)}
    adder_dict = {'count': get_prod(adder_dim)}
    adder_output_dict = {'count': get_prod(adder_output_dim)}
    muxout_reg_dict = {'count': get_prod(muxout_reg_dim)}
    wta_and_dict = {'count': get_prod(wta_and_dim)}
    pac_and_dict = {'count': get_prod(pac_and_dim)}

    performance_dict['subevent'] = OrderedDict({'edge2pulse': edge2pulse_dict,
                                                'pulse2edge': pulse2edge_dict,
                                                'WTA': wta_dict,
                                                'stdp_case_gen': stdp_case_gen_dict,
                                                # 'incdec': incdec_dict,
                                                # 'stabilize_func': stabilize_func_dict,
                                                'syn_readout': syn_readout_dict,
                                                'syn_weight_update': syn_weight_update_dict,
                                                'spike_gen': spike_gen_dict,
                                                'top_reg': top_reg_dict,
                                                'fsm_reg_1': fsm_reg_1_dict,
                                                'fsm_reg_2': fsm_reg_2_dict,
                                                'adder': adder_dict,
                                                'adder_output': adder_output_dict,
                                                'muxout_reg': muxout_reg_dict,
                                                'wta_and': wta_and_dict,
                                                'pac_and': pac_and_dict})
    return performance_dict