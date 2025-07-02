from zoo.llm.common.performance.memory import memory_scheduling
from zoo.llm.common.performance.router import router_scheduling
from zoo.llm.common.performance.mapping import array_performance
from collections import OrderedDict

def performance_count_to_int(performance_dict: OrderedDict) -> OrderedDict:
    """
    Sets the count value of a key in a dictionary
    """
    if 'cycle_count' in performance_dict:
        performance_dict['cycle_count']['value'] = performance_dict['cycle_count']['value']
    if 'runtime' in performance_dict:
        performance_dict['runtime']['value'] = performance_dict['runtime']['value']
    for key in performance_dict['subevent'].keys():
        performance_dict['subevent'][key]['count'] = int(performance_dict['subevent'][key]['count'])

    return performance_dict

def gemm_mapping(mapping_dict: OrderedDict, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    
    router = True if 'irouter' in architecture_dict else False
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict()

    # offchip scheduling, returns TiledGEMM object
    offchip_tiles = memory_scheduling.offchip_gemm_scheduling(batch=mapping_dict['batch'],
                                                                    m=mapping_dict['m'],
                                                                    n=mapping_dict['n'],
                                                                    k=mapping_dict['k'],
                                                                    architecture_dict=architecture_dict,
                                                                    workload_dict=workload_dict)
    

    # offchip memory events, returns dictionary containing input, weight, and output sram events and dram events
    offchip_memory_events_dict = memory_scheduling.offchip_gemm_events(tiles=offchip_tiles,
                                                                            architecture_dict=architecture_dict,
                                                                            workload_dict=workload_dict)
    performance_dict['subevent'].update(offchip_memory_events_dict['subevent'])
    
    # only compute router events if multinode configuration
    # average router events, returns dictionary containing average router events for input, weight, and output memory events
    # average event is not total, but router events per one sram event
    # needs offchip memory events to calculate router events
    if router:
        router_event_dict = router_scheduling.router_gemm_events(tiles=offchip_tiles,
                                                                       offchip_memory_events_dict=offchip_memory_events_dict,
                                                                       architecture_dict=architecture_dict,
                                                                       workload_dict=workload_dict)
        performance_dict['subevent'].update(router_event_dict['subevent'])
    
    # onchip scheduling, returns list of TiledGEMM objects. List is for each partial tiling configuration
    onchip_tiles_list = memory_scheduling.onchip_gemm_scheduling(m=mapping_dict['m'],
                                                                       n=mapping_dict['n'],
                                                                       k=mapping_dict['k'],
                                                                       tiles=offchip_tiles,
                                                                       architecture_dict=architecture_dict,
                                                                       workload_dict=workload_dict)
    
    # onchip memory events, returns dictionary containing input, weight, and output sram events
    onchip_memory_events_dict = memory_scheduling.onchip_gemm_events(onchip_tiling=onchip_tiles_list,
                                                                    architecture_dict=architecture_dict,
                                                                    workload_dict=workload_dict)
    performance_dict['subevent'].update(onchip_memory_events_dict['subevent'])


    # array-level events
    array_events_dict = array_performance.gemm_events(tiling=onchip_tiles_list,
                                                      architecture_dict=architecture_dict,
                                                      workload_dict=workload_dict,
                                                      performance_dict=performance_dict)
    performance_dict['subevent'].update(array_events_dict['subevent'])
        
    performance_dict = performance_count_to_int(performance_dict)
    return performance_dict

def nonlinear_mapping(mapping_dict: OrderedDict, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    
    router = True if 'irouter' in architecture_dict else False
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict()

    # offchip scheduling, returns Tilednonlinear object
    offchip_tiles = memory_scheduling.offchip_nonlinear_scheduling(batch=mapping_dict['batch'],
                                                                   m=mapping_dict['m'],
                                                                   n=mapping_dict['n'],
                                                                   architecture_dict=architecture_dict,
                                                                   workload_dict=workload_dict)
    

    # offchip memory events, returns dictionary containing input, weight, and output sram events and dram events
    offchip_memory_events_dict = memory_scheduling.offchip_nonlinear_events(tiles=offchip_tiles,
                                                                            architecture_dict=architecture_dict,
                                                                            workload_dict=workload_dict)
    performance_dict['subevent'].update(offchip_memory_events_dict['subevent'])
    
    # only compute router events if multinode configuration
    # average router events, returns dictionary containing average router events for input, weight, and output memory events
    # average event is not total, but router events per one sram event
    # needs offchip memory events to calculate router events
    if router:
        router_event_dict = router_scheduling.router_nonlinear_events(tiles=offchip_tiles,
                                                                      offchip_memory_events_dict=offchip_memory_events_dict,
                                                                      architecture_dict=architecture_dict,
                                                                      workload_dict=workload_dict)
        performance_dict['subevent'].update(router_event_dict['subevent'])
    
    # onchip scheduling, returns list of Tilednonlinear objects. List is for each partial tiling configuration
    onchip_tiles_list = memory_scheduling.onchip_nonlinear_scheduling(m=mapping_dict['m'],
                                                                      n=mapping_dict['n'],
                                                                      offchip_tiles=offchip_tiles,
                                                                      architecture_dict=architecture_dict,
                                                                      workload_dict=workload_dict)

    # onchip memory events, returns dictionary containing input, weight, and output sram events
    onchip_memory_events_dict = memory_scheduling.onchip_nonlinear_events(function=mapping_dict['function'],
                                                                          onchip_tiling=onchip_tiles_list,
                                                                          architecture_dict=architecture_dict,
                                                                          workload_dict=workload_dict)
    performance_dict['subevent'].update(onchip_memory_events_dict['subevent'])
    # array
    array_events_dict = array_performance.nonlinear_events(function=mapping_dict['function'],
                                                           tiling=onchip_tiles_list,
                                                           architecture_dict=architecture_dict,
                                                           workload_dict=workload_dict)
    performance_dict['subevent'].update(array_events_dict['subevent'])

    performance_dict = performance_count_to_int(performance_dict)
    return performance_dict

def mapping(mapping_dict: OrderedDict,architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    if mapping_dict['event'] == 'gemm':
        performance_dict = gemm_mapping(mapping_dict=mapping_dict,
                                        architecture_dict=architecture_dict,
                                        workload_dict=workload_dict)
    elif mapping_dict['event'] == 'nonlinear':
        performance_dict = nonlinear_mapping(mapping_dict=mapping_dict,
                                             architecture_dict=architecture_dict,
                                             workload_dict=workload_dict)
        
    return performance_dict
    