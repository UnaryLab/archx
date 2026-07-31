from collections import OrderedDict
from loguru import logger

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
    offchip_tiles = offchip_gemm_scheduling(batch=mapping_dict['batch'],
                                                              m=mapping_dict['m'],
                                                              n=mapping_dict['n'],
                                                              k=mapping_dict['k'],
                                                              architecture_dict=architecture_dict,
                                                              workload_dict=workload_dict)
    

    # offchip memory events, returns dictionary containing input, weight, and output sram events and dram events
    offchip_memory_events_dict = offchip_gemm_events(tiles=offchip_tiles,
                                                                            architecture_dict=architecture_dict,
                                                                            workload_dict=workload_dict)
    performance_dict['subevent'].update(offchip_memory_events_dict['subevent'])
    
    # only compute router events if multinode configuration
    # average router events, returns dictionary containing average router events for input, weight, and output memory events
    # average event is not total, but router events per one sram event
    # needs offchip memory events to calculate router events
    if router:
        router_event_dict = router_gemm_events(tiles=offchip_tiles,
                                                                       offchip_memory_events_dict=offchip_memory_events_dict,
                                                                       architecture_dict=architecture_dict,
                                                                       workload_dict=workload_dict)
        performance_dict['subevent'].update(router_event_dict['subevent'])
    
    # onchip scheduling, returns list of TiledGEMM objects. List is for each partial tiling configuration
    onchip_tiles_list = onchip_gemm_scheduling(m=mapping_dict['m'],
                                                                       n=mapping_dict['n'],
                                                                       k=mapping_dict['k'],
                                                                       tiles=offchip_tiles,
                                                                       architecture_dict=architecture_dict,
                                                                       workload_dict=workload_dict)
    
    # onchip memory events, returns dictionary containing input, weight, and output sram events
    onchip_memory_events_dict = onchip_gemm_events(onchip_tiling=onchip_tiles_list,
                                                                    architecture_dict=architecture_dict,
                                                                    workload_dict=workload_dict)
    performance_dict['subevent'].update(onchip_memory_events_dict['subevent'])


    # array-level events
    array_events_dict = gemm_events(tiling=onchip_tiles_list,
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
    offchip_tiles = offchip_nonlinear_scheduling(batch=mapping_dict['batch'],
                                                                   m=mapping_dict['m'],
                                                                   n=mapping_dict['n'],
                                                                   architecture_dict=architecture_dict,
                                                                   workload_dict=workload_dict)
    

    # offchip memory events, returns dictionary containing input, weight, and output sram events and dram events
    offchip_memory_events_dict = offchip_nonlinear_events(tiles=offchip_tiles,
                                                                            architecture_dict=architecture_dict,
                                                                            workload_dict=workload_dict)
    performance_dict['subevent'].update(offchip_memory_events_dict['subevent'])
    
    # only compute router events if multinode configuration
    # average router events, returns dictionary containing average router events for input, weight, and output memory events
    # average event is not total, but router events per one sram event
    # needs offchip memory events to calculate router events
    if router:
        router_event_dict = router_nonlinear_events(tiles=offchip_tiles,
                                                                      offchip_memory_events_dict=offchip_memory_events_dict,
                                                                      architecture_dict=architecture_dict,
                                                                      workload_dict=workload_dict)
        performance_dict['subevent'].update(router_event_dict['subevent'])
    
    # onchip scheduling, returns list of Tilednonlinear objects. List is for each partial tiling configuration
    onchip_tiles_list = onchip_nonlinear_scheduling(m=mapping_dict['m'],
                                                                      n=mapping_dict['n'],
                                                                      offchip_tiles=offchip_tiles,
                                                                      architecture_dict=architecture_dict,
                                                                      workload_dict=workload_dict)

    # onchip memory events, returns dictionary containing input, weight, and output sram events
    onchip_memory_events_dict = onchip_nonlinear_events(function=mapping_dict['function'],
                                                                          onchip_tiling=onchip_tiles_list,
                                                                          architecture_dict=architecture_dict,
                                                                          workload_dict=workload_dict)
    performance_dict['subevent'].update(onchip_memory_events_dict['subevent'])
    # array
    array_events_dict = nonlinear_events(function=mapping_dict['function'],
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

from agraph.designs.mugi.performance.utils import TiledGEMM, TiledMatrix
import math
from archx.utils import get_prod
from collections import OrderedDict

def offchip_gemm_scheduling(batch, m, k, n, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> TiledGEMM:
    """
    Sets tiling for dram <-> sram for single-node and multi-node configurations
    """
    # Retrieve dicts
    tile_m = workload_dict['noc_tile_m']
    tile_k = workload_dict['noc_tile_k']
    tile_n = workload_dict['noc_tile_n']

    # arch = workload_dict['architecture']
    # isram_dict = architecture_dict['isram']
    # wsram_dict = architecture_dict['wsram']
    # osram_dict = architecture_dict['osram']
    input_bitwidth = workload_dict['activation_bitwidth']
    weight_bitwidth = workload_dict['weight_bitwidth']
    output_bitwidth = workload_dict['activation_bitwidth']
    # stationary = workload_dict['noc_stationary']
    # array_width = architecture_dict['ififo']['instance'][-1]
    # array_height = architecture_dict['wfifo']['instance'][-1]

    # # SRAM configurations
    # # divide banks by to buffer
    # isram_bank = isram_dict['query']['bank'] / 2
    # wsram_bank = wsram_dict['query']['bank'] / 2
    # osram_bank = osram_dict['query']['bank'] / 2

    # isram_width = isram_dict['query']['width']
    # wsram_width = wsram_dict['query']['width']
    # osram_width = osram_dict['query']['width']

    # isram_depth = isram_dict['query']['depth']
    # wsram_depth = wsram_dict['query']['depth']
    # osram_depth = osram_dict['query']['depth']

    # isram_size = isram_bank * isram_width * isram_depth
    # wsram_size = wsram_bank * wsram_width * wsram_depth
    # osram_size = osram_bank * osram_width * osram_depth

    # isram_elements = isram_size / input_bitwidth
    # wsram_elements = wsram_size / weight_bitwidth
    # osram_elements = osram_size / output_bitwidth

    # # number of nodes
    # nodes = isram_dict['instance'][0] * isram_dict['instance'][1] if 'irouter' in architecture_dict else 1

    # if stationary == 'os':
    #     # Output stationary scheduling
    #     # initialize m tile size to m (in llms, very oftem batch size or smaller)
    #     # initialize n tile size to maximum size that fully utalizes noc
    #     tile_m = min(m, array_width)
    #     tile_n = ((n * batch) / nodes)

    #     # if tile_n does not fit in osram, reduce tile_n
    #     while(tile_n * tile_m > osram_elements):
    #         tile_n /= 2

    #     # if small tile n, and is smaller than array, increase tile_n
    #     while((tile_n * tile_m * 2 < osram_elements) and (tile_n < array_height)):
    #         tile_n *= 2

    #     # if small tile n, and can increase m, increase tile_m
    #     # multiply by 2 to check if you can increase tile_m
    #     while((tile_m * tile_n * 2 < osram_elements) and (tile_m < m)):
    #         tile_m *= 2

    #     # initialize k tile size to maximum size that fully utalizes wsram
    #     tile_k = wsram_elements / tile_n

    #     # if tile_k does not fit in isram, reduce tile_k
    #     while(tile_k * tile_m > isram_elements):
    #         tile_k /= 2

    #     while arch == 'tensor' and (tile_k < array_width):
    #           tile_k *= 2
    #           tile_n /= 2

    # elif stationary == 'ws':
    #     # Weight stationary scheduling
    #     # initialize m tile size to m (in llms, very oftem batch size or smaller)
    #     # initialize k tile size to maximum size that fully utalizes noc
    #     tile_m = min(m, array_width)
    #     tile_k = ((k * batch) / nodes)

    #     # if tile_k does not fit in isram, reduce tile_k
    #     while(tile_k * tile_m > isram_elements):
    #         tile_k /= 2

    #     # initialize n tile size to maximum size that fully utalizes wsram
    #     tile_n = wsram_elements / tile_k

    #     # if tile_n does not fit in osram or wsram, reduce tile_n
    #     while(tile_n > osram_elements / tile_m):
    #         tile_n /= 2

    #     # if small tile k, and can increase m, increase tile_m
    #     # multiply by 2 to check if you can increase tile_m
    #     while((tile_m * tile_k * 2 < isram_elements) and (tile_m < m)):
    #         tile_m *= 2

    # elif stationary == 'is':
    #     # Input stationary scheduling
    #     # initialize k tile size to m(quantized dim, smaller data size)
    #     # initialize m tile size to maximum size that fully utalizes noc
    #     tile_k = min(k, array_height)
    #     tile_m = ((m * batch) / nodes)

    #     # if tile_m does not fit in isram, reduce tile_m
    #     while(tile_m * tile_k > isram_elements):
    #         tile_m /= 2

    #     # initialize n tile size to maximum size that fully utalizes osram
    #     tile_n = osram_elements / tile_m
        
    #     # if tile_n does not fit in wsram, reduce tile_n
    #     while(tile_n * tile_k > wsram_elements):
    #         tile_n /= 2

    #     # if small tile k, and can increase k, increase tile_k
    #     # multiply by 2 to check if you can increase tile_k
    #     while((tile_k * tile_m * 2 < isram_elements) and (tile_k < k)):
    #         tile_k *= 2

    # if tile_m * tile_k > isram_elements:
    #     raise ValueError('Tile size exceeds isram size')
    # if tile_k * tile_n > wsram_elements:
    #     raise ValueError('Tile size exceeds wsram size')
    # if tile_m * tile_n > osram_elements:
    #     raise ValueError('Tile size exceeds osram size')

    # tile_m = 2 ** math.ceil(math.log2(tile_m))
    # tile_k = 2 ** math.ceil(math.log2(tile_k))
    # tile_n = 2 ** math.ceil(math.log2(tile_n))

    # tile_m = int(tile_m)
    # tile_k = int(tile_k)
    # tile_n = int(tile_n)

    tiles = TiledGEMM(batch=batch, m=m, k=k, n=n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=input_bitwidth, k_n_bitwidth=weight_bitwidth, m_n_bitwidth=output_bitwidth)
    if not tiles.is_valid:
        raise ValueError('Invalid tiling configuration')

    return tiles

def offchip_gemm_events(tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates dram and sram reads and writes to from offchip memory to onchip memory for GEMM operations
    """
    # Retrieve dicts

    # assume 64 bit wide
    dram_width = 64

    stationary = workload_dict['noc_stationary']
    isram_width = architecture_dict['isram']['query']['width']
    wsram_width = architecture_dict['wsram']['query']['width']
    osram_width = architecture_dict['osram']['query']['width']

    if stationary == 'os':
        # output stationary scheduling
        # M x K Input matrix dram <-> isram events (dram reads, isram writes)
        # input = batch x m_tiles x k_tiles x n_tiles x ceil(tile_bits / isram_width) -> (applies to all, but with flow adjusted for each, this is an output stationary example)
        # weight = batch x k_tiles x n_tiles x m_tiles x ceil(tile_bits / wsram_width)
        # output = batch x m_tiles x n_tiles x ceil(tile_bits / osram_width) -> (no k dim, as it's output stationary. Output stationary maps noc to outputs, so no need to map to k)
        # More detailed breakdown, allows for instances where the tile size is smaller than sram width, which increases events compared to using total_bits/width.
        m_full_k_full_events = tiles.m_full_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_full_tile_bits / dram_width)
        m_full_k_partial_events = tiles.m_full_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / dram_width)
        m_partial_k_full_events = tiles.m_partial_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / dram_width)
        m_partial_k_partial_events = tiles.m_partial_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / dram_width)

        # K x N Weight matrix dram <-> wsram events (dram reads, wsram writes)
        k_full_n_full_events = tiles.k_full_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_full_tile_bits / dram_width)
        k_full_n_partial_events = tiles.k_full_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / dram_width)
        k_partial_n_full_events = tiles.k_partial_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / dram_width)
        k_partial_n_partial_events = tiles.k_partial_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / dram_width)

        # M x N Output matrix osram <-> dram events (osram reads)
        m_full_n_full_events = tiles.m_full_n_full_total_tiles * math.ceil(tiles.m_full_n_full_tile_bits / dram_width)
        m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / dram_width)
        m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / dram_width)
        m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / dram_width)

    elif stationary == 'is':
        # input stationary scheduling
        # M x K Input matrix dram <-> isram events (dram reads, isram writes)
        m_full_k_full_events = tiles.m_full_k_full_total_tiles * math.ceil(tiles.m_full_k_full_tile_bits / dram_width)
        m_full_k_partial_events = tiles.m_full_k_partial_total_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / dram_width)
        m_partial_k_full_events = tiles.m_partial_k_full_total_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / dram_width)
        m_partial_k_partial_events = tiles.m_partial_k_partial_total_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / dram_width)

        # K x N Weight matrix dram <-> wsram events (dram reads, wsram writes)
        k_full_n_full_events = tiles.k_full_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_full_tile_bits / dram_width)
        k_full_n_partial_events = tiles.k_full_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / dram_width)
        k_partial_n_full_events = tiles.k_partial_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / dram_width)
        k_partial_n_partial_events = tiles.k_partial_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / dram_width)

        # M x N Output matrix osram <-> dram events (osram reads)
        m_full_n_full_events = tiles.m_full_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_full_tile_bits / dram_width)
        m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / dram_width)
        m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / dram_width)
        m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / dram_width)

    elif stationary == 'ws':
        # weight stationary scheduling
        # M x K Input matrix dram <-> isram events (dram reads, isram writes)
        m_full_k_full_events = tiles.m_full_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_full_tile_bits / dram_width)
        m_full_k_partial_events = tiles.m_full_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / dram_width)
        m_partial_k_full_events = tiles.m_partial_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / dram_width)
        m_partial_k_partial_events = tiles.m_partial_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / dram_width)

        # K x N Weight matrix dram <-> wsram events (dram reads, wsram writes)
        k_full_n_full_events = tiles.k_full_n_full_total_tiles * math.ceil(tiles.k_full_n_full_tile_bits / dram_width)
        k_full_n_partial_events = tiles.k_full_n_partial_total_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / dram_width)
        k_partial_n_full_events = tiles.k_partial_n_full_total_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / dram_width)
        k_partial_n_partial_events = tiles.k_partial_n_partial_total_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / dram_width)

        # M x N Output matrix osram <-> dram events (osram reads)
        m_full_n_full_events = tiles.m_full_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_full_tile_bits / dram_width)
        m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / dram_width)
        m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / dram_width)
        m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / dram_width)

    # Total dram <-> sram events
    m_k_events = m_full_k_full_events + m_full_k_partial_events + m_partial_k_full_events + m_partial_k_partial_events
    k_n_events = k_full_n_full_events + k_full_n_partial_events + k_partial_n_full_events + k_partial_n_partial_events
    m_n_events = m_full_n_full_events + m_full_n_partial_events + m_partial_n_full_events + m_partial_n_partial_events

    dram_input_reads_dict = OrderedDict({'count': m_k_events})
    dram_weight_reads_dict = OrderedDict({'count': k_n_events})
    dram_output_reads_dict = OrderedDict({'count': 0})
    dram_output_writes_dict = OrderedDict({'count': m_n_events})

    performance_dict = OrderedDict()

    if stationary == 'os':
        # output stationary scheduling
        # M x K Input matrix dram <-> isram events (dram reads, isram writes)
        # input = batch x m_tiles x k_tiles x n_tiles x ceil(tile_bits / isram_width) -> (applies to all, but with flow adjusted for each, this is an output stationary example)
        # weight = batch x k_tiles x n_tiles x m_tiles x ceil(tile_bits / wsram_width)
        # output = batch x m_tiles x n_tiles x ceil(tile_bits / osram_width) -> (no k dim, as it's output stationary. Output stationary maps noc to outputs, so no need to map to k)
        # More detailed breakdown, allows for instances where the tile size is smaller than sram width, which increases events compared to using total_bits/width.
        m_full_k_full_events = tiles.m_full_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_full_tile_bits / isram_width)
        m_full_k_partial_events = tiles.m_full_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / isram_width)
        m_partial_k_full_events = tiles.m_partial_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / isram_width)
        m_partial_k_partial_events = tiles.m_partial_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / isram_width)

        # K x N Weight matrix dram <-> wsram events (dram reads, wsram writes)
        k_full_n_full_events = tiles.k_full_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_full_tile_bits / wsram_width)
        k_full_n_partial_events = tiles.k_full_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / wsram_width)
        k_partial_n_full_events = tiles.k_partial_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / wsram_width)
        k_partial_n_partial_events = tiles.k_partial_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / wsram_width)

        # M x N Output matrix osram <-> dram events (osram reads)
        m_full_n_full_events = tiles.m_full_n_full_total_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    elif stationary == 'is':
        # input stationary scheduling
        # M x K Input matrix dram <-> isram events (dram reads, isram writes)
        m_full_k_full_events = tiles.m_full_k_full_total_tiles * math.ceil(tiles.m_full_k_full_tile_bits / isram_width)
        m_full_k_partial_events = tiles.m_full_k_partial_total_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / isram_width)
        m_partial_k_full_events = tiles.m_partial_k_full_total_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / isram_width)
        m_partial_k_partial_events = tiles.m_partial_k_partial_total_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / isram_width)

        # K x N Weight matrix dram <-> wsram events (dram reads, wsram writes)
        k_full_n_full_events = tiles.k_full_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_full_tile_bits / wsram_width)
        k_full_n_partial_events = tiles.k_full_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / wsram_width)
        k_partial_n_full_events = tiles.k_partial_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / wsram_width)
        k_partial_n_partial_events = tiles.k_partial_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / wsram_width)

        # M x N Output matrix osram <-> dram events (osram reads)
        m_full_n_full_events = tiles.m_full_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    elif stationary == 'ws':
        # weight stationary scheduling
        # M x K Input matrix dram <-> isram events (dram reads, isram writes)
        m_full_k_full_events = tiles.m_full_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_full_tile_bits / isram_width)
        m_full_k_partial_events = tiles.m_full_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / isram_width)
        m_partial_k_full_events = tiles.m_partial_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / isram_width)
        m_partial_k_partial_events = tiles.m_partial_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / isram_width)

        # K x N Weight matrix dram <-> wsram events (dram reads, wsram writes)
        k_full_n_full_events = tiles.k_full_n_full_total_tiles * math.ceil(tiles.k_full_n_full_tile_bits / wsram_width)
        k_full_n_partial_events = tiles.k_full_n_partial_total_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / wsram_width)
        k_partial_n_full_events = tiles.k_partial_n_full_total_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / wsram_width)
        k_partial_n_partial_events = tiles.k_partial_n_partial_total_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / wsram_width)

        # M x N Output matrix osram <-> dram events (osram reads)
        m_full_n_full_events = tiles.m_full_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    # Total dram <-> sram events
    m_k_events = m_full_k_full_events + m_full_k_partial_events + m_partial_k_full_events + m_partial_k_partial_events
    k_n_events = k_full_n_full_events + k_full_n_partial_events + k_partial_n_full_events + k_partial_n_partial_events
    m_n_events = m_full_n_full_events + m_full_n_partial_events + m_partial_n_full_events + m_partial_n_partial_events

    isram_offchip_writes_dict = OrderedDict({'count': m_k_events})
    wsram_offchip_writes_dict = OrderedDict({'count': k_n_events})
    osram_offchip_reads_dict = OrderedDict({'count': m_n_events})
    osram_offchip_writes_dict = OrderedDict({'count': 0})

    performance_dict['subevent'] = OrderedDict({
        'isram_offchip_writes': isram_offchip_writes_dict,
        'wsram_offchip_writes': wsram_offchip_writes_dict,
        'osram_offchip_reads': osram_offchip_reads_dict,
        'osram_offchip_writes': osram_offchip_writes_dict,
        'dram_input_reads': dram_input_reads_dict,
        'dram_weight_reads': dram_weight_reads_dict,
        'dram_output_reads': dram_output_reads_dict,
        'dram_output_writes': dram_output_writes_dict
    })

    return performance_dict

def onchip_gemm_scheduling(m, k, n, tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> list[TiledGEMM]:

    # Retrieve dicts
    arch = workload_dict['architecture']

    if arch == 'tensor':
        array_height = architecture_dict['wfifo']['instance'][-2]
        array_width = architecture_dict['ififo']['instance'][-2]
        array_depth = architecture_dict['wfifo']['instance'][-1]
    else:
        array_height = architecture_dict['wfifo']['instance'][-1]
        array_width = architecture_dict['ififo']['instance'][-1]
        array_depth = None

    tile_m = min(array_width, m)
    tile_k = min(array_height, k)
    tile_n = min(array_height, n)

    # Retrieve Reads/Writes for all possible tiling configurations
    tiling_configurations = []
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_full_n_full_total_tiles, m=tiles.tile_m, k=tiles.tile_k, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                                      # m full k full n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_full_n_partial_total_tiles, m=tiles.tile_m, k=tiles.tile_k, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                           # m full k full n partial
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_partial_n_full_total_tiles, m=tiles.tile_m, k=tiles.tile_k_partial, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                           # m full k partial n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_partial_n_partial_total_tiles, m=tiles.tile_m, k=tiles.tile_k_partial, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                # m full k partial n partial
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_full_n_full_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                           # m partial k full n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_full_n_partial_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                # m partial k full n partial
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_partial_n_full_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k_partial, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))                # m partial k partial n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_partial_n_partial_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k_partial, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height, array_depth=array_depth))     # m partial k partial n partial
    partial_tiles_tiling = []

    for tiling in tiling_configurations:
        if tiling.is_valid:
            partial_tiles_tiling.append(tiling)

    if partial_tiles_tiling == []:
        raise ValueError('Invalid tiling configuration')

    return partial_tiles_tiling

def onchip_gemm_events(onchip_tiling: list[TiledGEMM], architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates events from all node srams <-> nodes for GEMM operations.
    """

    performance_dict = OrderedDict()

    isram_onchip_reads_count = 0
    wsram_onchip_reads_count = 0
    osram_onchip_reads_count = 0
    osram_onchip_writes_count = 0

    for tiling in onchip_tiling:
        if not tiling.is_valid:
            raise ValueError('Invalid tiling configuration')
        isram_reads, wsram_reads, osram_reads, osram_writes = onchip_gemm_tile_events(tiling, architecture_dict, workload_dict)
        isram_onchip_reads_count += isram_reads
        wsram_onchip_reads_count += wsram_reads
        osram_onchip_reads_count += osram_reads
        osram_onchip_writes_count += osram_writes

    isram_onchip_reads_dict = OrderedDict({'count': isram_onchip_reads_count})
    wsram_onchip_reads_dict = OrderedDict({'count': wsram_onchip_reads_count})
    osram_onchip_reads_dict = OrderedDict({'count': osram_onchip_reads_count})
    osram_onchip_writes_dict = OrderedDict({'count': osram_onchip_writes_count})


    performance_dict['subevent'] = OrderedDict({
        'isram_onchip_reads': isram_onchip_reads_dict,
        'wsram_onchip_reads': wsram_onchip_reads_dict,
        'osram_onchip_reads': osram_onchip_reads_dict,
        'osram_onchip_writes': osram_onchip_writes_dict
    })

    return performance_dict

def onchip_gemm_tile_events(tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> tuple:
    """
    Estimates events from node srams <-> node for GEMM operation.
    """

    # Retrieve dicts
    stationary = workload_dict['node_stationary']
    isram_width = architecture_dict['isram']['query']['width']
    wsram_width = architecture_dict['wsram']['query']['width']
    osram_width = architecture_dict['osram']['query']['width']

    # scheduling
    if stationary == 'os':
        # output stationary scheduling
        # M x K Input matrix isram -> node writes
        m_full_k_full_read_events = tiles.m_full_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_full_tile_bits / isram_width)
        m_full_k_partial_read_events = tiles.m_full_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / isram_width)
        m_partial_k_full_read_events = tiles.m_partial_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / isram_width)
        m_partial_k_partial_read_events = tiles.m_partial_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / isram_width)

        # K x N Weight matrix wsram -> node writes
        k_full_n_full_read_events = tiles.k_full_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_full_tile_bits / wsram_width)
        k_full_n_partial_read_events = tiles.k_full_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / wsram_width)
        k_partial_n_full_read_events = tiles.k_partial_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / wsram_width)
        k_partial_n_partial_read_events = tiles.k_partial_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / wsram_width)

        # M x N Output matrix osram -> node reads
        m_full_n_full_write_events = tiles.m_full_n_full_total_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_write_events = tiles.m_full_n_partial_total_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_write_events = tiles.m_partial_n_full_total_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_write_events = tiles.m_partial_n_partial_total_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

        # M x N Output matrix osram -> node writes
        m_full_n_full_read_events = tiles.m_full_n_full_total_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_read_events = tiles.m_full_n_partial_total_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_read_events = tiles.m_partial_n_full_total_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_read_events = tiles.m_partial_n_partial_total_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    elif stationary == 'is':
        # input stationary scheduling
        # M x K Input matrix isram -> node writes
        m_full_k_full_read_events = tiles.m_full_k_full_total_tiles * math.ceil(tiles.m_full_k_full_tile_bits / isram_width)
        m_full_k_partial_read_events = tiles.m_full_k_partial_total_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / isram_width)
        m_partial_k_full_read_events = tiles.m_partial_k_full_total_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / isram_width)
        m_partial_k_partial_read_events = tiles.m_partial_k_partial_total_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / isram_width)

        # K x N Weight matrix wsram -> node writes
        k_full_n_full_read_events = tiles.k_full_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_full_tile_bits / wsram_width)
        k_full_n_partial_read_events = tiles.k_full_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / wsram_width)
        k_partial_n_full_read_events = tiles.k_partial_n_full_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / wsram_width)
        k_partial_n_partial_read_events = tiles.k_partial_n_partial_total_tiles * tiles.m_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / wsram_width)

        # M x N Output matrix osram -> node reads
        m_full_n_full_write_events = tiles.m_full_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_write_events = tiles.m_full_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_write_events = tiles.m_partial_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_write_events = tiles.m_partial_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

        # M x N Output matrix osram -> node writes
        m_full_n_full_read_events = tiles.m_full_n_full_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_read_events = tiles.m_full_n_partial_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_read_events = tiles.m_partial_n_full_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_read_events = tiles.m_partial_n_partial_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    elif stationary == 'ws':
        # weight stationary scheduling
        # M x K Input matrix isram -> node writes
        m_full_k_full_read_events = tiles.m_full_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_full_tile_bits / isram_width)
        m_full_k_partial_read_events = tiles.m_full_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_full_k_partial_tile_bits / isram_width)
        m_partial_k_full_read_events = tiles.m_partial_k_full_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_full_tile_bits / isram_width)
        m_partial_k_partial_read_events = tiles.m_partial_k_partial_total_tiles * tiles.n_tiles * math.ceil(tiles.m_partial_k_partial_tile_bits / isram_width)

        # K x N Weight matrix wsram -> node writes
        k_full_n_full_read_events = tiles.k_full_n_full_total_tiles * math.ceil(tiles.k_full_n_full_tile_bits / wsram_width)
        k_full_n_partial_read_events = tiles.k_full_n_partial_total_tiles * math.ceil(tiles.k_full_n_partial_tile_bits / wsram_width)
        k_partial_n_full_read_events = tiles.k_partial_n_full_total_tiles * math.ceil(tiles.k_partial_n_full_tile_bits / wsram_width)
        k_partial_n_partial_read_events = tiles.k_partial_n_partial_total_tiles * math.ceil(tiles.k_partial_n_partial_tile_bits / wsram_width)

        # M x N Output matrix osram -> node reads
        m_full_n_full_write_events = tiles.m_full_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_write_events = tiles.m_full_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_write_events = tiles.m_partial_n_full_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_write_events = tiles.m_partial_n_partial_total_tiles * tiles.k_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

        # M x N Output matrix osram -> node writes
        m_full_n_full_read_events = tiles.m_full_n_full_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
        m_full_n_partial_read_events = tiles.m_full_n_partial_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
        m_partial_n_full_read_events = tiles.m_partial_n_full_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
        m_partial_n_partial_read_events = tiles.m_partial_n_partial_total_tiles * (tiles.k_tiles - 1) * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    # Total sram <-> node events
    isram_reads = m_full_k_full_read_events + m_full_k_partial_read_events + m_partial_k_full_read_events + m_partial_k_partial_read_events
    wsram_reads = k_full_n_full_read_events + k_full_n_partial_read_events + k_partial_n_full_read_events + k_partial_n_partial_read_events
    osram_reads = m_full_n_full_read_events + m_full_n_partial_read_events + m_partial_n_full_read_events + m_partial_n_partial_read_events
    osram_writes = m_full_n_full_write_events + m_full_n_partial_write_events + m_partial_n_full_write_events + m_partial_n_partial_write_events

    return isram_reads, wsram_reads, osram_reads, osram_writes

def offchip_nonlinear_scheduling(batch, m, n, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> TiledMatrix:
    """
    Sets tiling for dram <-> sram for single-node and multi-node configurations
    """
    # Retrieve dicts
    arch = workload_dict['architecture']
    wsram_dict = architecture_dict['wsram']
    osram_dict = architecture_dict['osram']
    input_bitwidth = workload_dict['activation_bitwidth']
    output_bitwidth = workload_dict['activation_bitwidth']
    if arch != 'tensor':
        array_width = architecture_dict['ififo']['instance'][-1]
        array_height = architecture_dict['wfifo']['instance'][-1]
    else:
        array_width = get_prod(architecture_dict['wfifo']['instance'][-2:])
        array_height = get_prod(architecture_dict['ififo']['instance'][-2:])
    architecture = workload_dict['architecture'].lower()

    # SRAM configurations
    # divide banks by to buffer
    wsram_bank = wsram_dict['query']['bank'] / 2
    osram_bank = osram_dict['query']['bank'] / 2

    wsram_width = wsram_dict['query']['width']
    osram_width = osram_dict['query']['width']

    wsram_depth = wsram_dict['query']['depth']
    osram_depth = osram_dict['query']['depth']

    wsram_size = wsram_bank * wsram_width * wsram_depth
    osram_size = osram_bank * osram_width * osram_depth

    wsram_elements = wsram_size / input_bitwidth
    osram_elements = osram_size / output_bitwidth

    # number of nodes
    nodes = wsram_dict['instance'][0] * wsram_dict['instance'][1] if 'router' in architecture_dict else 1

    # input and output matrices are the same size, so so input and output must fit in both
    if architecture == 'mugi':
        elements = min(wsram_elements, osram_elements)
    else:
        elements = osram_elements
    
    # initialize m tile size to m (in llms, very oftem batch size or smaller)
    # initialize n tile size to maximum size that fully utalizes noc
    tile_m = min(m, array_height)
    tile_n = ((n * batch) / nodes)

    # if tile_n does not fit in osram, reduce tile_n
    while(tile_n * tile_m > elements):
        tile_n /= 2

    # if small tile n, and is smaller than array, increase tile_n
    while((tile_n * tile_m * 2 < elements)):
        tile_n *= 2

    # if small tile n, and can increase m, increase tile_m
    # multiply by 2 to check if you can increase tile_m
    while((tile_m * tile_n * 2 < elements) and (tile_m < m)):
        tile_m *= 2

    if tile_m * tile_n > wsram_elements:
        raise ValueError('Tile size exceeds wsram size')
    if tile_m * tile_n > osram_elements:
        raise ValueError('Tile size exceeds osram size')
    
    tile_m = int(tile_m)
    tile_n = int(tile_n)

    

    tiles = TiledMatrix(batch=batch, m=m, n=n, tile_m=tile_m, tile_n=tile_n, m_n_bitwidth=input_bitwidth)

    if not tiles.is_valid:
        raise ValueError('Invalid tiling configuration')

    return tiles

def offchip_nonlinear_events(tiles: TiledMatrix, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates dram and sram reads and writes to from offchip memory to onchip memory for non-linear
    """

    performance_dict = OrderedDict()

    # Retrieve dicts
    isram_width = architecture_dict['isram']['query']['width']
    osram_width = architecture_dict['osram']['query']['width']
    if 'irouter' in architecture_dict:
        router_height = architecture_dict['irouter']['instance'][0]
        router_width = architecture_dict['irouter']['instance'][1]
    else:
        router_height = 1
        router_width = 1
    architecture = workload_dict['architecture'].lower()

    noc_dim = router_height * router_width if 'router' in architecture_dict else 1

    m_full_n_full_events = tiles.m_full_n_full_total_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
    m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
    m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
    m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    m_n_event = m_full_n_full_events + m_full_n_partial_events + m_partial_n_full_events + m_partial_n_partial_events

    # If virtual lut is used in isram
    if architecture == 'mugi':
        lut_height = workload_dict['lut_height']
        lut_width = workload_dict['lut_width']
        weight_bitwidth = workload_dict['weight_bitwidth']
        lut_dim = lut_height * lut_width
        lut_bits = lut_dim * weight_bitwidth
        
        isram_writes_count = min(tiles.m_n_matrix_tiles, noc_dim) * math.ceil(lut_bits / isram_width)
        dram_input_reads_count = math.ceil(lut_bits / isram_width)
    else:
        isram_writes_count = 0
        dram_input_reads_count = 0

    wsram_writes_count = 0
    osram_reads_count = m_n_event
    osram_writes_count = m_n_event
    dram_weight_reads_count = 0
    dram_output_reads_count = m_n_event
    dram_output_writes_count = m_n_event

    isram_offchip_writes_dict = OrderedDict({'count': isram_writes_count})
    wsram_offchip_writes_dict = OrderedDict({'count': wsram_writes_count})
    osram_offchip_reads_dict = OrderedDict({'count': osram_reads_count})
    osram_offchip_writes_dict = OrderedDict({'count': osram_writes_count})
    dram_input_reads_dict = OrderedDict({'count': dram_input_reads_count})
    dram_weight_reads_dict = OrderedDict({'count': dram_weight_reads_count})
    dram_output_reads_dict = OrderedDict({'count': dram_output_reads_count})
    dram_output_writes_dict = OrderedDict({'count': dram_output_writes_count})

    performance_dict = OrderedDict()

    performance_dict['subevent'] = OrderedDict({
        'isram_offchip_writes': isram_offchip_writes_dict,
        'wsram_offchip_writes': wsram_offchip_writes_dict,
        'osram_offchip_reads': osram_offchip_reads_dict,
        'osram_offchip_writes': osram_offchip_writes_dict,
        'dram_input_reads': dram_input_reads_dict,
        'dram_weight_reads': dram_weight_reads_dict,
        'dram_output_reads': dram_output_reads_dict,
        'dram_output_writes': dram_output_writes_dict
    })

    return performance_dict

def onchip_nonlinear_events(function: str, onchip_tiling: list[TiledMatrix], architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates events from node srams <-> node for nonlinear operation.
    """

    performance_dict = OrderedDict()

    isram_onchip_reads_count = 0
    wsram_onchip_reads_count = 0
    osram_onchip_reads_count = 0
    osram_onchip_writes_count = 0

    for tiling in onchip_tiling:
        if not tiling.is_valid:
            raise ValueError('Invalid tiling configuration')
        isram_reads, wsram_reads, osram_reads, osram_writes = onchip_nonlinear_tile_events(function=function, tiles=tiling, architecture_dict=architecture_dict, workload_dict=workload_dict)
        isram_onchip_reads_count += isram_reads
        wsram_onchip_reads_count += wsram_reads
        osram_onchip_reads_count += osram_reads
        osram_onchip_writes_count += osram_writes

    isram_onchip_reads_dict = OrderedDict({'count': isram_onchip_reads_count})
    wsram_onchip_reads_dict = OrderedDict({'count': wsram_onchip_reads_count})
    osram_onchip_reads_dict = OrderedDict({'count': osram_onchip_reads_count})
    osram_onchip_writes_dict = OrderedDict({'count': osram_onchip_writes_count})

    performance_dict['subevent'] = OrderedDict({
        'isram_onchip_reads': isram_onchip_reads_dict,
        'wsram_onchip_reads': wsram_onchip_reads_dict,
        'osram_onchip_reads': osram_onchip_reads_dict,
        'osram_onchip_writes': osram_onchip_writes_dict
    })

    return performance_dict

def onchip_nonlinear_scheduling(m, n, offchip_tiles: TiledMatrix, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> TiledMatrix:

    arch = workload_dict['architecture']
    if arch != 'tensor':
        wfifo_width = architecture_dict['wfifo']['instance'][-1]
        ofifo_width = architecture_dict['ofifo']['instance'][-1]
    else:
        wfifo_width = get_prod(architecture_dict['ififo']['instance'][-2:])
        ofifo_width = get_prod(architecture_dict['ofifo']['instance'][-2:])
    
    array_height = ofifo_width

    if wfifo_width != ofifo_width:
        raise ValueError('wfifo and ofifo width must be the same')

    tile_m = min(array_height, m)
    tile_n = 1

    tiling_configurations = []
    tiling_configurations.append(TiledMatrix(batch=offchip_tiles.m_full_n_full_total_tiles, m=offchip_tiles.tile_m, n=offchip_tiles.tile_n, tile_m=tile_m, tile_n=tile_n, m_n_bitwidth=offchip_tiles.m_n_bitwidth, array_height=array_height))                         # m full n full
    tiling_configurations.append(TiledMatrix(batch=offchip_tiles.m_full_n_partial_total_tiles, m=offchip_tiles.tile_m, n=offchip_tiles.tile_n_partial, tile_m=tile_m, tile_n=tile_n, m_n_bitwidth=offchip_tiles.m_n_bitwidth, array_height=array_height))              # m full n partial
    tiling_configurations.append(TiledMatrix(batch=offchip_tiles.m_partial_n_full_total_tiles, m=offchip_tiles.tile_m_partial, n=offchip_tiles.tile_n, tile_m=tile_m, tile_n=tile_n, m_n_bitwidth=offchip_tiles.m_n_bitwidth, array_height=array_height))              # m partial n full
    tiling_configurations.append(TiledMatrix(batch=offchip_tiles.m_partial_n_partial_total_tiles, m=offchip_tiles.tile_m_partial, n=offchip_tiles.tile_n_partial, tile_m=tile_m, tile_n=tile_n, m_n_bitwidth=offchip_tiles.m_n_bitwidth, array_height=array_height))   # m partial n partial

    partial_tiles_tiling = []

    for tiling in tiling_configurations:
        if tiling.is_valid:
            partial_tiles_tiling.append(tiling)

    if partial_tiles_tiling == []:
        raise ValueError('Invalid tiling configuration')

    return partial_tiles_tiling

def onchip_nonlinear_tile_events(function: str, tiles: TiledMatrix, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> tuple:
    """
    Estimates events from node srams <-> node for nonlinear operation.
    """

    # check if LUT implementation (Mugi)
    architecture = workload_dict['architecture'].lower()

    # retrieve dicts
    isram_width = architecture_dict['isram']['query']['width']
    wsram_width = architecture_dict['wsram']['query']['width']
    osram_width = architecture_dict['osram']['query']['width']

    m_full_n_full_events = tiles.m_full_n_full_total_tiles * math.ceil(tiles.m_full_n_full_tile_bits / osram_width)
    m_full_n_partial_events = tiles.m_full_n_partial_total_tiles * math.ceil(tiles.m_full_n_partial_tile_bits / osram_width)
    m_partial_n_full_events = tiles.m_partial_n_full_total_tiles * math.ceil(tiles.m_partial_n_full_tile_bits / osram_width)
    m_partial_n_partial_events = tiles.m_partial_n_partial_total_tiles * math.ceil(tiles.m_partial_n_partial_tile_bits / osram_width)

    m_n_events = m_full_n_full_events + m_full_n_partial_events + m_partial_n_full_events + m_partial_n_partial_events

    if function == 'softmax':
        osram_reads = m_n_events * 2
        osram_writes = m_n_events * 2
    else: # default mapping for activation function. In this case, SiLU. Holds true to activation functions that apply element-wise operations (unlike softmax which divides by sum).
        osram_reads = m_n_events
        osram_writes = m_n_events

    if architecture == 'mugi':
        lut_height = workload_dict['lut_height']
        window_width = workload_dict['window_width']
        weight_bitwidth = workload_dict['weight_bitwidth']
        lut_bits = window_width * weight_bitwidth

        isram_reads = (tiles.m_n_total_tiles * lut_height) * math.ceil(lut_bits / isram_width)
    else:
        isram_reads = 0

    wsram_reads = 0

    return isram_reads, wsram_reads, osram_reads, osram_writes

def gemm_events(tiling: list[TiledGEMM], architecture_dict: OrderedDict, workload_dict: OrderedDict, performance_dict: OrderedDict) -> OrderedDict:

    performance_dict = None

    for i, tiles in enumerate(tiling):
        if tiles.is_valid:
            if performance_dict is None:
                performance_dict = gemm_tile_events(tiles=tiles, architecture_dict=architecture_dict, workload_dict=workload_dict)
            else:
                performance_dict = sum_subevents(performance_dict, gemm_tile_events(tiles=tiles, architecture_dict=architecture_dict, workload_dict=workload_dict))

    # gemm/nonlinear mux switching event (once per gemm)
    if workload_dict['architecture'] == 'mugi':
        performance_dict['subevent']['instruction'] = OrderedDict({'count': 1})

    return performance_dict

def nonlinear_events(function: str, tiling: list[TiledMatrix], architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:

    performance_dict = None

    for tiles in tiling:
        if tiles.is_valid:
            if performance_dict is None:
                performance_dict = nonlinear_tile_events(function=function, tiles=tiles, architecture_dict=architecture_dict, workload_dict=workload_dict)
            else:
                performance_dict = sum_subevents(performance_dict, nonlinear_tile_events(function=function, tiles=tiles, architecture_dict=architecture_dict, workload_dict=workload_dict))

    # gemm/nonlinear mux switching event (once per gemm)
    if workload_dict['architecture'] == 'mugi':
        performance_dict['subevent']['instruction'] = OrderedDict({'count': 1})

    return performance_dict

def gemm_tile_events(tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:

    vector_array_dim = architecture_dict['multiplier_vector']['instance'][-1]

    architecture = workload_dict['architecture']
    stationary = workload_dict['node_stationary']
    et_cycles = workload_dict.get('early_termination_cycles')
    cycles = workload_dict.get('cycles')

    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict()

    # load input
    if architecture == 'tensor':
        input_events = tiles.n_total_tiles * tiles.k_tiles if stationary == 'is' else tiles.m_n_total_tiles * tiles.k_tiles
    else:
        input_events = tiles.n_total_tiles * tiles.k if stationary == 'is' else tiles.m_n_total_tiles * tiles.k
    input_event_utilization = (tiles.m / (tiles.m_tiles * tiles.tile_m)) * tiles.m_util
    input_events *= input_event_utilization
    cycle_count_utilization = 1 / input_event_utilization
    performance_dict['subevent']['input_gemm'] = OrderedDict({'count': input_events,
                                                              'factor': {'cycle_count': cycle_count_utilization,
                                                                         'runtime': cycle_count_utilization}})
    # counter value reuse
    if architecture in ['mugi', 'carat']:
        counter_reuse_events = tiles.m_n_total_tiles * tiles.k
        counter_reuse_cycles_utalization = et_cycles / cycles if et_cycles is not None else 1
        performance_dict['subevent']['counter_reuse'] = OrderedDict({'count': counter_reuse_events,
                                                                     'factor': {'cycle_count': counter_reuse_cycles_utalization,
                                                                                'runtime': counter_reuse_cycles_utalization}})

    # broadcast value reuse
    if architecture in ['mugi', 'carat']:
        input_reuse_events = tiles.m_n_total_tiles * tiles.k
        input_reuse_event_utilization = (tiles.m / (tiles.m_tiles * tiles.tile_m)) * tiles.m_util
        input_reuse_events *= input_reuse_event_utilization
        input_reuse_cycle_utilization = et_cycles / cycles if et_cycles is not None else 1
        input_reuse_cycle_utilization *= 1 / input_reuse_event_utilization
        performance_dict['subevent']['input_reuse_gemm'] = OrderedDict({'count': input_reuse_events,
                                                                        'factor': {'cycle_count': input_reuse_cycle_utilization,
                                                                                   'runtime': input_reuse_cycle_utilization}})
    
    # load weight
    if architecture == 'tensor':
        weight_events = tiles.m_total_tiles * tiles.k_tiles if stationary == 'ws' else tiles.m_n_total_tiles * tiles.k_tiles
    else:
        weight_events = tiles.m_total_tiles * tiles.k if stationary == 'ws' else tiles.m_n_total_tiles * tiles.k
    weight_event_utilization = (tiles.n / (tiles.n_tiles * tiles.tile_n)) * tiles.n_util
    weight_events *= weight_event_utilization
    weight_cycle_utilization = 1 / weight_event_utilization
    performance_dict['subevent']['weight_gemm'] = OrderedDict({'count': weight_events,
                                                               'factor': {'cycle_count': weight_cycle_utilization,
                                                                          'runtime': weight_cycle_utilization}})

    # temporal conversion
    if architecture in ['mugi', 'carat']:
        weight_reuse_events = tiles.m_n_total_tiles * tiles.k
        weight_reuse_event_utilization = (tiles.n / (tiles.n_tiles * tiles.tile_n)) * tiles.n_util
        weight_reuse_events *= weight_reuse_event_utilization
        weight_reuse_cycle_utilization = et_cycles / cycles if et_cycles is not None else 1
        weight_reuse_cycle_utilization *= 1 / weight_reuse_event_utilization
        performance_dict['subevent']['weight_reuse_gemm'] = OrderedDict({'count': weight_reuse_events,
                                                                         'factor': {'cycle_count': weight_reuse_cycle_utilization,
                                                                                    'runtime': weight_reuse_cycle_utilization}})

    # array computation
    if architecture == 'tensor':
        array_events = tiles.m_k_n_total_tiles
    else:
        array_events = tiles.m_n_total_tiles * tiles.k
    array_events_utilization = input_event_utilization * weight_event_utilization
    array_events *= array_events_utilization
    array_cycle_utilization = et_cycles / cycles if et_cycles is not None else 1
    array_cycle_utilization *= 1 / array_events_utilization
    performance_dict['subevent']['array_gemm'] = OrderedDict({'count': array_events,
                                                              'factor': {'cycle_count': array_cycle_utilization,
                                                                         'runtime': array_cycle_utilization}})
    
    # hardwarea for multiple spikes in compute array.
    if architecture in ['mugi', 'carat']:
        array_events = tiles.m_n_total_tiles * tiles.k
        array_events_utilization = input_event_utilization * weight_event_utilization * 0.5
        array_events *= array_events_utilization
        array_cycle_utilization = et_cycles / cycles if et_cycles is not None else 1
        array_cycle_utilization *= 1 if array_events_utilization == 0 else 1 / array_events_utilization
        performance_dict['subevent']['array_fifo_gemm'] = OrderedDict({'count': array_events,
                                                                'factor': {'cycle_count': array_cycle_utilization,
                                                                            'runtime': array_cycle_utilization}})

    # vector scaling
    if architecture in ['mugi']:
        vector_events = tiles.m_total * tiles.n_tiles * (tiles.tile_n / vector_array_dim)
        vector_events_utilization = tiles.m_n_total_matrix / (vector_events * vector_array_dim)
        vector_events *= vector_events_utilization
        vector_cycle_utilization = 1 / vector_events_utilization
        performance_dict['subevent']['vector'] = OrderedDict({'count': vector_events,
                                                              'factor': {'cycle_count': vector_cycle_utilization,
                                                                         'runtime': vector_cycle_utilization}})
    elif architecture in ['carat', 'simd', 'systolic']:
        vector_events = tiles.m_total * tiles.n_tiles
        vector_events_utilization = tiles.n / (tiles.n_tiles * tiles.tile_n)
        vector_events *= vector_events_utilization
        vector_cycle_utilization = 1 / vector_events_utilization
        performance_dict['subevent']['vector_gemm'] = OrderedDict({'count': vector_events,
                                                          'factor': {'cycle_count': vector_cycle_utilization,
                                                                     'runtime': vector_cycle_utilization}})
    elif architecture in ['tensor']:
        vector_events = tiles.m_total_tiles * tiles.n_tiles
        vector_events_utilization = tiles.n / (tiles.n_tiles * tiles.tile_n)
        vector_events *= vector_events_utilization
        vector_cycle_utilization = 1 / vector_events_utilization
        performance_dict['subevent']['vector_gemm'] = OrderedDict({'count': vector_events,
                                                          'factor': {'cycle_count': vector_cycle_utilization,
                                                                     'runtime': vector_cycle_utilization}})

    # unused events
    if architecture == 'mugi':
        nonlinear_dict = 0
        performance_dict['subevent']['nonlinear_gemm'] = OrderedDict({'count': nonlinear_dict})

    return performance_dict

def nonlinear_tile_events(function: str, tiles: TiledMatrix, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:

    vector_array_dim = architecture_dict['multiplier_vector']['instance'][-1]
    architecture = workload_dict['architecture']
    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict()

    # counter value reuse
    if architecture in ['mugi']:
        # input load
        input_events = tiles.m_tiles * tiles.n_total
        performance_dict['subevent']['input_nonlinear'] = OrderedDict({'count': input_events})
        
        # counter value reuse
        counter_reuse_events = tiles.m_tiles * tiles.n_total
        performance_dict['subevent']['counter_reuse'] = OrderedDict({'count': counter_reuse_events})

        # load weight (activations in nonlinear)
        weight_events = tiles.m_total_tiles * tiles.n
        weight_events_utilization = (tiles.m / (tiles.m_tiles * tiles.tile_m)) * tiles.m_util
        weight_events *= weight_events_utilization
        weight_events_cycle_utilization = 1 / weight_events_utilization if function == 'softmax' else (1 / weight_events_utilization) * 2
        performance_dict['subevent']['weight_nonlinear'] = OrderedDict({'count': weight_events,
                                                                        'factor': {'cycle_count': weight_events_cycle_utilization,
                                                                                   'runtime': weight_events_cycle_utilization}})
        
        # temporal conversion (silu maps for both signs, so mulitply by 2)
        weight_reuse_events = tiles.m_tiles * tiles.n_total
        weight_reuse_utilization = weight_events_utilization if function == 'softmax' else weight_events_utilization * 2
        weight_reuse_events *= weight_reuse_utilization
        weight_reuse_cycle_utilization = 1 / weight_reuse_utilization
        performance_dict['subevent']['weight_reuse_nonlinear'] = OrderedDict({'count': weight_reuse_events,
                                                                              'factor': {'cycle_count': weight_reuse_cycle_utilization,
                                                                                         'runtime': weight_reuse_cycle_utilization}})

        if function == 'softmax':
            # array computation (LUT selection)
            array_events = tiles.m_tiles * tiles.n_total
            array_events_utilization = weight_events_utilization * 2
            array_events *= array_events_utilization
            array_cycle_utilization = 1 / array_events_utilization
            performance_dict['subevent']['array_nonlinear'] = OrderedDict({'count': array_events,
                                                                        'factor': {'cycle_count': array_cycle_utilization,
                                                                                    'runtime': array_cycle_utilization}})

            # summate exp
            summation_events = tiles.m_tiles * tiles.n_total
            summation_events_utilization = weight_events_utilization
            summation_events *= summation_events_utilization
            summation_cycle_utilization = 1 / summation_events_utilization
            performance_dict['subevent']['summation'] = OrderedDict({'count': summation_events,
                                                                     'factor': {'cycle_count': summation_cycle_utilization,
                                                                                'runtime': summation_cycle_utilization}})
            
            # divide by sum
            vector_events = tiles.m_tiles * (tiles.tile_m / vector_array_dim) * tiles.n_total
            vector_events_utilization = tiles.m_n_total_matrix / (vector_events * vector_array_dim)
            vector_events *= vector_events_utilization
            vector_cycle_utilization = 1 / vector_events_utilization
            performance_dict['subevent']['vector'] = OrderedDict({'count': vector_events,
                                                                  'factor': {'cycle_count': vector_cycle_utilization,
                                                                             'runtime': vector_cycle_utilization}})
        elif function == 'silu':
            # array computation (LUT selection)
            array_events = tiles.m_tiles * tiles.n_total
            array_events_utilization = weight_events_utilization * 2
            array_events *= array_events_utilization
            array_cycle_utilization = 1 / array_events_utilization
            performance_dict['subevent']['array_nonlinear'] = OrderedDict({'count': array_events,
                                                                        'factor': {'cycle_count': array_cycle_utilization,
                                                                                    'runtime': array_cycle_utilization}})

            # no sum / divide
            summation_events = 0
            performance_dict['subevent']['summation'] = OrderedDict({'count': summation_events})

            vector_events = 0
            performance_dict['subevent']['vector'] = OrderedDict({'count': vector_events})
    

        # hardware for multiple spikes in compute array.
        array_events = 0
        performance_dict['subevent']['array_fifo_nonlinear'] = OrderedDict({'count': array_events})

    # nonlinear vector
    if architecture in ['carat', 'simd', 'systolic']:
        vector_events = tiles.m_tiles * tiles.n_total
        vector_events_utilization = tiles.m / (tiles.m_tiles * tiles.tile_m) * tiles.m_util
        vector_events *= vector_events_utilization
        vector_cycle_utilization = 1 / vector_events_utilization

        if architecture in ['systolic']:
            performance_dict['subevent'][function + '_nonlinear'] = OrderedDict({'count': vector_events,
                                                                                 'factor': {'cycle_count': vector_cycle_utilization,
                                                                                            'runtime': vector_cycle_utilization}})
        else:
            performance_dict['subevent']['vector_nonlinear'] = OrderedDict({'count': vector_events,
                                                                            'factor': {'cycle_count': vector_cycle_utilization,
                                                                                       'runtime': vector_cycle_utilization}})

    if architecture in ['tensor']:
        vector_events = tiles.m_tiles * tiles.n_total
        vector_events_utilization = tiles.m / (tiles.m_tiles * tiles.tile_m) * tiles.m_util
        vector_events *= vector_events_utilization
        vector_cycle_utilization = 1 / vector_events_utilization

        performance_dict['subevent']['vector_nonlinear'] = OrderedDict({'count': vector_events,
                                                                        'factor': {'cycle_count': vector_cycle_utilization,
                                                                                    'runtime': vector_cycle_utilization}})

    # unused events
    gemm_events = 0
    performance_dict['subevent']['gemm_nonlinear'] = OrderedDict({'count': gemm_events})

    return performance_dict

def sum_subevents(performance_dict_1: OrderedDict, performance_dict_2) -> OrderedDict:

    assert performance_dict_1.keys() == performance_dict_2.keys(), logger.error(f'performance_dicts must have the same keys to sum')
    assert performance_dict_1['subevent'].keys() == performance_dict_2['subevent'].keys(), logger.error(f'performance dict subevents must have the same keys to sum')

    sum_performance_dict = OrderedDict()

    for key, value in performance_dict_1.items():
        sum_performance_dict[key] = OrderedDict()
        for subkey, subvalue in value.items():
            if isinstance(subvalue, str):
                sum_performance_dict[key][subkey] = subvalue

    for subevent, subevent_dict in performance_dict_1['subevent'].items():
        sum_performance_dict['subevent'][subevent] = OrderedDict()

        
        for metric, value in subevent_dict.items():
            if not isinstance(value, dict):
                sum_performance_dict['subevent'][subevent][metric] = performance_dict_1['subevent'][subevent][metric] + performance_dict_2['subevent'][subevent][metric]
            else:
                performance_dict_1_count_average = 0 if sum_performance_dict['subevent'][subevent]['count'] == 0 else performance_dict_1['subevent'][subevent]['count'] / sum_performance_dict['subevent'][subevent]['count']
                performance_dict_2_count_average = 0 if sum_performance_dict['subevent'][subevent]['count'] == 0 else performance_dict_2['subevent'][subevent]['count'] / sum_performance_dict['subevent'][subevent]['count']
                sum_performance_dict['subevent'][subevent][metric] = OrderedDict()
                for submetric, subvalue in value.items():
                    sum_performance_dict['subevent'][subevent][metric][submetric] = (performance_dict_1['subevent'][subevent][metric][submetric] * performance_dict_1_count_average) + (performance_dict_2['subevent'][subevent][metric][submetric] * performance_dict_2_count_average)

    return sum_performance_dict

class TiledGEMM:
    """
    Class that tiles a GEMM of two input matrices, given the size of the matrices and size of each tile.
    Handles partial tiling, and computes memory sizes for matrices and tiles.
    This class simulates dimensions of matrices, tiles, and combination of dimensions, not a populated instantiated tile.
    """
    def __init__(self, batch, m, k, n, tile_m, tile_k, tile_n, m_k_bitwidth, k_n_bitwidth, m_n_bitwidth, array_width=None, array_height=None, array_depth=None):

        if 0 in (m, k, n, tile_m, tile_k, tile_n):
            self.is_valid = False
            return

        self.is_valid = True

        #initialize
        self.batch = batch
        self.m = m
        self.k = k
        self.n = n
        self.tile_m = min(tile_m, self.m)
        self.tile_k = min(tile_k, self.k)
        self.tile_n = min(tile_n, self.n)
        self.m_k_bitwidth = m_k_bitwidth
        self.k_n_bitwidth = k_n_bitwidth
        self.m_n_bitwidth = m_n_bitwidth
        self.k_util = self.tile_k / array_depth if array_depth else 1
        self.m_util = self.tile_m / array_width if array_width else 1
        self.n_util = self.tile_n / array_height if array_height else 1
        
        # Batch Dims
        self.m_total = self.m * self.batch
        self.k_total = self.k * self.batch
        self.n_total = self.n * self.batch

        #Total Tiles
        self.m_tiles = math.ceil(self.m / self.tile_m)
        self.k_tiles = math.ceil(self.k / self.tile_k)
        self.n_tiles = math.ceil(self.n / self.tile_n)
        
        # Batch Tiles
        self.m_total_tiles = self.m_tiles * self.batch
        self.k_total_tiles = self.k_tiles * self.batch
        self.n_total_tiles = self.n_tiles * self.batch        

        # GEMM Tiles
        self.m_k_matrix_tiles = self.m_tiles * self.k_tiles
        self.k_n_matrix_tiles = self.k_tiles * self.n_tiles
        self.m_n_matrix_tiles = self.m_tiles * self.n_tiles
        self.m_k_n_matrix_tiles = self.m_tiles * self.k_tiles * self.n_tiles

        # GEMM tiles across batch
        self.m_k_total_tiles = self.m_k_matrix_tiles * self.batch
        self.k_n_total_tiles = self.k_n_matrix_tiles * self.batch
        self.m_n_total_tiles = self.m_n_matrix_tiles * self.batch
        self.m_k_n_total_tiles = self.m_k_n_matrix_tiles * self.batch

        # Matrices
        self.m_k_matrix = self.m * self.k
        self.k_n_matrix = self.k * self.n
        self.m_n_matrix = self.m * self.n
        self.m_k_n_matrix = self.m * self.k * self.n
        self.m_k_total_matrix = self.m_k_matrix * self.batch
        self.k_n_total_matrix = self.k_n_matrix * self.batch
        self.m_n_total_matrix = self.m_n_matrix * self.batch
        self.m_k_n_total_matrix = self.m_k_n_matrix * self.batch
        
        # Full Tiles (non-fractional tiles)
        self.m_full_tiles = math.floor(self.m / self.tile_m)
        self.k_full_tiles = math.floor(self.k / self.tile_k)
        self.n_full_tiles = math.floor(self.n / self.tile_n)

        # GEMM full tiles
        self.m_full_k_full_matrix_tiles = self.m_full_tiles * self.k_full_tiles
        self.k_full_n_full_matrix_tiles = self.k_full_tiles * self.n_full_tiles
        self.m_full_n_full_matrix_tiles = self.m_full_tiles * self.n_full_tiles
        self.m_full_k_full_n_full_matrix_tiles = self.m_full_tiles * self.k_full_tiles * self.n_full_tiles

        # GEMM full tiles across batch
        self.m_full_k_full_total_tiles = self.m_full_k_full_matrix_tiles * self.batch
        self.k_full_n_full_total_tiles = self.k_full_n_full_matrix_tiles * self.batch
        self.m_full_n_full_total_tiles = self.m_full_n_full_matrix_tiles * self.batch
        self.m_full_k_full_n_full_total_tiles = self.m_full_k_full_n_full_matrix_tiles * self.batch
        
        #Partial Tiles (fractional tiles)
        # Partial tiles within tile dimensions, can only equal 1 or 0
        self.m_partial_tiles = self.m_tiles - self.m_full_tiles
        self.k_partial_tiles = self.k_tiles - self.k_full_tiles
        self.n_partial_tiles = self.n_tiles - self.n_full_tiles
        
        # Partial tiles across tile dimension, equal to full tile dimension if partial tile exists
        self.m_full_k_partial_tiles = self.m_full_tiles * self.k_partial_tiles
        self.m_partial_k_full_tiles = self.k_full_tiles * self.m_partial_tiles
        self.k_full_n_partial_tiles = self.k_full_tiles * self.n_partial_tiles
        self.k_partial_n_full_tiles = self.n_full_tiles * self.k_partial_tiles
        self.m_full_n_partial_tiles = self.m_full_tiles * self.n_partial_tiles
        self.m_partial_n_full_tiles = self.n_full_tiles * self.m_partial_tiles

        # Partial corner tile. Only exists when both dimensions have partial tiles across tile dimensions. Can only be 1 or 0.
        self.m_partial_k_partial_tile = self.m_partial_tiles * self.k_partial_tiles
        self.k_partial_n_partial_tile = self.k_partial_tiles * self.n_partial_tiles
        self.m_partial_n_partial_tile = self.m_partial_tiles * self.n_partial_tiles

        # Partial tiles of matrix (GEMMs)
        self.m_full_k_full_n_partial_tiles = self.m_full_tiles * self.k_full_n_partial_tiles
        self.m_full_k_partial_n_full_tiles = self.m_full_k_partial_tiles * self.k_partial_n_full_tiles
        self.m_full_k_partial_n_partial_tiles = self.m_full_k_partial_tiles * self.k_partial_n_partial_tile
        self.m_partial_k_full_n_full_tiles = self.m_partial_k_full_tiles * self.n_full_tiles
        self.m_partial_k_full_n_partial_tiles = self.m_partial_k_full_tiles * self.k_full_n_partial_tiles
        self.m_partial_k_partial_n_full_tiles = self.m_partial_k_partial_tile * self.k_partial_n_full_tiles
        self.m_partial_k_partial_n_partial_tiles = self.m_partial_tiles * self.k_partial_tiles * self.n_partial_tiles

        # Partial tiles across batch 
        self.m_full_k_partial_total_tiles = self.m_full_k_partial_tiles * self.batch
        self.m_partial_k_full_total_tiles = self.m_partial_k_full_tiles * self.batch
        self.k_full_n_partial_total_tiles = self.k_full_n_partial_tiles * self.batch
        self.k_partial_n_full_total_tiles = self.k_partial_n_full_tiles * self.batch
        self.m_full_n_partial_total_tiles = self.m_full_n_partial_tiles * self.batch
        self.m_partial_n_full_total_tiles = self.m_partial_n_full_tiles * self.batch
        self.m_partial_k_partial_total_tiles = self.m_partial_k_partial_tile * self.batch
        self.k_partial_n_partial_total_tiles = self.k_partial_n_partial_tile * self.batch
        self.m_partial_n_partial_total_tiles = self.m_partial_n_partial_tile * self.batch

        # Partial tiles across batch (GEMM)
        self.m_full_k_full_n_partial_total_tiles = self.m_full_k_full_n_partial_tiles * self.batch
        self.m_full_k_partial_n_full_total_tiles = self.m_full_k_partial_n_full_tiles * self.batch
        self.m_full_k_partial_n_partial_total_tiles = self.m_full_k_partial_n_partial_tiles * self.batch
        self.m_partial_k_full_n_full_total_tiles = self.m_partial_k_full_n_full_tiles * self.batch
        self.m_partial_k_full_n_partial_total_tiles = self.m_partial_k_full_n_partial_tiles * self.batch
        self.m_partial_k_partial_n_full_total_tiles = self.m_partial_k_partial_n_full_tiles * self.batch
        self.m_partial_k_partial_n_partial_total_tiles = self.m_partial_k_partial_n_partial_tiles * self.batch

        # Partial tile sizes.
        self.tile_m_partial = self.m % self.tile_m
        self.tile_k_partial = self.k % self.tile_k
        self.tile_n_partial = self.n % self.tile_n

        #memory Size
        # Matrix memory Sizes
        self.m_k_matrix_bits = self.m_k_bitwidth * self.m * self.k
        self.k_n_matrix_bits = self.k_n_bitwidth * self.k * self.n
        self.m_n_matrix_bits = self.m_n_bitwidth * self.m * self.n
        self.m_k_total_bits = self.m_k_matrix_bits * self.batch
        self.k_n_total_bits = self.k_n_matrix_bits * self.batch
        self.m_n_total_bits = self.m_n_matrix_bits * self.batch

        # Tile memory sizes
        # Full tile memory sizes
        self.m_full_k_full_tile_bits = self.m_k_bitwidth * self.tile_m * self.tile_k
        self.k_full_n_full_tile_bits = self.k_n_bitwidth * self.tile_k * self.tile_n
        self.m_full_n_full_tile_bits = self.m_n_bitwidth * self.tile_m * self.tile_n

        # Memory Size of all full tiles
        self.m_full_k_full_matrix_bits = self.m_full_k_full_tile_bits * self.m_full_k_full_matrix_tiles
        self.k_full_n_full_matrix_bits = self.k_full_n_full_tile_bits * self.k_full_n_full_matrix_tiles
        self.m_full_n_full_matrix_bits = self.m_full_n_full_tile_bits * self.m_full_n_full_matrix_tiles

        # Memory size of all full tiles across batch
        self.m_full_k_full_total_bits = self.m_full_k_full_matrix_bits * self.batch
        self.k_full_n_full_total_bits = self.k_full_n_full_matrix_bits * self.batch
        self.m_full_n_full_total_bits = self.m_full_n_full_matrix_bits * self.batch

        # Partial tile memory sizes
        self.m_partial_k_full_tile_bits = self.m_k_bitwidth * self.tile_m_partial * self.tile_k
        self.m_full_k_partial_tile_bits = self.m_k_bitwidth * self.tile_m * self.tile_k_partial
        self.k_partial_n_full_tile_bits = self.k_n_bitwidth * self.tile_k_partial * self.tile_n
        self.k_full_n_partial_tile_bits = self.k_n_bitwidth * self.tile_k * self.tile_n_partial
        self.m_partial_n_full_tile_bits = self.m_n_bitwidth * self.tile_m_partial * self.tile_n
        self.m_full_n_partial_tile_bits = self.m_n_bitwidth * self.tile_m * self.tile_n_partial
        self.m_partial_k_partial_tile_bits = self.m_k_bitwidth * self.tile_m_partial * self.tile_k_partial
        self.k_partial_n_partial_tile_bits = self.k_n_bitwidth * self.tile_k_partial * self.tile_n_partial
        self.m_partial_n_partial_tile_bits = self.m_n_bitwidth * self.tile_m_partial * self.tile_n_partial

        # Memory size of all partial tiles
        self.m_partial_k_full_matrix_bits = self.m_partial_k_full_tile_bits * self.m_partial_k_full_tiles
        self.m_full_k_partial_matrix_bits = self.m_full_k_partial_tile_bits * self.m_full_k_partial_tiles
        self.k_partial_n_full_matrix_bits = self.k_partial_n_full_tile_bits * self.k_partial_n_full_tiles
        self.k_full_n_partial_matrix_bits = self.k_full_n_partial_tile_bits * self.k_full_n_partial_tiles
        self.m_partial_n_full_matrix_bits = self.m_partial_n_full_tile_bits * self.m_partial_n_full_tiles
        self.m_full_n_partial_matrix_bits = self.m_full_n_partial_tile_bits * self.m_full_n_partial_tiles
        self.m_partial_k_partial_matrix_bits = self.m_partial_k_partial_tile_bits * self.m_partial_k_partial_tile
        self.k_partial_n_partial_matrix_bits = self.k_partial_n_partial_tile_bits * self.k_partial_n_partial_tile
        self.m_partial_n_partial_matrix_bits = self.m_partial_n_partial_tile_bits * self.m_partial_n_partial_tile

        # Memory size of all partial tiles across batch
        self.m_partial_k_full_total_bits = self.m_partial_k_full_matrix_bits * self.batch
        self.m_full_k_partial_total_bits = self.m_full_k_partial_matrix_bits * self.batch
        self.k_partial_n_full_total_bits = self.k_partial_n_full_matrix_bits * self.batch
        self.k_full_n_partial_total_bits = self.k_full_n_partial_matrix_bits * self.batch
        self.m_partial_n_full_total_bits = self.m_partial_n_full_matrix_bits * self.batch
        self.m_full_n_partial_total_bits = self.m_full_n_partial_matrix_bits * self.batch
        self.m_partial_k_partial_total_bits = self.m_partial_k_partial_matrix_bits * self.batch
        self.k_partial_n_partial_total_bits = self.k_partial_n_partial_matrix_bits * self.batch
        self.m_partial_n_partial_total_bits = self.m_partial_n_partial_matrix_bits * self.batch
        
class TiledMatrix:
    def __init__(self, batch, m, n, tile_m, tile_n, m_n_bitwidth, array_height=None):

        if 0 in (m, n, tile_m, tile_n):
            self.is_valid = False
            return

        self.is_valid = True

        #initialize
        self.batch = batch
        self.m = m
        self.n = n
        self.tile_m = min(tile_m, self.m)
        self.tile_n = min(tile_n, self.n)
        self.m_n_bitwidth = m_n_bitwidth
        self.m_util = self.tile_m / array_height if array_height else 1

        # Batch Dims
        self.m_total = self.m * self.batch
        self.n_total = self.n * self.batch

        #Total Tiles
        self.m_tiles = math.ceil(self.m / self.tile_m)
        self.n_tiles = math.ceil(self.n / self.tile_n)
        self.m_n_matrix_tiles = self.m_tiles * self.n_tiles
        self.m_n_total_tiles = self.m_n_matrix_tiles * self.batch
    
        # Batch Tiles
        self.m_total_tiles = self.m_tiles * self.batch
        self.n_total_tiles = self.n_tiles * self.batch

        #Full Tiles (non-fractional tiles)
        self.m_full_tiles = math.floor(self.m / self.tile_m)
        self.n_full_tiles = math.floor(self.n / self.tile_n)
        self.m_full_n_full_matrix_tiles = self.m_full_tiles * self.n_full_tiles
        self.m_full_n_full_total_tiles = self.m_full_n_full_matrix_tiles * self.batch

        # Matrices
        self.m_n_matrix = self.m * self.n
        self.m_n_total_matrix = self.m_n_matrix * self.batch

        # Partial Tiles (fractional tiles)
        # Partial tiles within tile dimensions, can only equal 1 or 0
        self.m_partial_tiles = self.m_tiles - self.m_full_tiles
        self.n_partial_tiles = self.n_tiles - self.n_full_tiles
        
        # Partial tiles across tile dimension, equal to full tile dimension if partial tile exists
        self.m_full_n_partial_tiles = self.m_full_tiles * self.n_partial_tiles
        self.m_partial_n_full_tiles = self.n_full_tiles * self.m_partial_tiles

        # Partial corner tile. Only exists when both dimensions have partial tiles across tile dimensions. Can only be 1 or 0.
        self.m_partial_n_partial_tile = self.m_partial_tiles * self.n_partial_tiles

        # Partial tiles across batch 
        self.m_full_n_partial_total_tiles = self.m_full_n_partial_tiles * self.batch
        self.m_partial_n_full_total_tiles = self.m_partial_n_full_tiles * self.batch
        self.m_partial_n_partial_total_tiles = self.m_partial_n_partial_tile * self.batch

        # Partial tile sizes.
        self.tile_m_partial = self.m % self.tile_m
        self.tile_n_partial = self.n % self.tile_n

        # memory Size
        # Matrix memory Sizes
        self.m_n_matrix_bits = self.m_n_bitwidth * self.m * self.n
        self.m_n_total_bits = self.m_n_matrix_bits * self.batch

        # Tile memory sizes
        # Full tile memory sizes
        self.m_full_n_full_tile_bits = self.m_n_bitwidth * self.tile_m * self.tile_n

        # Memory Size of all full tiles
        self.m_full_n_full_matrix_bits = self.m_full_n_full_tile_bits * self.m_full_n_full_matrix_tiles

        # Memory size of all full tiles across batch
        self.m_full_n_full_total_bits = self.m_full_n_full_matrix_bits * self.batch

        # Partial tile memory sizes
        self.m_partial_n_full_tile_bits = self.m_n_bitwidth * self.tile_m_partial * self.tile_n
        self.m_full_n_partial_tile_bits = self.m_n_bitwidth * self.tile_m * self.tile_n_partial
        self.m_partial_n_partial_tile_bits = self.m_n_bitwidth * self.tile_m_partial * self.tile_n_partial

        # Memory size of all partial tiles
        self.m_partial_n_full_matrix_bits = self.m_partial_n_full_tile_bits * self.m_partial_n_full_tiles
        self.m_full_n_partial_matrix_bits = self.m_full_n_partial_tile_bits * self.m_full_n_partial_tiles
        self.m_partial_n_partial_matrix_bits = self.m_partial_n_partial_tile_bits * self.m_partial_n_partial_tile

        # Memory size of all partial tiles across batch
        self.m_partial_n_full_total_bits = self.m_partial_n_full_matrix_bits * self.batch
        self.m_full_n_partial_total_bits = self.m_full_n_partial_matrix_bits * self.batch
        self.m_partial_n_partial_total_bits = self.m_partial_n_partial_matrix_bits * self.batch

def router_gemm_events(tiles: TiledGEMM, offchip_memory_events_dict: OrderedDict, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates events from router to onchip memory for GEMM operations.
    Needs dram <-> sram performance events passed.
    """
    performance_dict = OrderedDict()

    # Retrieve dicts
    router_height = architecture_dict['irouter']['instance'][0]
    router_width = architecture_dict['irouter']['instance'][1]
    router_dim = router_height * router_width
    scheduling = workload_dict['noc_stationary']
    osram_width = architecture_dict['osram']['query']['width']
    isram_width = architecture_dict['isram']['query']['width']
    wsram_width = architecture_dict['wsram']['query']['width']

    # adjust based on scheduling
    if scheduling == 'os':
        m_k_tiles = tiles.m_k_n_total_tiles
        k_n_tiles = tiles.m_k_n_total_tiles
        m_n_tiles = tiles.m_n_total_tiles
    elif scheduling == 'is':
        m_k_tiles = tiles.m_k_total_tiles
        k_n_tiles = tiles.m_k_n_total_tiles
        m_n_tiles = tiles.m_k_n_total_tiles
    elif scheduling == 'ws':
        m_k_tiles = tiles.m_k_n_total_tiles
        k_n_tiles = tiles.k_n_total_tiles
        m_n_tiles = tiles.m_k_n_total_tiles

    # calculate average router event across all mappings
    m_k_full_router_events = math.floor(m_k_tiles / router_dim)
    k_n_full_router_events = math.floor(k_n_tiles / router_dim)
    m_n_full_router_events = math.floor(m_n_tiles / router_dim)

    m_k_partial_router_events = 1 if m_k_tiles % router_dim > 0 else 0
    k_n_partial_router_events = 1 if k_n_tiles % router_dim > 0 else 0
    m_n_partial_router_events = 1 if m_n_tiles % router_dim > 0 else 0

    m_k_mappings = m_k_full_router_events + m_k_partial_router_events
    k_n_mappings = k_n_full_router_events + k_n_partial_router_events
    m_n_mappings = m_n_full_router_events + m_n_partial_router_events

    m_k_partial_router_height = round(math.sqrt(m_k_partial_router_events))
    k_n_partial_router_height = round(math.sqrt(k_n_partial_router_events))
    m_n_partial_router_height = round(math.sqrt(m_n_partial_router_events))

    m_k_average_router_events = ((router_height * m_k_full_router_events) + (m_k_partial_router_height * m_k_partial_router_events)) / m_k_mappings
    k_n_average_router_events = ((router_height * k_n_full_router_events) + (k_n_partial_router_height * k_n_partial_router_events)) / k_n_mappings
    m_n_average_router_events = ((router_height * m_n_full_router_events) + (m_n_partial_router_height * m_n_partial_router_events)) / m_n_mappings

    irouter_count = m_k_average_router_events * offchip_memory_events_dict['subevent']['isram_offchip_writes']['count'] * osram_width
    wrouter_count = k_n_average_router_events * offchip_memory_events_dict['subevent']['wsram_offchip_writes']['count'] * isram_width
    orouter_count = m_n_average_router_events * offchip_memory_events_dict['subevent']['osram_offchip_reads']['count'] * wsram_width

    irouter_dict = OrderedDict({'count': irouter_count})
    wrouter_dict = OrderedDict({'count': wrouter_count})
    orouter_dict = OrderedDict({'count': orouter_count})

    performance_dict['subevent'] = OrderedDict({
        'irouter_mapping': irouter_dict,
        'wrouter_mapping': wrouter_dict,
        'orouter_mapping': orouter_dict
    })

    return performance_dict

def router_nonlinear_events(tiles: TiledMatrix, offchip_memory_events_dict: OrderedDict, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates events from router to onchip memory for GEMM operations.
    Needs dram <-> sram performance events passed within performance_dict.
    """
    offchip_memory_events_dict = offchip_memory_events_dict['subevent']
    performance_dict = OrderedDict()

    # Retrieve dicts
    router_height = architecture_dict['irouter']['instance'][0]
    router_width = architecture_dict['irouter']['instance'][1]
    router_dim = router_height * router_width
    architecture = workload_dict['architecture']

    # calculate average router event across all mappings
    m_n_full_router_events = math.floor(tiles.m_n_total_tiles / router_dim)
    m_n_partial_router_events = 1 if tiles.m_n_total_tiles % router_dim > 0 else 0
    m_n_mappings = m_n_full_router_events + m_n_partial_router_events

    m_n_partial_router_height = round(math.sqrt(m_n_partial_router_events))
    m_n_average_router_events = ((router_height * m_n_full_router_events) + (m_n_partial_router_height * m_n_partial_router_events)) / m_n_mappings

    irouter_average_events = m_n_average_router_events if tiles.m_n_total_tiles < router_dim else router_height

    if architecture == 'mugi':
        irouter_count = irouter_average_events * offchip_memory_events_dict['isram_offchip_writes']['count']
        wrouter_count = m_n_average_router_events * offchip_memory_events_dict['wsram_offchip_writes']['count']
        orouter_count = m_n_average_router_events * (offchip_memory_events_dict['osram_offchip_reads']['count'] + offchip_memory_events_dict['osram_offchip_writes']['count'])
    else:
        irouter_count = 0
        wrouter_count = 0
        orouter_count = m_n_average_router_events * (offchip_memory_events_dict['osram_offchip_reads']['count'] + offchip_memory_events_dict['osram_offchip_writes']['count'])

    irouter_dict = OrderedDict({'count': irouter_count})
    wrouter_dict = OrderedDict({'count': wrouter_count})
    orouter_dict = OrderedDict({'count': orouter_count})

    performance_dict['subevent'] = OrderedDict({
        'irouter_mapping': irouter_dict,
        'wrouter_mapping': wrouter_dict,
        'orouter_mapping': orouter_dict
    })

    return performance_dict