from zoo.llm.common.performance.utils import TiledGEMM, TiledMatrix
import math
from collections import OrderedDict

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
