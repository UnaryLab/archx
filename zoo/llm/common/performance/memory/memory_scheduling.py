from zoo.llm.common.performance.utils import TiledGEMM, TiledMatrix
import math
from collections import OrderedDict

def offchip_gemm_scheduling(batch, m, k, n, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> TiledGEMM:
    """
    Sets tiling for dram <-> sram for single-node and multi-node configurations
    """
    # Retrieve dicts
    isram_dict = architecture_dict['isram']
    wsram_dict = architecture_dict['wsram']
    osram_dict = architecture_dict['osram']
    input_bitwidth = workload_dict['activation_bitwidth']
    weight_bitwidth = workload_dict['weight_bitwidth']
    output_bitwidth = workload_dict['activation_bitwidth']
    stationary = workload_dict['noc_stationary']
    array_width = architecture_dict['ififo']['instance'][-1]
    array_height = architecture_dict['wfifo']['instance'][-1]

    # SRAM configurations
    # divide banks by to buffer
    isram_bank = isram_dict['query']['bank'] / 2
    wsram_bank = wsram_dict['query']['bank'] / 2
    osram_bank = osram_dict['query']['bank'] / 2

    isram_width = isram_dict['query']['width']
    wsram_width = wsram_dict['query']['width']
    osram_width = osram_dict['query']['width']

    isram_depth = isram_dict['query']['depth']
    wsram_depth = wsram_dict['query']['depth']
    osram_depth = osram_dict['query']['depth']

    isram_size = isram_bank * isram_width * isram_depth
    wsram_size = wsram_bank * wsram_width * wsram_depth
    osram_size = osram_bank * osram_width * osram_depth

    isram_elements = isram_size / input_bitwidth
    wsram_elements = wsram_size / weight_bitwidth
    osram_elements = osram_size / output_bitwidth

    # number of nodes
    nodes = isram_dict['instance'][0] * isram_dict['instance'][1] if 'irouter' in architecture_dict else 1

    if stationary == 'os':
        # Output stationary scheduling
        # initialize m tile size to m (in llms, very oftem batch size or smaller)
        # initialize n tile size to maximum size that fully utalizes noc
        tile_m = min(m, array_width)
        tile_n = ((n * batch) / nodes)

        # if tile_n does not fit in osram, reduce tile_n
        while(tile_n * tile_m > osram_elements):
            tile_n /= 2

        # if small tile n, and is smaller than array, increase tile_n
        while((tile_n * tile_m * 2 < osram_elements) and (tile_n < array_height)):
            tile_n *= 2

        # if small tile n, and can increase m, increase tile_m
        # multiply by 2 to check if you can increase tile_m
        while((tile_m * tile_n * 2 < osram_elements) and (tile_m < m)):
            tile_m *= 2

        # initialize k tile size to maximum size that fully utalizes wsram
        tile_k = wsram_elements / tile_n

        # if tile_k does not fit in isram, reduce tile_k
        while(tile_k * tile_m > isram_elements):
            tile_k /= 2

    elif stationary == 'ws':
        # Weight stationary scheduling
        # initialize m tile size to m (in llms, very oftem batch size or smaller)
        # initialize k tile size to maximum size that fully utalizes noc
        tile_m = min(m, array_width)
        tile_k = ((k * batch) / nodes)

        # if tile_k does not fit in isram, reduce tile_k
        while(tile_k * tile_m > isram_elements):
            tile_k /= 2

        # initialize n tile size to maximum size that fully utalizes wsram
        tile_n = wsram_elements / tile_k

        # if tile_n does not fit in osram or wsram, reduce tile_n
        while(tile_n > osram_elements / tile_m):
            tile_n /= 2

        # if small tile k, and can increase m, increase tile_m
        # multiply by 2 to check if you can increase tile_m
        while((tile_m * tile_k * 2 < isram_elements) and (tile_m < m)):
            tile_m *= 2

    elif stationary == 'is':
        # Input stationary scheduling
        # initialize k tile size to m(quantized dim, smaller data size)
        # initialize m tile size to maximum size that fully utalizes noc
        tile_k = min(k, array_height)
        tile_m = ((m * batch) / nodes)

        # if tile_m does not fit in isram, reduce tile_m
        while(tile_m * tile_k > isram_elements):
            tile_m /= 2

        # initialize n tile size to maximum size that fully utalizes osram
        tile_n = osram_elements / tile_m
        
        # if tile_n does not fit in wsram, reduce tile_n
        while(tile_n * tile_k > wsram_elements):
            tile_n /= 2

        # if small tile k, and can increase k, increase tile_k
        # multiply by 2 to check if you can increase tile_k
        while((tile_k * tile_m * 2 < isram_elements) and (tile_k < k)):
            tile_k *= 2

    if tile_m * tile_k > isram_elements:
        raise ValueError('Tile size exceeds isram size')
    if tile_k * tile_n > wsram_elements:
        raise ValueError('Tile size exceeds wsram size')
    if tile_m * tile_n > osram_elements:
        raise ValueError('Tile size exceeds osram size')

    tile_m = 2 ** math.ceil(math.log2(tile_m))
    tile_k = 2 ** math.ceil(math.log2(tile_k))
    tile_n = 2 ** math.ceil(math.log2(tile_n))

    tile_m = int(tile_m)
    tile_k = int(tile_k)
    tile_n = int(tile_n)

    tiles = TiledGEMM(batch=batch, m=m, k=k, n=n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=input_bitwidth, k_n_bitwidth=weight_bitwidth, m_n_bitwidth=output_bitwidth)
    if not tiles.is_valid:
        raise ValueError('Invalid tiling configuration')
        
    return tiles

def offchip_gemm_events(tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    """
    Estimates dram and sram reads and writes to from offchip memory to onchip memory for GEMM operations
    """
    # Retrieve dicts
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
    dram_input_reads_dict = OrderedDict({'count': m_k_events})
    dram_weight_reads_dict = OrderedDict({'count': k_n_events})
    dram_output_reads_dict = OrderedDict({'count': 0})
    dram_output_writes_dict = OrderedDict({'count': m_n_events})

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

def onchip_gemm_scheduling(m, k, n, tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> list[TiledGEMM]:

    # Retrieve dicts
    array_height = architecture_dict['wfifo']['instance'][-1]
    array_width = architecture_dict['ififo']['instance'][-1]

    tile_m = min(array_width, m)
    tile_k = min(array_height, k)
    tile_n = min(array_height, n)

    # Retrieve Reads/Writes for all possible tiling configurations
    tiling_configurations = []
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_full_n_full_total_tiles, m=tiles.tile_m, k=tiles.tile_k, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                                      # m full k full n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_full_n_partial_total_tiles, m=tiles.tile_m, k=tiles.tile_k, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                           # m full k full n partial
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_partial_n_full_total_tiles, m=tiles.tile_m, k=tiles.tile_k_partial, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                           # m full k partial n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_full_k_partial_n_partial_total_tiles, m=tiles.tile_m, k=tiles.tile_k_partial, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                # m full k partial n partial
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_full_n_full_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                           # m partial k full n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_full_n_partial_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                # m partial k full n partial
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_partial_n_full_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k_partial, n=tiles.tile_n, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))                # m partial k partial n full
    tiling_configurations.append(TiledGEMM(batch=tiles.m_partial_k_partial_n_partial_total_tiles, m=tiles.tile_m_partial, k=tiles.tile_k_partial, n=tiles.tile_n_partial, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, m_k_bitwidth=tiles.m_k_bitwidth, k_n_bitwidth=tiles.k_n_bitwidth, m_n_bitwidth=tiles.m_n_bitwidth, array_width=array_width, array_height=array_height))     # m partial k partial n partial
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
    wsram_dict = architecture_dict['wsram']
    osram_dict = architecture_dict['osram']
    input_bitwidth = workload_dict['activation_bitwidth']
    output_bitwidth = workload_dict['activation_bitwidth']
    array_width = architecture_dict['ififo']['instance'][-1]
    array_height = architecture_dict['wfifo']['instance'][-1]
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

    wfifo_width = architecture_dict['wfifo']['instance'][-1]
    ofifo_width = architecture_dict['ofifo']['instance'][-1]
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