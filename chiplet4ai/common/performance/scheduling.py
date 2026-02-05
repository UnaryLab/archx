from chiplet4ai.common.performance import memory_scheduling
from collections import OrderedDict
from collections import OrderedDict
import math

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
    #performance_dict['subevent'].update(offchip_memory_events_dict['subevent'])
    
    # only compute router events if multinode configuration
    # average router events, returns dictionary containing average router events for input, weight, and output memory events
    # average event is not total, but router events per one sram event
    # needs offchip memory events to calculate router events
    # if router:
    #     router_event_dict = router_scheduling.router_gemm_events(tiles=offchip_tiles,
    #                                                                    offchip_memory_events_dict=offchip_memory_events_dict,
    #                                                                    architecture_dict=architecture_dict,
    #                                                                    workload_dict=workload_dict)
    #     performance_dict['subevent'].update(router_event_dict['subevent'])
    
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
    array_events_dict = gemm_events(tiling=onchip_tiles_list,
                                                      architecture_dict=architecture_dict,
                                                      workload_dict=workload_dict,
                                                      performance_dict=performance_dict)
    performance_dict['subevent'].update(array_events_dict['subevent'])
        
    performance_dict = performance_count_to_int(performance_dict)
    return performance_dict

# def nonlinear_mapping(mapping_dict: OrderedDict, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    
#     router = True if 'irouter' in architecture_dict else False
#     performance_dict = OrderedDict()
#     performance_dict['subevent'] = OrderedDict()

#     # offchip scheduling, returns Tilednonlinear object
#     offchip_tiles = memory_scheduling.offchip_nonlinear_scheduling(batch=mapping_dict['batch'],
#                                                                    m=mapping_dict['m'],
#                                                                    n=mapping_dict['n'],
#                                                                    architecture_dict=architecture_dict,
#                                                                    workload_dict=workload_dict)
    

#     # offchip memory events, returns dictionary containing input, weight, and output sram events and dram events
#     offchip_memory_events_dict = memory_scheduling.offchip_nonlinear_events(tiles=offchip_tiles,
#                                                                             architecture_dict=architecture_dict,
#                                                                             workload_dict=workload_dict)
#     performance_dict['subevent'].update(offchip_memory_events_dict['subevent'])
    
#     # only compute router events if multinode configuration
#     # average router events, returns dictionary containing average router events for input, weight, and output memory events
#     # average event is not total, but router events per one sram event
#     # needs offchip memory events to calculate router events
#     if router:
#         router_event_dict = router_scheduling.router_nonlinear_events(tiles=offchip_tiles,
#                                                                       offchip_memory_events_dict=offchip_memory_events_dict,
#                                                                       architecture_dict=architecture_dict,
#                                                                       workload_dict=workload_dict)
#         performance_dict['subevent'].update(router_event_dict['subevent'])
    
#     # onchip scheduling, returns list of Tilednonlinear objects. List is for each partial tiling configuration
#     onchip_tiles_list = memory_scheduling.onchip_nonlinear_scheduling(m=mapping_dict['m'],
#                                                                       n=mapping_dict['n'],
#                                                                       offchip_tiles=offchip_tiles,
#                                                                       architecture_dict=architecture_dict,
#                                                                       workload_dict=workload_dict)

#     # onchip memory events, returns dictionary containing input, weight, and output sram events
#     onchip_memory_events_dict = memory_scheduling.onchip_nonlinear_events(function=mapping_dict['function'],
#                                                                           onchip_tiling=onchip_tiles_list,
#                                                                           architecture_dict=architecture_dict,
#                                                                           workload_dict=workload_dict)
#     performance_dict['subevent'].update(onchip_memory_events_dict['subevent'])
#     # array
#     array_events_dict = nonlinear_events(function=mapping_dict['function'],
#                                                            tiling=onchip_tiles_list,
#                                                            architecture_dict=architecture_dict,
#                                                            workload_dict=workload_dict)
#     performance_dict['subevent'].update(array_events_dict['subevent'])

#     performance_dict = performance_count_to_int(performance_dict)
#     return performance_dict

def mapping(mapping_dict: OrderedDict,architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:
    if mapping_dict['event'] == 'gemm':
        performance_dict = gemm_mapping(mapping_dict=mapping_dict,
                                        architecture_dict=architecture_dict,
                                        workload_dict=workload_dict)
    # elif mapping_dict['event'] == 'nonlinear':
    #     performance_dict = nonlinear_mapping(mapping_dict=mapping_dict,
    #                                          architecture_dict=architecture_dict,
    #                                          workload_dict=workload_dict)
        
    return performance_dict
   
def gemm_events(tiling: list[TiledGEMM], architecture_dict: OrderedDict, workload_dict: OrderedDict, performance_dict: OrderedDict) -> OrderedDict:

    performance_dict = None

    for i, tiles in enumerate(tiling):
        if tiles.is_valid:
            if performance_dict is None:
                performance_dict = gemm_tile_events(tiles=tiles, architecture_dict=architecture_dict, workload_dict=workload_dict)
            else:
                performance_dict = sum_subevents(performance_dict, gemm_tile_events(tiles=tiles, architecture_dict=architecture_dict, workload_dict=workload_dict))

    return performance_dict

def gemm_tile_events(tiles: TiledGEMM, architecture_dict: OrderedDict, workload_dict: OrderedDict) -> OrderedDict:

    performance_dict = OrderedDict()
    performance_dict['subevent'] = OrderedDict()

    # load input
    
    input_events = tiles.m_n_total_tiles * tiles.k
    input_event_utilization = (tiles.m / (tiles.m_tiles * tiles.tile_m)) * tiles.m_util
    input_events *= input_event_utilization
    cycle_count_utilization = 1 / input_event_utilization
    # performance_dict['subevent']['input_gemm'] = OrderedDict({'count': input_events,
    #                                                           'factor': {'cycle_count': cycle_count_utilization,
    #                                                                      'runtime': cycle_count_utilization}})
    
    weight_events = tiles.m_total_tiles * tiles.k
    weight_event_utilization = (tiles.n / (tiles.n_tiles * tiles.tile_n)) * tiles.n_util
    weight_events *= weight_event_utilization
    weight_cycle_utilization = 1 / weight_event_utilization
    performance_dict['subevent']['weight_mapping'] = OrderedDict({'count': weight_events,
                                                               'factor': {'cycle_count': weight_cycle_utilization,
                                                                          'runtime': weight_cycle_utilization}})

    array_events = tiles.m_n_total_tiles * tiles.k
    array_events_utilization = input_event_utilization * weight_event_utilization
    array_events *= array_events_utilization
    array_cycle_utilization = 1 / array_events_utilization
    performance_dict['subevent']['array_mapping'] = OrderedDict({'count': array_events,
                                                              'factor': {'cycle_count': array_cycle_utilization,
                                                                         'runtime': array_cycle_utilization}})

    return performance_dict

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