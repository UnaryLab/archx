from archx.utils import get_prod
import math, inspect
from collections import OrderedDict
from loguru import logger

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