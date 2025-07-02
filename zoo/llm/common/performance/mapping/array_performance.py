from zoo.llm.common.performance.utils import TiledGEMM, TiledMatrix, sum_subevents
from archx.utils import get_prod
import math
from collections import OrderedDict

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
    array_events = tiles.m_n_total_tiles * tiles.k
    array_events_utilization = input_event_utilization * weight_event_utilization
    array_events *= array_events_utilization
    array_cycle_utilization = et_cycles / cycles if et_cycles is not None else 1
    array_cycle_utilization *= 1 / array_events_utilization
    performance_dict['subevent']['array_gemm'] = OrderedDict({'count': array_events,
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
            array_events_utilization = weight_events_utilization
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
    
    # unused events
    gemm_events = 0
    performance_dict['subevent']['gemm_nonlinear'] = OrderedDict({'count': gemm_events})

    return performance_dict