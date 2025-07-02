from collections import OrderedDict
from loguru import logger
import math

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
                performance_dict_1_count_average = performance_dict_1['subevent'][subevent]['count'] / sum_performance_dict['subevent'][subevent]['count']
                performance_dict_2_count_average = performance_dict_2['subevent'][subevent]['count'] / sum_performance_dict['subevent'][subevent]['count']
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
    def __init__(self, batch, m, k, n, tile_m, tile_k, tile_n, m_k_bitwidth, k_n_bitwidth, m_n_bitwidth, array_width=None, array_height=None):

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