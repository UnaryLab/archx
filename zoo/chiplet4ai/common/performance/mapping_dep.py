import math


def gemm_ws_model(
    M, K, N,
    array_m, array_n,
    sram_ifmap, sram_weight, sram_ofmap,
    bytes_per_element=2, frequency=1e9, dram_bandwidth=100e9,
):
    # ---------------------------------------------------------
    # 1. Tile dimensions
    # ---------------------------------------------------------

    Mt = min(M, array_m)
    Nt = min(N, array_n)

    # Weight tile = Kt x Nt
    # IFMAP tile  = Mt x Kt
    Kt_ifmap = sram_ifmap / Mt
    Kt_weight = sram_weight / Nt
    Kt = min(K, Kt_ifmap, Kt_weight)

    if Kt <= 0:
        raise ValueError("SRAM is too small for one element.")

    # ---------------------------------------------------------
    # 2. Number of tiles / folding
    # ---------------------------------------------------------

    num_m_tiles = math.ceil(M / Mt)
    num_n_tiles = math.ceil(N / Nt)

    # Fractional K tiles captures partial final fold
    num_k_tiles = K / Kt

    num_mappings = num_m_tiles * num_n_tiles * math.ceil(num_k_tiles)

    # ---------------------------------------------------------
    # 3. Utilization
    # ---------------------------------------------------------

    m_utilization = Mt / array_m
    n_utilization = Nt / array_n

    # Utilization of the final K fold
    k_utilization = min(1, K / (math.floor(num_k_tiles) * Kt)) if num_k_tiles > 1 else 1

    array_utilization = m_utilization * n_utilization

    # ---------------------------------------------------------
    # 4. Compute cycles
    # ---------------------------------------------------------

    cycles_per_full_mapping = Mt + Nt + Kt
    compute_cycles = num_m_tiles * num_n_tiles * num_k_tiles * cycles_per_full_mapping

    # ---------------------------------------------------------
    # 5. DRAM traffic
    #
    # WEIGHT STATIONARY:
    #
    # Weights are reused across M tiles.
    #
    # IFMAP:
    #   Each M/K tile is needed for every N tile.
    #
    # WEIGHTS:
    #   Each K/N tile is reused across M tiles.
    #
    # OFMAP:
    #   Written once for every M/N output tile.
    # ---------------------------------------------------------

    # IFMAP: Mt x Kt
    dram_ifmap_elements = num_m_tiles * num_n_tiles * num_k_tiles * Mt * Kt

    # WEIGHTS: Kt x Nt
    dram_weight_elements = num_n_tiles * num_k_tiles * Kt * Nt

    # OFMAP: Mt x Nt
    dram_ofmap_elements = num_m_tiles * num_n_tiles * Mt * Nt

    # ---------------------------------------------------------
    # 6. Partial tile correction
    # ---------------------------------------------------------

    last_k = K % Kt

    if last_k != 0:
        dram_ifmap_elements -= num_m_tiles * num_n_tiles * (Kt - last_k) * Mt
        dram_weight_elements -= num_n_tiles * (Kt - last_k) * Nt

    # ---------------------------------------------------------
    # 7. Convert to bytes
    # ---------------------------------------------------------

    dram_ifmap_bytes = dram_ifmap_elements * bytes_per_element
    dram_weight_bytes = dram_weight_elements * bytes_per_element
    dram_ofmap_bytes = dram_ofmap_elements * bytes_per_element
    dram_bytes = dram_ifmap_bytes + dram_weight_bytes + dram_ofmap_bytes

    # ---------------------------------------------------------
    # 8. Required DRAM bandwidth
    # ---------------------------------------------------------

    compute_time = compute_cycles / frequency
    required_bandwidth = dram_bytes / compute_time

    # ---------------------------------------------------------
    # 9. Stall cycles
    # ---------------------------------------------------------

    memory_time = dram_bytes / dram_bandwidth
    total_time = max(compute_time, memory_time)
    total_cycles = total_time * frequency
    stall_cycles = max(0, total_cycles - compute_cycles)

    return {
        # Tiles
        "tile_m": Mt,
        "tile_k": Kt,
        "tile_n": Nt,

        # Mapping / folding
        "num_m_tiles": num_m_tiles,
        "num_k_tiles": num_k_tiles,
        "num_n_tiles": num_n_tiles,
        "num_mappings": num_mappings,

        # Utilization
        "m_utilization": m_utilization,
        "n_utilization": n_utilization,
        "k_utilization": k_utilization,
        "array_utilization": array_utilization,

        # Cycles
        "compute_cycles": compute_cycles,
        "stall_cycles": stall_cycles,
        "total_cycles": total_cycles,

        # DRAM traffic
        "dram_ifmap_bytes": dram_ifmap_bytes,
        "dram_weight_bytes": dram_weight_bytes,
        "dram_ofmap_bytes": dram_ofmap_bytes,
        "dram_bytes": dram_bytes,

        # Bandwidth
        "required_dram_bandwidth": required_bandwidth,

        # Time
        "compute_time": compute_time,
        "execution_time": total_time,
    }