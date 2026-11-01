"""Maps one GEMM or nonlinear layer onto the Mugi memory hierarchy and array.

A layer is tiled twice: dram <-> node srams (offchip, fixed noc tiles) and node srams <->
array (onchip, array-sized tiles). Each step returns subevent counts for the leaf
software event that called it.
"""

import math
from collections import OrderedDict

from archx.utils import get_prod

DRAM_WIDTH = 64  # bits per dram transfer


def sum_subevents(a: OrderedDict, b: OrderedDict) -> OrderedDict:
    """Sum the counts of two performance dicts; each factor becomes the count-weighted mean."""
    assert a["subevent"].keys() == b["subevent"].keys()
    total = OrderedDict()
    for name, sub_a in a["subevent"].items():
        sub_b = b["subevent"][name]
        count = sub_a["count"] + sub_b["count"]
        total[name] = OrderedDict({"count": count})
        if "factor" in sub_a:
            weight_a = 0 if count == 0 else sub_a["count"] / count
            weight_b = 0 if count == 0 else sub_b["count"] / count
            total[name]["factor"] = OrderedDict(
                {
                    k: (sub_a["factor"][k] * weight_a) + (sub_b["factor"][k] * weight_b)
                    for k in sub_a["factor"]
                }
            )
    return OrderedDict({"subevent": total})


class TiledGEMM:
    """Tile counts and tile sizes (in bits) of a batch of (m x k) @ (k x n) GEMMs,
    split into full tiles and the partial tiles at the m, k and n edges."""

    def __init__(
        self,
        batch,
        m,
        k,
        n,
        tile_m,
        tile_k,
        tile_n,
        m_k_bitwidth,
        k_n_bitwidth,
        m_n_bitwidth,
        array_width=None,
        array_height=None,
    ):
        if 0 in (m, k, n, tile_m, tile_k, tile_n):
            self.is_valid = False
            return
        self.is_valid = True

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

        self.m_total = self.m * self.batch

        self.m_tiles = math.ceil(self.m / self.tile_m)
        self.k_tiles = math.ceil(self.k / self.tile_k)
        self.n_tiles = math.ceil(self.n / self.tile_n)
        self.m_total_tiles = self.m_tiles * self.batch
        self.n_total_tiles = self.n_tiles * self.batch

        self.m_k_total_tiles = self.m_tiles * self.k_tiles * self.batch
        self.k_n_total_tiles = self.k_tiles * self.n_tiles * self.batch
        self.m_n_total_tiles = self.m_tiles * self.n_tiles * self.batch
        self.m_k_n_total_tiles = self.m_tiles * self.k_tiles * self.n_tiles * self.batch
        self.m_n_total_matrix = self.m * self.n * self.batch

        # full (non-fractional) tiles
        m_full = math.floor(self.m / self.tile_m)
        k_full = math.floor(self.k / self.tile_k)
        n_full = math.floor(self.n / self.tile_n)
        self.m_full_k_full_total_tiles = m_full * k_full * self.batch
        self.k_full_n_full_total_tiles = k_full * n_full * self.batch
        self.m_full_n_full_total_tiles = m_full * n_full * self.batch
        self.m_full_k_full_n_full_total_tiles = m_full * k_full * n_full * self.batch

        # partial tiles at each edge: 1 if the dimension has a partial tile, else 0
        m_partial = self.m_tiles - m_full
        k_partial = self.k_tiles - k_full
        n_partial = self.n_tiles - n_full
        m_full_k_partial = m_full * k_partial
        m_partial_k_full = k_full * m_partial
        k_full_n_partial = k_full * n_partial
        k_partial_n_full = n_full * k_partial
        m_full_n_partial = m_full * n_partial
        m_partial_n_full = n_full * m_partial
        m_partial_k_partial = m_partial * k_partial
        k_partial_n_partial = k_partial * n_partial
        m_partial_n_partial = m_partial * n_partial

        self.m_full_k_partial_total_tiles = m_full_k_partial * self.batch
        self.m_partial_k_full_total_tiles = m_partial_k_full * self.batch
        self.k_full_n_partial_total_tiles = k_full_n_partial * self.batch
        self.k_partial_n_full_total_tiles = k_partial_n_full * self.batch
        self.m_full_n_partial_total_tiles = m_full_n_partial * self.batch
        self.m_partial_n_full_total_tiles = m_partial_n_full * self.batch
        self.m_partial_k_partial_total_tiles = m_partial_k_partial * self.batch
        self.k_partial_n_partial_total_tiles = k_partial_n_partial * self.batch
        self.m_partial_n_partial_total_tiles = m_partial_n_partial * self.batch

        # partial GEMM tiles across batch
        self.m_full_k_full_n_partial_total_tiles = (
            m_full * k_full_n_partial * self.batch
        )
        self.m_full_k_partial_n_full_total_tiles = (
            m_full_k_partial * k_partial_n_full * self.batch
        )
        self.m_full_k_partial_n_partial_total_tiles = (
            m_full_k_partial * k_partial_n_partial * self.batch
        )
        self.m_partial_k_full_n_full_total_tiles = (
            m_partial_k_full * n_full * self.batch
        )
        # quirk: counts k_full twice (k_full^2 * m_partial * n_partial); no tiling in this design space has an m partial tile
        self.m_partial_k_full_n_partial_total_tiles = (
            m_partial_k_full * k_full_n_partial * self.batch
        )
        self.m_partial_k_partial_n_full_total_tiles = (
            m_partial_k_partial * k_partial_n_full * self.batch
        )
        self.m_partial_k_partial_n_partial_total_tiles = (
            m_partial * k_partial * n_partial * self.batch
        )

        self.tile_m_partial = self.m % self.tile_m
        self.tile_k_partial = self.k % self.tile_k
        self.tile_n_partial = self.n % self.tile_n

        # tile sizes in bits
        self.m_full_k_full_tile_bits = self.m_k_bitwidth * self.tile_m * self.tile_k
        self.k_full_n_full_tile_bits = self.k_n_bitwidth * self.tile_k * self.tile_n
        self.m_full_n_full_tile_bits = self.m_n_bitwidth * self.tile_m * self.tile_n
        self.m_partial_k_full_tile_bits = (
            self.m_k_bitwidth * self.tile_m_partial * self.tile_k
        )
        self.m_full_k_partial_tile_bits = (
            self.m_k_bitwidth * self.tile_m * self.tile_k_partial
        )
        self.k_partial_n_full_tile_bits = (
            self.k_n_bitwidth * self.tile_k_partial * self.tile_n
        )
        self.k_full_n_partial_tile_bits = (
            self.k_n_bitwidth * self.tile_k * self.tile_n_partial
        )
        self.m_partial_n_full_tile_bits = (
            self.m_n_bitwidth * self.tile_m_partial * self.tile_n
        )
        self.m_full_n_partial_tile_bits = (
            self.m_n_bitwidth * self.tile_m * self.tile_n_partial
        )
        self.m_partial_k_partial_tile_bits = (
            self.m_k_bitwidth * self.tile_m_partial * self.tile_k_partial
        )
        self.k_partial_n_partial_tile_bits = (
            self.k_n_bitwidth * self.tile_k_partial * self.tile_n_partial
        )
        self.m_partial_n_partial_tile_bits = (
            self.m_n_bitwidth * self.tile_m_partial * self.tile_n_partial
        )


class TiledMatrix:
    """Tile counts and tile sizes (in bits) of a batch of (m x n) matrices for an elementwise op."""

    def __init__(self, batch, m, n, tile_m, tile_n, m_n_bitwidth, array_height=None):
        if 0 in (m, n, tile_m, tile_n):
            self.is_valid = False
            return
        self.is_valid = True

        self.batch = batch
        self.m = m
        self.n = n
        self.tile_m = min(tile_m, self.m)
        self.tile_n = min(tile_n, self.n)
        self.m_n_bitwidth = m_n_bitwidth
        self.m_util = self.tile_m / array_height if array_height else 1

        self.n_total = self.n * self.batch
        self.m_tiles = math.ceil(self.m / self.tile_m)
        self.n_tiles = math.ceil(self.n / self.tile_n)
        self.m_n_matrix_tiles = self.m_tiles * self.n_tiles
        self.m_n_total_tiles = self.m_n_matrix_tiles * self.batch
        self.m_total_tiles = self.m_tiles * self.batch
        self.m_n_total_matrix = self.m * self.n * self.batch

        m_full = math.floor(self.m / self.tile_m)
        n_full = math.floor(self.n / self.tile_n)
        m_partial = self.m_tiles - m_full
        n_partial = self.n_tiles - n_full
        self.m_full_n_full_total_tiles = m_full * n_full * self.batch
        self.m_full_n_partial_total_tiles = m_full * n_partial * self.batch
        self.m_partial_n_full_total_tiles = n_full * m_partial * self.batch
        self.m_partial_n_partial_total_tiles = m_partial * n_partial * self.batch

        self.tile_m_partial = self.m % self.tile_m
        self.tile_n_partial = self.n % self.tile_n

        self.m_full_n_full_tile_bits = self.m_n_bitwidth * self.tile_m * self.tile_n
        self.m_partial_n_full_tile_bits = (
            self.m_n_bitwidth * self.tile_m_partial * self.tile_n
        )
        self.m_full_n_partial_tile_bits = (
            self.m_n_bitwidth * self.tile_m * self.tile_n_partial
        )
        self.m_partial_n_partial_tile_bits = (
            self.m_n_bitwidth * self.tile_m_partial * self.tile_n_partial
        )


# per stationary: how often each operand tile is moved, as a TiledGEMM attribute (None = once)
GEMM_TILE_REUSE = {
    "os": ("n_tiles", "m_tiles", None),
    "is": (None, "m_tiles", "k_tiles"),
    "ws": ("n_tiles", None, "k_tiles"),
}


def _gemm_transfers(
    tiles: TiledGEMM, stationary, mk_width, kn_width, mn_width, mn_reuse_offset=0
):
    """Transfers of the input (m x k), weight (k x n) and output (m x n) tiles, each tile moved
    ceil(tile_bits / width) times per use."""
    mk_reuse, kn_reuse, mn_reuse = (
        1 if r is None else getattr(tiles, r) for r in GEMM_TILE_REUSE[stationary]
    )
    mn_reuse -= mn_reuse_offset

    def moves(parts, reuse, width):
        return sum(
            getattr(tiles, f"{p}_total_tiles")
            * reuse
            * math.ceil(getattr(tiles, f"{p}_tile_bits") / width)
            for p in parts
        )

    mk = moves(
        [
            "m_full_k_full",
            "m_full_k_partial",
            "m_partial_k_full",
            "m_partial_k_partial",
        ],
        mk_reuse,
        mk_width,
    )
    kn = moves(
        [
            "k_full_n_full",
            "k_full_n_partial",
            "k_partial_n_full",
            "k_partial_n_partial",
        ],
        kn_reuse,
        kn_width,
    )
    mn = moves(
        [
            "m_full_n_full",
            "m_full_n_partial",
            "m_partial_n_full",
            "m_partial_n_partial",
        ],
        mn_reuse,
        mn_width,
    )
    return mk, kn, mn


def _matrix_transfers(tiles: TiledMatrix, width):
    parts = [
        "m_full_n_full",
        "m_full_n_partial",
        "m_partial_n_full",
        "m_partial_n_partial",
    ]
    return sum(
        getattr(tiles, f"{p}_total_tiles")
        * math.ceil(getattr(tiles, f"{p}_tile_bits") / width)
        for p in parts
    )


def _sram_widths(architecture_dict):
    return (
        architecture_dict["isram"]["query"]["width"],
        architecture_dict["wsram"]["query"]["width"],
        architecture_dict["osram"]["query"]["width"],
    )


def _counts(**counts):
    return OrderedDict(
        {name: OrderedDict({"count": count}) for name, count in counts.items()}
    )


def _router_average_events(tile_count, router_height, router_dim):
    """Average routers a tile mapping crosses: router_height for each full mesh of tiles,
    1 for the last partial mesh."""
    full = math.floor(tile_count / router_dim)
    partial = 1 if tile_count % router_dim > 0 else 0
    partial_height = round(math.sqrt(partial))
    return ((router_height * full) + (partial_height * partial)) / (full + partial)


# region GEMM
def offchip_gemm_events(tiles: TiledGEMM, architecture_dict, workload):
    """dram reads/writes and node sram writes/reads moving the GEMM's noc tiles."""
    stationary = workload["noc_stationary"]
    isram_width, wsram_width, osram_width = _sram_widths(architecture_dict)
    dram_mk, dram_kn, dram_mn = _gemm_transfers(
        tiles, stationary, DRAM_WIDTH, DRAM_WIDTH, DRAM_WIDTH
    )
    sram_mk, sram_kn, sram_mn = _gemm_transfers(
        tiles, stationary, isram_width, wsram_width, osram_width
    )
    return _counts(
        isram_offchip_writes=sram_mk,
        wsram_offchip_writes=sram_kn,
        osram_offchip_reads=sram_mn,
        osram_offchip_writes=0,
        dram_input_reads=dram_mk,
        dram_weight_reads=dram_kn,
        dram_output_reads=0,
        dram_output_writes=dram_mn,
    )


def router_gemm_events(tiles: TiledGEMM, offchip_events, architecture_dict, workload):
    router_height, router_width = architecture_dict["irouter"]["instance"][0:2]
    router_dim = router_height * router_width
    stationary = workload["noc_stationary"]
    isram_width, wsram_width, osram_width = _sram_widths(architecture_dict)

    if stationary == "os":
        mk_tiles, kn_tiles, mn_tiles = (
            tiles.m_k_n_total_tiles,
            tiles.m_k_n_total_tiles,
            tiles.m_n_total_tiles,
        )
    elif stationary == "is":
        mk_tiles, kn_tiles, mn_tiles = (
            tiles.m_k_total_tiles,
            tiles.m_k_n_total_tiles,
            tiles.m_k_n_total_tiles,
        )
    elif stationary == "ws":
        mk_tiles, kn_tiles, mn_tiles = (
            tiles.m_k_n_total_tiles,
            tiles.k_n_total_tiles,
            tiles.m_k_n_total_tiles,
        )

    mk_average = _router_average_events(mk_tiles, router_height, router_dim)
    kn_average = _router_average_events(kn_tiles, router_height, router_dim)
    mn_average = _router_average_events(mn_tiles, router_height, router_dim)

    # quirk: each router scales by another sram's width (irouter by osram, wrouter by isram, orouter by wsram)
    return _counts(
        irouter_mapping=mk_average
        * offchip_events["isram_offchip_writes"]["count"]
        * osram_width,
        wrouter_mapping=kn_average
        * offchip_events["wsram_offchip_writes"]["count"]
        * isram_width,
        orouter_mapping=mn_average
        * offchip_events["osram_offchip_reads"]["count"]
        * wsram_width,
    )


def onchip_gemm_scheduling(m, k, n, tiles: TiledGEMM, architecture_dict):
    """Array-sized tilings of each kind of noc tile (full, or partial along m, k or n)."""
    array_height = architecture_dict["wfifo"]["instance"][-1]
    array_width = architecture_dict["ififo"]["instance"][-1]
    tile_m = min(array_width, m)
    tile_k = min(array_height, k)
    tile_n = min(array_height, n)

    tilings = []
    for edges in (
        ("full", "full", "full"),
        ("full", "full", "partial"),
        ("full", "partial", "full"),
        ("full", "partial", "partial"),
        ("partial", "full", "full"),
        ("partial", "full", "partial"),
        ("partial", "partial", "full"),
        ("partial", "partial", "partial"),
    ):
        m_edge, k_edge, n_edge = edges
        tilings.append(
            TiledGEMM(
                batch=getattr(tiles, f"m_{m_edge}_k_{k_edge}_n_{n_edge}_total_tiles"),
                m=tiles.tile_m if m_edge == "full" else tiles.tile_m_partial,
                k=tiles.tile_k if k_edge == "full" else tiles.tile_k_partial,
                n=tiles.tile_n if n_edge == "full" else tiles.tile_n_partial,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                m_k_bitwidth=tiles.m_k_bitwidth,
                k_n_bitwidth=tiles.k_n_bitwidth,
                m_n_bitwidth=tiles.m_n_bitwidth,
                array_width=array_width,
                array_height=array_height,
            )
        )
    tilings = [t for t in tilings if t.is_valid]
    if not tilings:
        raise ValueError("Invalid tiling configuration")
    return tilings


def onchip_gemm_events(tilings, architecture_dict, workload):
    """node sram <-> array reads and writes."""
    stationary = workload["node_stationary"]
    isram_width, wsram_width, osram_width = _sram_widths(architecture_dict)
    isram_reads = wsram_reads = osram_reads = osram_writes = 0
    for tiles in tilings:
        mk, kn, mn_writes = _gemm_transfers(
            tiles, stationary, isram_width, wsram_width, osram_width
        )
        # a partial sum is read back for every k tile but the first, except output stationary keeps it in the array
        _, _, mn_reads = _gemm_transfers(
            tiles,
            stationary,
            isram_width,
            wsram_width,
            osram_width,
            mn_reuse_offset=0 if stationary == "os" else 1,
        )
        isram_reads += mk
        wsram_reads += kn
        osram_reads += mn_reads
        osram_writes += mn_writes
    return _counts(
        isram_onchip_reads=isram_reads,
        wsram_onchip_reads=wsram_reads,
        osram_onchip_reads=osram_reads,
        osram_onchip_writes=osram_writes,
    )


def gemm_tile_events(
    tiles: TiledGEMM, architecture_dict, workload, early_termination_cycles
):
    vector_array_dim = architecture_dict["multiplier_vector"]["instance"][-1]
    stationary = workload["node_stationary"]
    cycles = workload["cycles"]
    # early termination shortens every bit-serial array step from cycles to early_termination_cycles
    et_factor = (
        early_termination_cycles / cycles if early_termination_cycles is not None else 1
    )

    subevent = OrderedDict()

    def add(name, count, cycle_factor):
        subevent[name] = OrderedDict(
            {
                "count": count,
                "factor": {"cycle_count": cycle_factor, "runtime": cycle_factor},
            }
        )

    # load input
    input_events = (
        tiles.n_total_tiles * tiles.k
        if stationary == "is"
        else tiles.m_n_total_tiles * tiles.k
    )
    input_utilization = (tiles.m / (tiles.m_tiles * tiles.tile_m)) * tiles.m_util
    add("input_gemm", input_events * input_utilization, 1 / input_utilization)

    # counter value reuse
    subevent["counter_reuse"] = OrderedDict(
        {
            "count": tiles.m_n_total_tiles * tiles.k,
            "factor": {"cycle_count": et_factor, "runtime": et_factor},
        }
    )

    # broadcast value reuse
    add(
        "input_reuse_gemm",
        tiles.m_n_total_tiles * tiles.k * input_utilization,
        et_factor * (1 / input_utilization),
    )

    # load weight
    weight_events = (
        tiles.m_total_tiles * tiles.k
        if stationary == "ws"
        else tiles.m_n_total_tiles * tiles.k
    )
    weight_utilization = (tiles.n / (tiles.n_tiles * tiles.tile_n)) * tiles.n_util
    add("weight_gemm", weight_events * weight_utilization, 1 / weight_utilization)

    # temporal conversion
    add(
        "weight_reuse_gemm",
        tiles.m_n_total_tiles * tiles.k * weight_utilization,
        et_factor * (1 / weight_utilization),
    )

    # array computation
    array_utilization = input_utilization * weight_utilization
    add(
        "array_gemm",
        tiles.m_n_total_tiles * tiles.k * array_utilization,
        et_factor * (1 / array_utilization),
    )

    # hardware for multiple spikes in compute array
    fifo_utilization = input_utilization * weight_utilization * 0.5
    add(
        "array_fifo_gemm",
        tiles.m_n_total_tiles * tiles.k * fifo_utilization,
        et_factor * (1 if fifo_utilization == 0 else 1 / fifo_utilization),
    )

    # vector scaling
    vector_events = tiles.m_total * tiles.n_tiles * (tiles.tile_n / vector_array_dim)
    vector_utilization = tiles.m_n_total_matrix / (vector_events * vector_array_dim)
    add("vector", vector_events * vector_utilization, 1 / vector_utilization)

    subevent["nonlinear_gemm"] = OrderedDict({"count": 0})
    return OrderedDict({"subevent": subevent})


def gemm_mapping(batch, m, k, n, architecture_dict, workload, early_termination_cycles):
    """Subevent counts of a batch of (m x k) @ (k x n) GEMMs.

    early_termination_cycles: average array cycles per bit-serial step, or None for the full workload['cycles'].
    """
    offchip_tiles = TiledGEMM(
        batch=batch,
        m=m,
        k=k,
        n=n,
        tile_m=workload["noc_tile_m"],
        tile_k=workload["noc_tile_k"],
        tile_n=workload["noc_tile_n"],
        m_k_bitwidth=workload["activation_bitwidth"],
        k_n_bitwidth=workload["weight_bitwidth"],
        m_n_bitwidth=workload["activation_bitwidth"],
    )
    if not offchip_tiles.is_valid:
        raise ValueError("Invalid tiling configuration")

    subevent = offchip_gemm_events(offchip_tiles, architecture_dict, workload)
    # routers exist only in multi-node architectures
    if "irouter" in architecture_dict:
        subevent.update(
            router_gemm_events(offchip_tiles, subevent, architecture_dict, workload)
        )

    tilings = onchip_gemm_scheduling(m, k, n, offchip_tiles, architecture_dict)
    subevent.update(onchip_gemm_events(tilings, architecture_dict, workload))

    array_events = None
    for tiles in tilings:
        tile_events = gemm_tile_events(
            tiles, architecture_dict, workload, early_termination_cycles
        )
        array_events = (
            tile_events
            if array_events is None
            else sum_subevents(array_events, tile_events)
        )
    # gemm/nonlinear mux switching event (once per gemm)
    array_events["subevent"]["instruction"] = OrderedDict({"count": 1})
    subevent.update(array_events["subevent"])

    for sub in subevent.values():
        sub["count"] = int(sub["count"])
    return OrderedDict({"subevent": subevent})


# endregion


# region nonlinear
def offchip_nonlinear_scheduling(batch, m, n, architecture_dict, workload):
    """noc tile of an elementwise op: input and output tiles must both fit in wsram and osram."""
    wsram = architecture_dict["wsram"]
    osram = architecture_dict["osram"]
    bitwidth = workload["activation_bitwidth"]
    array_height = architecture_dict["wfifo"]["instance"][-1]

    # half the banks hold a tile while the other half fill (double buffering)
    wsram_elements = (
        (wsram["query"]["bank"] / 2)
        * wsram["query"]["width"]
        * wsram["query"]["depth"]
        / bitwidth
    )
    osram_elements = (
        (osram["query"]["bank"] / 2)
        * osram["query"]["width"]
        * osram["query"]["depth"]
        / bitwidth
    )
    elements = min(wsram_elements, osram_elements)

    # quirk: tiles assume 1 node on every mesh (the original model checked for a 'router' module, which no design has)
    nodes = 1

    tile_m = min(m, array_height)
    tile_n = (n * batch) / nodes
    # shrink tile_n until the tile fits, then grow tile_n and tile_m while twice the tile still fits
    while tile_n * tile_m > elements:
        tile_n /= 2
    while tile_n * tile_m * 2 < elements:
        tile_n *= 2
    while (tile_m * tile_n * 2 < elements) and (tile_m < m):
        tile_m *= 2

    if tile_m * tile_n > wsram_elements:
        raise ValueError("Tile size exceeds wsram size")
    if tile_m * tile_n > osram_elements:
        raise ValueError("Tile size exceeds osram size")

    tiles = TiledMatrix(
        batch=batch,
        m=m,
        n=n,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        m_n_bitwidth=bitwidth,
    )
    if not tiles.is_valid:
        raise ValueError("Invalid tiling configuration")
    return tiles


def offchip_nonlinear_events(tiles: TiledMatrix, architecture_dict, workload):
    """dram <-> node sram moves of the noc tiles, plus loading the nonlinear LUT into the isram."""
    isram_width, _, osram_width = _sram_widths(architecture_dict)
    # quirk: the LUT goes to 1 node on every mesh (the original model checked for a 'router' module, which no design has)
    noc_dim = 1

    m_n_events = _matrix_transfers(tiles, osram_width)
    lut_bits = (
        workload["lut_height"] * workload["lut_width"] * workload["weight_bitwidth"]
    )

    return _counts(
        isram_offchip_writes=min(tiles.m_n_matrix_tiles, noc_dim)
        * math.ceil(lut_bits / isram_width),
        wsram_offchip_writes=0,
        osram_offchip_reads=m_n_events,
        osram_offchip_writes=m_n_events,
        dram_input_reads=math.ceil(lut_bits / isram_width),
        dram_weight_reads=0,
        dram_output_reads=m_n_events,
        dram_output_writes=m_n_events,
    )


def router_nonlinear_events(tiles: TiledMatrix, offchip_events, architecture_dict):
    router_height, router_width = architecture_dict["irouter"]["instance"][0:2]
    router_dim = router_height * router_width

    mn_average = _router_average_events(
        tiles.m_n_total_tiles, router_height, router_dim
    )
    irouter_average = (
        mn_average if tiles.m_n_total_tiles < router_dim else router_height
    )

    return _counts(
        irouter_mapping=irouter_average
        * offchip_events["isram_offchip_writes"]["count"],
        wrouter_mapping=mn_average * offchip_events["wsram_offchip_writes"]["count"],
        orouter_mapping=mn_average
        * (
            offchip_events["osram_offchip_reads"]["count"]
            + offchip_events["osram_offchip_writes"]["count"]
        ),
    )


def onchip_nonlinear_scheduling(m, offchip_tiles: TiledMatrix, architecture_dict):
    """Array-sized tilings (one column at a time) of each kind of noc tile."""
    wfifo_width = architecture_dict["wfifo"]["instance"][-1]
    array_height = architecture_dict["ofifo"]["instance"][-1]
    if wfifo_width != array_height:
        raise ValueError("wfifo and ofifo width must be the same")

    tile_m = min(array_height, m)
    tilings = []
    for m_edge, n_edge in (
        ("full", "full"),
        ("full", "partial"),
        ("partial", "full"),
        ("partial", "partial"),
    ):
        tilings.append(
            TiledMatrix(
                batch=getattr(offchip_tiles, f"m_{m_edge}_n_{n_edge}_total_tiles"),
                m=(
                    offchip_tiles.tile_m
                    if m_edge == "full"
                    else offchip_tiles.tile_m_partial
                ),
                n=(
                    offchip_tiles.tile_n
                    if n_edge == "full"
                    else offchip_tiles.tile_n_partial
                ),
                tile_m=tile_m,
                tile_n=1,
                m_n_bitwidth=offchip_tiles.m_n_bitwidth,
                array_height=array_height,
            )
        )
    tilings = [t for t in tilings if t.is_valid]
    if not tilings:
        raise ValueError("Invalid tiling configuration")
    return tilings


def onchip_nonlinear_events(function, tilings, architecture_dict, workload):
    """node sram <-> array reads and writes; each array step reads one LUT window from the isram."""
    isram_width, _, osram_width = _sram_widths(architecture_dict)
    window_bits = workload["window_width"] * workload["weight_bitwidth"]
    isram_reads = osram_reads = osram_writes = 0
    for tiles in tilings:
        m_n_events = _matrix_transfers(tiles, osram_width)
        # softmax reads and writes each element twice (exp, then divide by the sum)
        passes = 2 if function == "softmax" else 1
        osram_reads += m_n_events * passes
        osram_writes += m_n_events * passes
        isram_reads += (tiles.m_n_total_tiles * workload["lut_height"]) * math.ceil(
            window_bits / isram_width
        )
    return _counts(
        isram_onchip_reads=isram_reads,
        wsram_onchip_reads=0,
        osram_onchip_reads=osram_reads,
        osram_onchip_writes=osram_writes,
    )


def nonlinear_tile_events(function, tiles: TiledMatrix, architecture_dict):
    vector_array_dim = architecture_dict["multiplier_vector"]["instance"][-1]
    subevent = OrderedDict()

    def add(name, count, cycle_factor):
        subevent[name] = OrderedDict(
            {
                "count": count,
                "factor": {"cycle_count": cycle_factor, "runtime": cycle_factor},
            }
        )

    # input load
    subevent["input_nonlinear"] = OrderedDict({"count": tiles.m_tiles * tiles.n_total})
    # counter value reuse
    subevent["counter_reuse"] = OrderedDict({"count": tiles.m_tiles * tiles.n_total})

    # load weight (activations in nonlinear); silu maps both signs, so its steps take twice the cycles
    weight_utilization = (tiles.m / (tiles.m_tiles * tiles.tile_m)) * tiles.m_util
    weight_cycle_factor = (
        1 / weight_utilization
        if function == "softmax"
        else (1 / weight_utilization) * 2
    )
    add(
        "weight_nonlinear",
        tiles.m_total_tiles * tiles.n * weight_utilization,
        weight_cycle_factor,
    )

    # temporal conversion
    weight_reuse_utilization = (
        weight_utilization if function == "softmax" else weight_utilization * 2
    )
    add(
        "weight_reuse_nonlinear",
        tiles.m_tiles * tiles.n_total * weight_reuse_utilization,
        1 / weight_reuse_utilization,
    )

    # array computation (LUT selection)
    array_utilization = weight_utilization * 2
    add(
        "array_nonlinear",
        tiles.m_tiles * tiles.n_total * array_utilization,
        1 / array_utilization,
    )

    if function == "softmax":
        # sum the exponents, then divide by the sum
        add(
            "summation",
            tiles.m_tiles * tiles.n_total * weight_utilization,
            1 / weight_utilization,
        )
        vector_events = (
            tiles.m_tiles * (tiles.tile_m / vector_array_dim) * tiles.n_total
        )
        vector_utilization = tiles.m_n_total_matrix / (vector_events * vector_array_dim)
        add("vector", vector_events * vector_utilization, 1 / vector_utilization)
    else:
        subevent["summation"] = OrderedDict({"count": 0})
        subevent["vector"] = OrderedDict({"count": 0})

    subevent["array_fifo_nonlinear"] = OrderedDict({"count": 0})
    subevent["gemm_nonlinear"] = OrderedDict({"count": 0})
    return OrderedDict({"subevent": subevent})


def nonlinear_mapping(function, batch, m, n, architecture_dict, workload):
    """Subevent counts of an elementwise nonlinear function ('softmax' or 'silu') over a batch of (m x n) matrices."""
    offchip_tiles = offchip_nonlinear_scheduling(
        batch, m, n, architecture_dict, workload
    )
    subevent = offchip_nonlinear_events(offchip_tiles, architecture_dict, workload)
    # routers exist only in multi-node architectures
    if "irouter" in architecture_dict:
        subevent.update(
            router_nonlinear_events(offchip_tiles, subevent, architecture_dict)
        )

    tilings = onchip_nonlinear_scheduling(m, offchip_tiles, architecture_dict)
    subevent.update(
        onchip_nonlinear_events(function, tilings, architecture_dict, workload)
    )

    array_events = None
    for tiles in tilings:
        tile_events = nonlinear_tile_events(function, tiles, architecture_dict)
        array_events = (
            tile_events
            if array_events is None
            else sum_subevents(array_events, tile_events)
        )
    # gemm/nonlinear mux switching event (once per nonlinear op)
    array_events["subevent"]["instruction"] = OrderedDict({"count": 1})
    subevent.update(array_events["subevent"])

    for sub in subevent.values():
        sub["count"] = int(sub["count"])
    return OrderedDict({"subevent": subevent})


# endregion
