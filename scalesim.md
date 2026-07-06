# ArchX chiplet4ai vs SCALE-Sim WS Modeling Notes

## Context

`chiplet4ai` currently models TPU-style weight-stationary GEMM using ArchX events. The model is mostly analytical: it counts array, SRAM, and DRAM events from tiling formulas, then ArchX aggregates those event counts into area, energy, and runtime.

SCALE-Sim also supports weight-stationary systolic arrays, but its internal model is different. It builds per-cycle operand demand matrices for IFMAP, filter, and OFMAP, runs them through double-buffered SRAM/DRAM models, then reports cycles, stalls, bandwidth, and utilization. Even when traces are not saved, SCALE-Sim uses trace-like demand matrices internally.

The goal here is to make `chiplet4ai` closer to SCALE-Sim's modeling behavior without emitting or consuming traces.

## Main Gap

Our model is an event-count model. SCALE-Sim is a cycle-demand model.

The closest trace-free direction is to make our events represent SCALE-Sim-like phases:

- fold setup
- weight load / prefetch
- IFMAP streaming
- compute wavefront
- OFMAP drain/write
- memory stalls from prefetch or output drain limits

This preserves ArchX's event-based structure, but makes each event's count and runtime reflect the same timing assumptions SCALE-Sim uses.

## Ranked Improvements

### 1. Replace GEMM compute timing with SCALE-Sim-style fold timing - Done

Current `chiplet4ai/common/performance/mapping.py` estimates array work from tile counts and utilization factors:

```text
m_tiles = ceil(M / mt)
k_tiles = ceil(K / kt)
n_tiles = ceil(N / nt)
compute_events ~= m_tiles * k_tiles * n_tiles * utilization
```

SCALE-Sim's WS model instead maps GEMM as:

```text
Sr = K
Sc = N
T  = M
row_folds = ceil(Sr / array_rows)
col_folds = ceil(Sc / array_cols)
```

For each fold, SCALE-Sim accounts for systolic fill/drain overhead. A trace-free approximation should compute per-fold cycles as:

```text
cycles_this_fold ~= T + array_rows + array_cols - 2
```

Then total non-stalled compute cycles are summed over all row and column folds.

This should be the first change because it directly affects runtime and utilization, and it is the foundation for later memory-stall modeling.

Status: implemented in `chiplet4ai/common/performance/mapping.py::gemm(...)` using shared WS folds and `M + array_rows + array_cols - 2` per fold.

### 2. Track mapping efficiency and compute utilization explicitly - Done

SCALE-Sim reports two separate quantities:

- mapping efficiency: spatial occupancy of the array for each fold
- compute utilization: useful MACs divided by available MAC slots over the fold's active cycles

Our model currently folds this into event factors. That makes it harder to reason about whether poor performance comes from edge-tile underutilization, systolic fill/drain overhead, or memory stalls.

For each WS fold, we should compute:

```text
row_used = min(array_rows, remaining_K)
col_used = min(array_cols, remaining_N)
mapping_efficiency = row_used * col_used / (array_rows * array_cols)
useful_macs = M * row_used * col_used
available_mac_slots = cycles_this_fold * array_rows * array_cols
compute_utilization = useful_macs / available_mac_slots
```

These can be exposed as specified metrics or used internally to derive event factors.

Status: implemented in `chiplet4ai/common/performance/mapping.py::gemm(...)` as `mapping_efficiency` and `compute_utilization` specified metrics.

### 3. Model per-operand demand rates, not only total counts - Done

SCALE-Sim derives SRAM bandwidth from when operands are demanded, not only how many operands exist.

Without traces, we can summarize each fold into phases:

```text
weight_load_phase
ifmap_stream_phase
compute_phase
ofmap_drain_phase
```

For each phase, estimate:

- duration in cycles
- IFMAP words/cycle
- weight words/cycle
- OFMAP words/cycle

This gives a schedule-level model while staying analytical.

Status: implemented at a fold-summary level in `chiplet4ai/common/performance/mapping.py::sram(...)` and `dram(...)` by deriving IFMAP, weight, and OFMAP movement from the same WS folds as `gemm(...)`. This does not yet model memory stalls.

### 4. Add a double-buffer feasibility/stall model - Done

SCALE-Sim's biggest behavioral difference is that memory can stall compute. It models active and prefetch buffers for reads, and active/drain behavior for output writes.

Trace-free approximation:

```text
prefetch_cycles = ceil(bytes_to_prefetch / dram_bandwidth)
available_overlap_cycles = compute_cycles_before_data_needed
read_stall = max(0, prefetch_cycles - available_overlap_cycles)

drain_cycles = ceil(output_bytes_to_drain / dram_bandwidth)
available_output_overlap = compute_cycles_before_output_buffer_full
write_stall = max(0, drain_cycles - available_output_overlap)
```

This would let ArchX report SCALE-Sim-like `stall_cycles` without address-level traces.

This is the analytical memory stall modeling item.

Status: implemented in `chiplet4ai/common/performance/mapping.py` as a shared WS schedule helper. The helper estimates residual DRAM read prefetch stalls and output drain stalls from fold-level operand bytes, DRAM bandwidth, and available compute overlap. This is still analytical and trace-free; it does not model SCALE-Sim's full active/prefetch buffer address state.

### 5. Make SRAM capacity semantics closer to SCALE-Sim - Done

SCALE-Sim splits SRAM into active and prefetch regions, commonly with an active fraction around 0.5.

Our model currently uses expressions like `2 * sram_elements` in places, which effectively treats double buffering as extra total residency. To be closer to SCALE-Sim:

```text
active_elements = floor(total_elements * active_fraction)
prefetch_elements = total_elements - active_elements
```

Tile fitting should use active capacity. Prefetch feasibility should use prefetch capacity.

Status: implemented in `chiplet4ai/common/performance/mapping.py::_ws_schedule(...)`. SRAM capacity is split into active and prefetch regions using `active_fraction` / `active_buffer_fraction` when present, defaulting to 0.5. Active capacity controls IFMAP/weight/output residency. Prefetch capacity contributes to read-prefetch feasibility.

### 6. Revisit output / partial-sum behavior - Done

Our model explicitly estimates output partial read/write traffic when outputs do not fit. SCALE-Sim's WS model primarily represents OFMAP writes through its write buffer and derives DRAM writes/drain timing from that.

This is not necessarily a bug in our model, but it is a modeling difference. We should decide whether the goal is:

- closer SCALE-Sim compatibility, using mostly write-buffer-style OFMAP accounting, or
- more explicit partial-sum read/write modeling, accepting differences from SCALE-Sim.

This should come after compute and memory timing are aligned, because output behavior depends on the chosen accumulation model.

Status: implemented as a trace-free output-buffer occupancy model in `chiplet4ai/common/performance/mapping.py::_ws_schedule(...)`. OFMAP bytes enter an output write buffer, drain at the modeled output service rate, and add write stalls when occupancy exceeds buffer capacity. Output partial reads are counted when an output tile cannot fit in active output SRAM across K folds.

## Recommended Sequence

1. Done - Implement SCALE-Sim-style WS fold timing.
2. Done - Add explicit mapping efficiency and compute utilization.
3. Done - Replace aggregate operand traffic with phase-level operand demand rates.
4. Done - Add analytical double-buffer read/write stall estimates.
5. Done - Adjust SRAM capacity modeling to active/prefetch regions.
6. Done - Add output write-buffer occupancy and partial-output spill handling.

## Design Principle

Do not make ArchX trace-based. Instead, make the event model compute the same summaries SCALE-Sim derives from traces:

- fold count
- useful MACs
- total cycles
- stall cycles
- SRAM reads/writes
- DRAM reads/writes
- average SRAM bandwidth
- average DRAM bandwidth
- mapping efficiency
- compute utilization

This keeps ArchX fast and compositional while making its TPU WS model much easier to compare against SCALE-Sim.
