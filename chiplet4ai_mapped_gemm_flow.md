# Chiplet4AI Mapped GEMM Performance Flow

This note describes the current event-based performance flow for a mapped GEMM in `chiplet4ai/common/performance/mapping.py`.

## Entry Points

The mapped GEMM model is exposed through three performance functions:

- `gemm(...)`: array-side mapping and compute metrics.
- `sram(...)`: SRAM mapping event counts and SRAM-side bandwidth metrics.
- `dram(...)`: DRAM movement counts, DRAM-side stall factors, and DRAM bandwidth metrics.

All three call the shared scheduler:

```text
gemm/sram/dram
  -> _ws_schedule(...)
```

This keeps array, SRAM, and DRAM views consistent for the same mapped GEMM.

## Event Graph Context

In the Llama description, each GEMM-like layer event is split into three mapping events:

```text
<gemm_event>
  -> <gemm_event>_arr
  -> <gemm_event>_sram
  -> <gemm_event>_dram
```

Those map to lower-level events:

```text
<gemm_event>_arr
  -> array_input_mapping
  -> array_weight_mapping
  -> array_compute_mapping

<gemm_event>_sram
  -> sram_input_write_mapping
  -> sram_weight_write_mapping
  -> sram_output_read_mapping
  -> sram_output_write_mapping

<gemm_event>_dram
  -> dram_input_read
  -> dram_weight_read
  -> dram_output_read
  -> dram_output_write
```

The performance model returns event counts, factors, and specified metrics for these mapping nodes. ArchX then aggregates them through the event graph.

## Schedule Setup

`_ws_schedule(...)` first normalizes architecture inputs:

```text
architecture_dict
  -> pe rows/cols
  -> isram/wsram/osram/dram modules
```

It derives:

- element widths for input, weight, and output.
- SRAM active and prefetch capacities using `_buffer_elements(...)`.
- DRAM bytes per cycle.
- SRAM bytes per cycle for input, weight, and output SRAMs.
- per-operand DRAM prefetch service rates:
  - input: `min(dram_bpc, isram_bpc)`
  - weight: `min(dram_bpc, wsram_bpc)`
  - output drain: `min(dram_bpc, osram_bpc)`

The model treats SRAM as double-buffered by splitting each SRAM into active and prefetch/drain regions.

## Step and Fold Traversal

The model supports stepped dimensions via `_step_config(...)` and `_step_dims(...)`.

For each step, `_ws_fold_infos(...)` generates weight-stationary folds:

```text
for k_start in range(0, K, array_rows):
  row_used = min(array_rows, K - k_start)
  for n_start in range(0, N, array_cols):
    col_used = min(array_cols, N - n_start)
    cycles_this_fold = M + array_rows + array_cols - 2
```

Each fold carries `k_start` and `n_start`, which are needed for residency keys, output reuse, and fold-level scheduling.

## Per-Fold Data Sizes

For each fold:

```text
input_elements  = batch * M_step * row_used
weight_elements = batch * row_used * col_used
output_elements = batch * M_step * col_used
```

These are converted into bytes using the operand bitwidths from the SRAM query widths.

## Chunk-Level Input and Weight Residency

Input and weight residency are tracked with chunk-level keys:

```text
input_key  = (step, k_start, row_used, M_step, batch)
weight_key = (step, k_start, n_start, row_used, col_used, batch)
```

`_touch_chunked_resident(...)` splits each fold tile into chunks bounded by active and prefetch capacities. It returns:

- total missed elements.
- missed elements in the active startup window.
- total missed chunks.
- missed chunks in the active startup window.

This means the model distinguishes:

- total bytes that must be moved for the fold.
- bytes that must be ready before compute can begin.

The remaining bytes can be prefetched while earlier work executes.

## DRAM Read Prefetch Scheduling

Input and weight use separate read engines:

```text
input_read_engine_available_cycle
weight_read_engine_available_cycle
```

For each operand:

```text
prefetch_cycles = service_cycles(total_missing_bytes, operand_prefetch_bpc)
startup_cycles  = service_cycles(startup_missing_bytes, operand_prefetch_bpc)
```

The full prefetch cycles reserve the operand read engine. Only startup cycles gate the fold's compute start.

The fold can begin when:

```text
compute_start_cycle = max(
  previous_compute_end_cycle,
  input_ready_cycle,
  weight_ready_cycle,
)
```

Read stall metrics are attributed separately:

- `input_read_stall_cycle_count`
- `weight_read_stall_cycle_count`

The fold-level read stall uses max composition across input and weight readiness.

## Phase-Based SRAM Stream Modeling

After DRAM readiness, the fold is split into three event-level phases:

```text
weight_fill_cycles = batch * max(0, array_rows - 1)
steady_cycles      = batch * M_step
output_tail_cycles = batch * max(0, array_cols - 1)
```

These sum to the original weight-stationary fold latency:

```text
batch * (M_step + array_rows + array_cols - 2)
```

SRAM pressure is assigned to the phase where each operand is needed:

- weight SRAM traffic -> weight-fill phase.
- input SRAM traffic -> steady-state phase.
- output reread traffic -> steady-state phase.
- output write traffic -> output-tail phase.

Within a phase, service pressure is max-composed. Across phases, stalls are sequentially summed:

```text
weight_fill_stall = max(0, weight_service - weight_fill_cycles)
steady_state_stall = max(0, max(input_service, output_read_service) - steady_cycles)
output_tail_stall = max(0, output_write_service - output_tail_cycles)
```

The model exposes:

- `weight_fill_stall_cycle_count`
- `steady_state_stall_cycle_count`
- `output_tail_stall_cycle_count`

This remains event/fold/phase based. It does not generate per-cycle traces or demand matrices.

## Output Residency and Accumulation

Output residency is tracked at tile granularity:

```text
output_key = (step, n_start, col_used, M_step, batch)
```

If the output tile fits in active OSRAM, it can remain resident across K folds. If it does not fit, partial outputs are modeled as needing SRAM output rereads after the first K fold:

```text
if not output_fits and k_fold_idx > 0:
  output_read_bytes += fold_output_bytes
```

Every fold produces output write traffic.

## Output Drain Modeling

Output writeback uses active/drain semantics:

```text
output_active_capacity_bytes
output_drain_capacity_bytes
output_total_capacity_bytes
output_free_bytes
output_drain_end_cycle
```

When active output occupancy crosses the threshold, `_launch_output_drain(...)` starts a drain into DRAM. If a later fold exhausts output free space while a drain is still in flight, `_service_output_write(...)` adds write stall cycles and advances the fold timeline.

At the end of the schedule, `_drain_remaining_output(...)` drains any remaining output data.

The model tracks:

- `write_stall_cycles`
- `output_transfer_window_cycles`
- output DRAM bandwidth

## Schedule Outputs

`_ws_schedule(...)` returns a single `OrderedDict` containing:

- event counts for SRAM and DRAM mapping.
- byte counts for DRAM input, weight, output read, and output write.
- byte counts for SRAM input, weight, and output traffic.
- transfer-window cycle counts.
- compute, read stall, write stall, and phase stall cycle counts.
- mapping efficiency and compute utilization.
- SRAM and DRAM bandwidth metrics.
- cycle runtime conversion.

The public functions consume this shared schedule differently.

## `gemm(...)`

`gemm(...)` builds array mapping subevents:

```text
array_input_mapping
array_weight_mapping
array_compute_mapping
```

It computes logical input, weight, and compute tile counts from array tiling and utilization. It assigns the schedule's compute cycles as factors on the array mapping events.

It also exposes specified metrics such as:

- `compute_cycle_count`
- `stall_cycle_count`
- `read_stall_cycle_count`
- `write_stall_cycle_count`
- phase stall metrics
- SRAM/DRAM bandwidth metrics
- SRAM window cycle metrics
- mapping efficiency
- compute utilization

## `sram(...)`

`sram(...)` builds SRAM mapping subevents:

```text
sram_input_write_mapping
sram_weight_write_mapping
sram_output_read_mapping
sram_output_write_mapping
```

The counts come from the shared schedule's SRAM movement counts. The subevent factors are zero for cycle/runtime because SRAM timing pressure is modeled in the schedule's stall metrics rather than as leaf cycle factors.

It exposes SRAM-side bandwidth and SRAM active-window cycle metrics.

## `dram(...)`

`dram(...)` builds DRAM mapping subevents:

```text
dram_input_read
dram_weight_read
dram_output_read
dram_output_write
```

Input and weight reads receive separate stall-derived factors:

```text
input_read_cycle_factor  = input_read_stall_cycles / input_read_bytes
weight_read_cycle_factor = weight_read_stall_cycles / weight_read_bytes
```

Output writes are marked sequential so they compose after read-side DRAM activity:

```text
dram_output_write:
  aggregation: sequential
```

`dram(...)` exposes:

- memory stall metrics.
- read/write stall metrics.
- phase stall metrics.
- input/weight/output transfer-window cycles.
- input/weight/output DRAM bandwidth.

## What Is Event Based

The current mapped GEMM model is event based because it operates on:

- GEMM events.
- array/SRAM/DRAM mapping events.
- steps and folds.
- operand chunks.
- fold phases.
- analytical service windows.

It does not build:

- per-cycle request traces.
- address demand matrices.
- SCALE-Sim-style `trace_matrix` objects.
- per-address SRAM bank conflict traces.

The closest trace-like behavior is analytical chunk residency, but chunks are only identifiers for fold-level reuse and prefetch accounting.

## Current Modeling Boundaries

The model now captures:

- separate input and weight prefetch engines.
- chunk-level active/prefetch residency.
- double-buffered SRAM capacity.
- output active/drain behavior.
- operand-specific SRAM and DRAM bandwidth metrics.
- fold-level fill/steady/tail phase stall placement.

The model does not yet capture:

- per-address bank conflicts.
- per-cycle demand row shifts.
- exact SCALE-Sim trace timing.
- detailed layout-dependent memory conflicts.

## Simple Flow

1. ArchX enters a GEMM mapping event and calls `gemm(...)`, `sram(...)`, or `dram(...)`.
2. The selected function calls `_ws_schedule(...)` to build one shared mapped-GEMM schedule.
3. `_ws_schedule(...)` normalizes the architecture and reads PE, SRAM, and DRAM parameters.
4. SRAM capacity is split into active and prefetch/drain regions for double-buffer modeling.
5. The GEMM dimensions are expanded into steps when a stepped dimension is configured.
6. Each step is tiled into weight-stationary K/N folds with `_ws_fold_infos(...)`.
7. Each fold computes input, weight, and output element and byte counts.
8. Input and weight chunks are checked against active-buffer residency.
9. Missing input and weight chunks become DRAM prefetch work.
10. Startup-missing chunks determine when the fold can begin compute.
11. Input and weight read engines reserve full prefetch service time independently.
12. The fold starts after prior compute and required input/weight startup data are ready.
13. The fold is split into weight-fill, steady-state, and output-tail phases.
14. SRAM service pressure is assigned to the phase where each operand is consumed or produced.
15. Phase stalls are added to the fold duration using max pressure inside each phase.
16. Output accumulation checks whether the output tile remains resident in active OSRAM.
17. Output writes enter the active/drain output-buffer model.
18. Output drains launch when active occupancy crosses the drain threshold.
19. Output write stalls occur only when output space is exhausted before a drain completes.
20. Final output data is drained after all folds complete.
21. The schedule returns counts, bytes, cycles, stalls, bandwidths, and utilization metrics.
22. `gemm(...)` maps schedule results to array mapping subevents and compute metrics.
23. `sram(...)` maps schedule results to SRAM movement counts and SRAM bandwidth metrics.
24. `dram(...)` maps schedule results to DRAM movement counts, stall factors, and DRAM bandwidth metrics.
25. ArchX aggregates the returned event counts, factors, and specified metrics through the event graph.
