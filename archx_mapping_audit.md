# ArchX WS Mapping Audit

## Quick Verdict

The current implementation is closer to correct for accurate end-to-end runtime and cycle count after Issue 1 was addressed. The mapping math is moving in the right direction, but several validation and modeling issues remain.

## What Is OK

- It still matches the structural event names in `chiplet4ai/llama/description.py`: `*_arr`, `*_sram`, `*_dram`, `array_*_mapping`, `sram_*_mapping`, and `dram_*`.
- `llama_model.py` call signatures still match the description.
- ArchX can consume the returned `count`, `aggregation`, `factor`, and extra specified metrics.
- `mapping.py::gemm(...)` now calculates WS fill/drain compute cycles per fold.

## Issues

### 1. End-to-end `cycle_count` timing ownership

Status: addressed in `chiplet4ai/common/performance/node.py`.

`description.py` makes each op sequential:

```text
proj_q_pf = proj_q_pf_arr -> proj_q_pf_sram -> proj_q_pf_dram
```

The intended model sets:

- `*_arr` to compute cycles
- `*_sram` to zero timing
- `*_dram` to residual stall timing

That can work if `mapping.py` owns schedule timing and the lower mapping nodes only expand access/energy counts. `node.py` has now been normalized so the array and SRAM mapping events contribute unit timing through metric-specific `cycle_count` / `runtime` factors while preserving their dynamic-energy access counts.

Remaining check: run an end-to-end ArchX aggregate on a small GEMM and verify that `*_arr + *_dram` equals the analytical schedule's compute cycles plus residual stall cycles.

### 2. Per-event stall visibility

Status: addressed through structural cycle queries in `chiplet4ai/llama/query/utils.py`.

`mapping.py` returns `stall_cycle_count`, `read_stall_cycle_count`, and `write_stall_cycle_count`. ArchX stores them as specified metrics, but `description.py` only declares:

```python
cycle_count
runtime
```

Adding those as independent `specified` metrics is not straightforward because ArchX's specified aggregation only accepts injected specified values on events that connect directly to leaf modules. Most GEMM timing events are wrappers around child events, so exposing stall by adding metric declarations would either be ignored or rejected during aggregation.

Instead, the query layer now exposes the schedule decomposition from the existing event structure:

```text
compute_cycle_count       = sum(cycle_count(*_arr))
sram_cycle_count          = sum(cycle_count(*_sram))
memory_stall_cycle_count  = sum(cycle_count(*_dram))
unattributed_cycle_count  = cycle_count(parent) - components
```

This keeps `cycle_count` as the authoritative ArchX timing metric while making compute and memory-stall components visible for validation.

### 3. Runtime units are suspect

For array paths, `mapping.py` sets runtime factors equal to cycle factors. That works only because leaf array events have runtime in ms-per-cycle.

For DRAM paths, runtime factor is converted relative to `memory.py`'s per-byte runtime. This is workable, but easy to get wrong and should be validated with a direct aggregate check.

### 4. Event counts are not fully actual event counts anymore

For SRAM mapping, counts are normalized by SRAM capacity:

```text
input_elements / isram_elements
```

Then `node.py` expands one `sram_input_write_mapping` into `isram_depth * bank/2` writes. Multiplying them recovers roughly element traffic, but the intermediate event count is "number of full SRAM-capacity chunks", not actual SRAM access count.

So `aggregate_event_count` for these mapping nodes is not semantically an actual access count.

### 5. Lower-level output SRAM ownership

Status: addressed in `chiplet4ai/common/performance/node.py`.

`array_compute_mapping` still expands to:

```text
array_compute
sram_output_read
sram_output_write
```

with fixed access expansion. Timing had already been normalized. The remaining issue was dynamic-energy ownership: output SRAM traffic could be counted through both the array compute path and the explicit SRAM path.

The output SRAM edges under `array_compute_mapping` now set `dynamic_energy` factor to zero, so:

```text
array path = compute/FIFO/register energy and timing
sram path  = SRAM read/write energy
dram path  = DRAM traffic and stall timing
```

### 6. The memory stall model is not SCALE-Sim-accurate

Status: addressed in `chiplet4ai/common/performance/mapping.py` with a trace-free cycle-demand approximation.

The previous model estimated stalls from fold bytes and DRAM bandwidth overlap only. The schedule model now tracks:

- active/prefetch buffer capacity
- IFMAP and weight residency/reuse
- SRAM service limits derived from bank/width
- output write-buffer occupancy
- ordered prefetch readiness across folds

It is still analytical, not trace-based SCALE-Sim, but the model now has the main state variables needed to approximate SCALE-Sim-style memory stalls without generating traces.

## Bottom Line

- Accurate fill/drain: mostly yes for array compute folds.
- Accurate memory stalls: improved trace-free approximation with active/prefetch buffers, reuse, SRAM service limits, and output buffer occupancy.
- Accurate end-to-end runtime/cycles in ArchX: likely closer after Issue 1; Issue 2 now exposes component cycle queries for direct aggregate validation.
- Accurate per-event counts: not uniformly. Some counts are normalized chunk counts used to recover traffic after lower-level expansion, not literal event counts.
- Matches `description.py` structurally: yes.
- Matches `description.py` semantically: closer after Issue 5; output SRAM energy ownership now sits on the explicit SRAM path.

## Most Important Next Fix

Validate the timing aggregation end to end using the structural cycle breakdown.

Recommended direction:

- Confirm `mapping.py` owns all schedule timing in the aggregate output.
- Keep `node.py` as access/energy expansion with unit or zero timing factors.
- Use `query_cycle_breakdown(...)` to check whether `cycle_count(parent)` decomposes into `*_arr + *_sram + *_dram`.
