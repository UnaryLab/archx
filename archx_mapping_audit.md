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

### 2. Per-event `stall_cycle_count` is not an active ArchX metric

`mapping.py` returns `stall_cycle_count`, `read_stall_cycle_count`, and `write_stall_cycle_count`. ArchX stores them as specified metrics, but `description.py` only declares:

```python
cycle_count
runtime
```

So those stall metrics are informational only unless they are added to the metric set or folded into `cycle_count` / `runtime` through edge factors. Right now `dram(...)` tries to fold stall into `cycle_count`, but the named stall metrics themselves are not queryable via the current metric YAML.

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

### 5. Lower-level `node.py` still encodes conceptual access assumptions

`array_compute_mapping` still expands to:

```text
array_compute
sram_output_read
sram_output_write
```

with fixed access expansion. Timing has been normalized, but there is still a conceptual mismatch between which level accounts for output reads/writes as traffic versus schedule time.

### 6. The memory stall model is not SCALE-Sim-accurate

It estimates stalls from fold bytes and DRAM bandwidth overlap. It does not model:

- active/prefetch buffer capacity
- address reuse/hits
- bank conflicts
- SRAM ports
- output write-buffer occupancy
- true IFMAP/filter prefetch ordering

So it is analytical, not simulated SCALE-Sim-equivalent.

## Bottom Line

- Accurate fill/drain: mostly yes for array compute folds.
- Accurate memory stalls: no, only a first analytical estimate.
- Accurate end-to-end runtime/cycles in ArchX: likely closer after Issue 1, but still needs a direct aggregate validation.
- Accurate per-event counts: not uniformly. Some counts are normalized chunk counts used to recover traffic after lower-level expansion, not literal event counts.
- Matches `description.py` structurally: yes.
- Matches `description.py` semantically: partially; `node.py` and `mapping.py` now disagree in places about output read/write timing.

## Most Important Next Fix

Validate the timing aggregation end to end and then make metric declarations expose the stall counters that `mapping.py` already computes.

Recommended direction:

- Confirm `mapping.py` owns all schedule timing in the aggregate output.
- Keep `node.py` as access/energy expansion with unit or zero timing factors.
- Add stall metrics to `description.py` / metric YAML if they should be queried directly.
