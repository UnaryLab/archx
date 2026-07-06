# Fixable SCALE-Sim Memory-Model Gaps

This file lists the main differences between the current ArchX/Chiplet4AI event-based memory model and SCALE-Sim that can be improved without generating full traces.

## 1. Split IFMAP and Weight Prefetch Scheduling

Status: implemented in `chiplet4ai/common/performance/mapping.py`.

Current model:
- IFMAP and weight DRAM reads use separate prefetch cycle calculations.
- IFMAP and weight use separate read engine availability times.
- The fold-level read stall is the max of input and weight readiness stalls.
- Separate `input_read_stall_cycle_count` and `weight_read_stall_cycle_count` metrics are exposed.

Why this differs from SCALE-Sim:
- SCALE-Sim has separate IFMAP and filter read buffers.
- Each buffer has its own active/prefetch state and backing-port service timing.
- The memory system services IFMAP, filter, and OFMAP each cycle and applies the maximum stall among them.

Event-based fix:
- Track separate `ifmap_read_engine_available_cycle` and `weight_read_engine_available_cycle`.
- Compute IFMAP and weight prefetch cycles independently.
- For a fold, set read stall from the max of IFMAP readiness and weight readiness.
- Expose separate `input_read_stall_cycle_count` and `weight_read_stall_cycle_count` metrics.

Expected benefit:
- More realistic operand-specific bandwidth and stall attribution.
- Less artificial coupling between input and weight movement.

## 2. Use Chunk-Level Active/Prefetch Buffer Residency

Status: implemented in `chiplet4ai/common/performance/mapping.py`.

Current model:
- IFMAP and weight residency is tracked at chunk granularity.
- Chunks are sized by the active/prefetch buffer capacities.
- The full missing chunk traffic is scheduled on the operand read engine, but only the missing active-window chunks gate compute start.

Why this differs from SCALE-Sim:
- SCALE-Sim partitions operand matrices into active and prefetch buffer sets.
- The active buffer advances through sets as demand progresses.
- Partial residency and rolling prefetch windows create gradual capacity effects.

Event-based fix:
- Split each fold's IFMAP and weight data into chunks sized by prefetch-buffer capacity or active-buffer capacity.
- Track active chunk windows rather than whole fold tiles.
- Model how many chunks are already resident and how many must be prefetched for the next fold/window.
- Keep this analytical: no per-address trace, only chunk IDs and chunk sizes.

Expected benefit:
- SRAM-size sweeps should become more gradual.
- Required bandwidth should change smoothly when capacity changes, not only at large tile-fit thresholds.

## 3. Make Output Buffer Modeling Follow Active/Drain Semantics

Status: implemented in `chiplet4ai/common/performance/mapping.py`.

Current model:
- Output writes track total free space, active-buffer threshold, drain-buffer chunk size, and drain completion cycle.
- A drain is launched when active occupancy crosses the threshold.
- If output production exhausts free space while a drain is still in flight, the fold timeline stalls until the drain completes.
- Final drain waits for any in-flight drain and drains remaining active-buffer contents.

Why this differs from SCALE-Sim:
- SCALE-Sim's write buffer has an active region and a drain region.
- Writes fill the active buffer.
- When active occupancy crosses a threshold, a drain operation is started.
- Stalls occur only when insufficient free space remains before the current drain completes.

Event-based fix:
- Track output active-buffer free space and drain-buffer occupancy separately.
- Trigger drain events when active-buffer occupancy exceeds the drain threshold.
- Maintain `output_drain_end_cycle`.
- Stall only when a fold produces output while free space is exhausted and the drain has not completed.

Expected benefit:
- More accurate write stalls and OFMAP bandwidth.
- Better agreement with SCALE-Sim's output-buffer drain behavior.

## 4. Expose Operand-Specific SRAM and DRAM Bandwidth Metrics

Status: implemented in `chiplet4ai/common/performance/mapping.py` and `chiplet4ai/llama/query/fig_2_query.py`.

Current model:
- IFMAP, weight, and output DRAM bandwidth are exposed as specified metrics.
- IFMAP, weight, and output SRAM-side bandwidth are exposed as specified metrics.
- Transfer-window metrics are exposed for DRAM movement, and active SRAM window metrics are exposed for SRAM-side weighted averages.
- Fig. 2 uses model-provided DRAM bandwidth metrics for input/weight movement while preserving the legacy CSV column names as aliases.

Why this differs from SCALE-Sim:
- SCALE-Sim reports average IFMAP, filter, and OFMAP bandwidth separately.
- It distinguishes SRAM-side bandwidth from DRAM-side bandwidth.

Event-based fix:
- Add specified metrics from `_ws_schedule(...)` for:
  - `input_sram_bandwidth`
  - `weight_sram_bandwidth`
  - `output_sram_bandwidth`
  - `input_dram_bandwidth`
  - `weight_dram_bandwidth`
  - `output_dram_bandwidth`
- Use transfer-window or active-service denominators consistently.
- Update query scripts to plot the intended metric explicitly.

Expected benefit:
- Fig. 2 can distinguish SRAM demand pressure from DRAM prefetch pressure.
- Results become easier to compare against SCALE-Sim's bandwidth reports.

## 5. Add Analytical SRAM Bank-Conflict Penalties

Current model:
- SRAM service rate is modeled as:
  - half the banks times width per cycle.
- It assumes ideal distribution over banks.

Why this differs from SCALE-Sim:
- SCALE-Sim can model per-cycle bank conflicts when layout evaluation is enabled.
- Multiple requests to the same bank can serialize.

Event-based fix:
- Add an analytical bank-conflict factor.
- Estimate conflict probability from concurrent accesses, bank count, and assumed layout distribution.
- Inflate SRAM stream cycles by this factor.
- Optionally expose a per-SRAM conflict penalty metric.

Expected benefit:
- More realistic SRAM-side stalls for large array widths or low bank counts.
- Still avoids address-level traces.

## 6. Improve Fold-Level Bubble/Stall Placement

Status: implemented in `chiplet4ai/common/performance/mapping.py`.

Current model:
- Each fold is split into weight-fill, steady-state, and output-tail phases.
- SRAM stream pressure is assigned to the phase where that operand is needed.
- Per-phase stalls are composed with max pressure within the phase and summed across sequential phases.
- Phase stall metrics are exposed as specified metrics.

Why this differs from SCALE-Sim:
- SCALE-Sim services demand rows cycle by cycle.
- IFMAP, filter, and OFMAP stalls can appear at different cycles and shift later demand rows.

Event-based fix:
- Keep fold-level execution, but split each fold into phases:
  - initial weight fill
  - steady-state compute
  - output drain/fill tail
- Assign stalls to the phase where the operand is needed.
- Use max-stall composition across IFMAP, weight, and output phases.

Expected benefit:
- Better cycle-count decomposition.
- Better memory-stall attribution while remaining trace-free.

## Recommended Implementation Order

1. Split IFMAP and weight prefetch scheduling.
2. Add chunk-level active/prefetch residency.
3. Rework output buffer active/drain semantics.
4. Expose operand-specific SRAM/DRAM bandwidth metrics.
5. Add analytical bank-conflict penalties.
6. Improve fold-level phase/bubble attribution.
