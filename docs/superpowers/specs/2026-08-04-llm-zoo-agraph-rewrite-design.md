# LLM zoo → AGraph `description.py` rewrite

**Date:** 2026-08-04
**Status:** Approved (design), pending implementation plan

## Goal

Rewrite `zoo/llm/` so each design is **defined programmatically** through the AGraph
frontend (`archx.programming`, a `description.py` per design compiled with
`archx -compile … -full`), matching the structure already used in `zoo/agraph/designs/`.
Replace the YAML-template + `generate_dicts.py`/`generate_runs.py` fan-out entirely.

## Decisions (locked)

1. **Scope:** all 5 designs in one pass (carat, mugi, simd, systolic, tensor).
2. **Granularity:** one `description.py` per **event topology** = `(subarchitecture × network-mode)`.
   The AGraph frontend emits exactly one `event.yaml` per description, so every distinct
   event graph needs its own description, the same reason `zoo/agraph` splits `systolic_cg`/`systolic_fg`.
3. **Old machinery:** replace fully. Delete the per-design `template/` trees,
   `scripts/generate_dicts.py`, `scripts/generate_runs.py`, `scripts/architecture_configuration.yaml`;
   rewrite `llm_script.sh` to the compile flow; repoint `results/` queries to the new runs layout.
4. **Fidelity:** structure-faithful + spot-checked. Transcribe the per-size instance/width/depth
   values from the templates; verify by compiling each description and running a few configs,
   **not** by full oracle-equivalence against the old fan-out.

## Topology enumeration (20 descriptions)

| family   | subarchitectures        | network modes    | count |
|----------|-------------------------|------------------|-------|
| systolic | mac, figna, pwl, taylor | single, multi    | 8     |
| simd     | mac, figna              | single, multi    | 4     |
| mugi     | lut, vlp                | single, multi    | 4     |
| carat    | (none)                  | single, multi    | 2     |
| tensor   | (none)                  | single, multi    | 2     |

- **single**: no router module/events; node dimension `[]` (array-only instance).
- **multi**: adds router module + `irouter/wrouter/orouter` events; node dims prepended to
  `instance` (systolic/simd/mugi/carat: `[4,4]`,`[8,8]`; tensor: `[2,1]`,`[2,2]`).

## Target layout

```
zoo/llm/
  common/
    description/          # NEW shared builders (import from each description.py)
      __init__.py
      metric.py          # add_metric ×5
      workload.py        # model catalogue + shared swept config → workload configs
      software_events.py # per-family software event tree + leaf→hardware/memory/router wiring
      memory.py          # memory modules + memory events
      router.py          # router module + router events (multi only)
    performance/ …       # UNCHANGED
    metric/metric.yaml   # kept for reference; metrics now emitted by description
  designs/
    <family>/
      performance/…      # shared per family (moved up from the old <design>/performance)
      <topology>/description.py
      <topology>/query.py
  llm_script.sh          # compile → run → results
  results/               # queries repointed to runs/<workload>/arch_<i>/config_<j>/checkpoint.json
```

## Per-`description.py` contract

Mirror `zoo/agraph/designs/systolic_cg/description.py`:

1. `agraph = AGraph(path); architecture, event, metric, workload = agraph.…`
2. `architecture.add_attributes(technology=45, frequency=400, interface='csv_cmos_asplos_2026_ae')`
3. Architecture modules built inline from the template fragments
   (`memory.py` helper for dram/isram/wsram/osram; fifo; pe cluster; array; vector/tc;
   subarch modules for this topology; `router.py` helper if multi). The array-size axis
   is a swept list per module; node dims folded into `instance` for multi.
4. Events: `software_events.build(event, …, hw_events, mem_events, router_events)` +
   this topology's hardware events (`gemm`/`silu`/`softmax` or `gemm`/`nonlinear`).
5. `metric.py` helper adds the 5 metrics.
6. `workload.py` helper adds the model catalogue + shared swept config, returning handles.
7. One `direct_constraint([...])` tying every array-axis IntVar to a shared index.
8. `return agraph.generate()`

## Driver

`llm_script.sh`:
```bash
export PYTHONPATH="$(pwd)/zoo:$PYTHONPATH"
for dir in zoo/llm/designs/*/*/; do            # each topology folder with a description.py
  [ -f "$dir/description.py" ] || continue
  archx -compile "$dir/description.py" -r "$dir/description" -full
done
# concatenate runs.txt → run_archx.sh → results
```

## Results

AGraph writes runs under `<run_dir>/runs/<workload_name>/arch_<i>/config_<j>/` with
`checkpoint.json`. The `results/query/*.py` + `figure_generation.py` currently walk the old
`runs/<arch>/<net_dim>/<subarch>/<array>/<model>/…` tree; repoint them to the AGraph layout
(driven off each topology's `configurations.csv`, which already carries `run_path`/`checkpoint_path`).

## Verification (spot-check)

Per topology: `archx -compile … -full` must succeed and emit `configurations.csv` + `runs.txt`;
then execute a handful of runs and confirm non-error `{value, unit}` metrics. Confirm the
5 families' full driver produces the expected topology count. No full oracle gate.

## Non-goals

- No change to `src/archx/` (engine, frontend, interfaces, performance framework).
- No change to other zoos.
- Performance-model `.py` files are moved, not rewritten.
