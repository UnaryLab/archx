# Archx examples

Small, self-contained designs that exercise the full Archx pipeline. Each `input/` directory holds the four YAML inputs (architecture, metric, workload, event) and the performance models for one design; the matching `tests/test_<name>.py` runs the same inputs through the Python API.

## Contents

- `mac_1_cycle/`: the canonical minimal example. A MAC array (multipliers + adders, priced via the `csv_cmos` interface) with an SRAM (priced via `cacti7`) running GEMM workloads; every MAC completes in one cycle.
- `mac_1_cycle_factor/`: `mac_1_cycle` with per-edge `factor` scaling in the performance models.
- `mac_2_cycle/`: the same design with `aggregation: sequential` on the multiplier/adder and SRAM-write edges, so their cycle counts add up (a two-cycle MAC) instead of overlapping.
- `systolic_array/`: a design-space sweep. `description.py` enumerates systolic-array configurations with the `archx.programming` frontend; `performance.py` holds the shared performance models.
- `run_batch.args`: a sample line format for batch runs (`archx -x runs.txt`).

## Running an example

Single run, from the repository root:

```bash
archx -r runs/mac_1_cycle \
      -a examples/mac_1_cycle/input/architecture/example.architecture.yaml \
      -m examples/mac_1_cycle/input/metric/example.metric.yaml \
      -w examples/mac_1_cycle/input/workload/example.workload.yaml \
      -e examples/mac_1_cycle/input/event/example.event.yaml \
      -c runs/mac_1_cycle/graph.json -t -s
```

Sweep example:

```bash
archx -r runs/systolic -compile examples/systolic_array/description.py -full
```

> **Known issue:** these examples have not yet been migrated to the current input formats. The `mac_*` workload files use a multi-workload-per-file layout, while the current parser expects one workload per file with `name:` and `configuration:` keys (see the workload section of the top-level README), so those runs and their tests fail at the workload-parsing step. `systolic_array/description.py` likewise predates the current `archx.programming` API and fails in `-compile`. `chiplet4ai/` uses the current formats.

## Querying the results

After a run, load the `.json` checkpoint and aggregate metrics; see "Querying results" in the top-level [README](../README.md) or the assertions in `tests/test_mac_1_cycle.py` for worked queries (area per event, dynamic energy per workload, runtime, tag-based aggregation).
