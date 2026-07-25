# Archx examples

Small, self-contained designs that exercise the full Archx pipeline. Each `input/` directory holds the four YAML inputs (architecture, metric, workload, event) and the performance models for one design; the matching `tests/test_<name>.py` runs the same inputs through the Python API.

## Contents

- `mac_1_cycle/`: the canonical minimal example. A MAC array (multipliers + adders, priced via the `csv_cmos_32nm` interface) with an SRAM (priced via `cacti7`) running GEMM workloads; every MAC completes in one cycle.
- `mac_1_cycle_factor/`: `mac_1_cycle` with per-edge `factor` scaling in the performance models.
- `mac_2_cycle/`: the same design with `aggregation: sequential` on the multiplier/adder and SRAM-write edges, so their cycle counts add up (a two-cycle MAC) instead of overlapping.
- `systolic_array/`: a design-space sweep. `description.py` enumerates 256 systolic-array configurations with the `archx.programming` frontend, priced via `csv_cmos` at 45 nm; `performance.py` holds the shared performance models.
- `run_batch.args`: a sample line format for batch runs (`archx -x runs.txt`).

Each workload file holds one workload under a `name` and a `configuration` dict. Where a design has a second workload it gets its own workload and event file, as `mac_1_cycle`'s `example_gemm32.workload.yaml` does; a workload file may also `path:` to another one.

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

Batch runs read a file of argument lines, one run per line:

```bash
bash src/archx/bin/run_archx.sh examples/run_batch.args
```

## Querying the results

After a run, load the `.json` checkpoint and aggregate metrics; see "Querying results" in the top-level [README](../README.md) or the assertions in `tests/test_mac_1_cycle.py` for worked queries (area per event, dynamic energy per workload, runtime, tag-based aggregation).
