# A-Graph case studies

Case-study designs ported from the `micro_agraph` branch. Each design under
`designs/` is a self-contained cost-model study: an architecture description, an
event graph, a performance model, and a query script that aggregates per-run
metrics into `results/*.csv`. The plotting scripts in `res/scripts/` turn those
CSVs (and the runtime logs in `runtime/`) into the figures in `res/figures/`.

This tree ships **source only**. The generated `designs/<name>/description/`
intermediates, the `designs/<name>/results/` CSVs, and `res/figures/` are all
produced by running the flow below; `runtime/` holds the external EDA
measurements, `designs/riscv/if_id_trace.html` is a design input, and
`interface/` holds the characterization bundles the designs query (see
Interfaces below).

## Running a design

```bash
archx -compile designs/<name>/description.py -r designs/<name>/description -full
python designs/<name>/query.py
```

`-compile ... -full` generates the run configs and executes every run, writing a
`checkpoint.json` per run under `designs/<name>/description/runs/<event>/arch_<i>/config_<j>/`
(these run outputs are git-ignored). `query.py` reads those checkpoints and
writes `designs/<name>/results/*.csv`.

Regenerate all figures after the designs have been run:

```bash
for s in res/scripts/*.py; do python "$s"; done
```

`agraph.sh` runs the full sweep (compile + query for every design, then every
plotting script).

## Designs that run on this branch (verified)

These 8 designs run against this branch's `archx` and, when generated, reproduce
the original `micro_agraph` `results/` (to floating-point precision) and all five
figures:

- `fft_cg`, `fft_fg`
- `tnn_cg`, `tnn_fg`
- `sc_cnn`, `sc_fir`
- `systolic_cg`, `systolic_fg`

## Interfaces (`interface/`)

Every hardware-characterization interface the designs query is bundled under
`interface/<name>/` as a complete, registerable directory (query shim
`<name>.py` + its `include/csv` data). This makes the case studies self-contained
rather than depending on characterization data baked into the framework package.

| Interface | Used by | Notes |
| --- | --- | --- |
| `csv_cmos` | fft, tnn, sc_cnn, systolic, mugi | 56 classes (completed with mugi's `vlp_*`/`exp_*`) |
| `csv_sc` | sc_cnn, sc_fir | superconducting stochastic-computing cells |
| `cacti7` | systolic, mugi | analytical SRAM/DRAM model (config-driven, no CSV) |
| `csv_riscv` | riscv | RISC-V pipeline per-instruction characterization |
| `csv_h200` | gpu_gpt2, gpu_llama3_2, gpu_qwen2_5 | per-kernel GPU characterization (~4,276 CSVs) |

Each bundle was verified to load and answer a query through this branch's
`archx`. Register one into the framework before running the designs that use it:

```bash
archx -register_interface <name> agraph/interface/<name>   # e.g. csv_h200
# ... run designs ...
archx -unregister_interface <name>                         # to remove it again
```

`csv_cmos`, `csv_sc`, and `cacti7` already ship inside `src/archx/interface/` on
this branch, so `agraph.sh` runs the 8 CMOS/SC designs without any registration;
registering `csv_riscv` and `csv_h200` (and the completed `csv_cmos`, if mugi
needs the added classes) enables `riscv`, the three GPU designs, and `mugi`.

Note: the GPU/riscv/mugi **design** code is ported but was not run end-to-end on
this branch, so it may still need the same serialization adaptations described
below (the interface bundles themselves are verified).

## Note on the port

This branch (`agraph_hpca_2027`) changed the A-Graph serialization contract
relative to `micro_agraph`, so the ported case-study code was adapted to match:

- `performance.py`: the per-run workload dict is now `{configuration, name}`
  instead of being keyed by the configuration name, so
  `workload_dict['<cfg>']['configuration']` became `workload_dict['configuration']`.
- `description.py` (systolic): `add_parameter(...)` now returns
  `{'parameter': <var>}`, so the workload sweep variable is referenced as
  `param['parameter']`.
- `query.py`: run outputs moved to `runs/<event>/arch_<i>/config_<j>/` with a
  JSON checkpoint (`checkpoint.json`), replacing the old
  `runs/config_<i>/checkpoint.gt` layout.

The 5 backend-dependent designs above may need the same serialization
adaptations once their interface libraries are available; those edits could not
be verified here without the backends.
