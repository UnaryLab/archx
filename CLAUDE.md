# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Archx is

A cost-modeling framework for computer-system design-space exploration, built around the **A-Graph**. It computes hardware metrics (area, power, energy, cycle count, runtime, any user-defined quantity) for an architecture running a workload by building a directed graph of **events** and hardware **modules** joined by **subevent** edges, pricing the leaf modules through pluggable hardware **interfaces**, running Python **performance models** that set per-edge call counts, then aggregating up the graph.

The graph engine is Rust (PyO3), exposed as `archx._core.ArchxGraph`. Everything else is Python.

## Build

A **maturin/PyO3 hybrid**: Python under `src/archx/`, Rust compiled from `src/archx/rust-core/` into `archx._core`.

```bash
conda env create -f environment.yaml
conda activate archx
pip install -e . --no-deps
```

- **Requires the Rust toolchain** on PATH. Python 3.10+.
- **Editing any `.rs` file requires a rebuild.** Python edits are live; Rust edits are not. Rerun `pip install -e . --no-deps`, or `maturin develop` **from the repo root**, never `maturin develop -m src/archx/rust-core/Cargo.toml`, which bypasses `pyproject.toml` and builds a stray `archx_core` package instead of `archx._core`.
- The version is declared in **three** places that must agree: `pyproject.toml`, `src/archx/rust-core/Cargo.toml`, and the `archx-core` entry in `Cargo.lock`. So is the ABI target: `pyproject.toml`'s `[tool.maturin] features` and `Cargo.toml`'s pyo3 features both name `abi3-py310`. pyo3 takes the **lowest** `abi3-pyXX` enabled, so a stale one in `Cargo.toml` silently overrides `pyproject.toml`.
- `ortools` installs through the `pip:` subsection of `environment.yaml`; the conda package under that name is a different project.

## Tests

```bash
pytest                                        # full suite, testpaths = tests/
pytest tests/test_mac_1_cycle.py              # one file
pytest tests/test_mac_1_cycle.py::test_area   # one test
```

Most test files run the **whole pipeline at module import time** against `examples/mac_1_cycle/`, then assert in individual `test_*` functions; a trailing `test_cleanup()` removes the run directory, so it only exists mid-run. The Rust side has one unit test (`metric.rs::matches_python_repr`, pinning float formatting to Python's `repr`, run with `cargo test`); the engine itself is exercised through the Python suite.

- `tests/test_numerical_equivalence.py` is the regression pin for the Rust aggregation engine. It builds an A-Graph directly through the `archx._core` API with no hardware interface and checks every aggregation mode against hand-derived values. It needs no CACTI.
- `tests/test_description.py` compiles `examples/systolic_array/description.py` and asserts the sweep emits exactly 256 configurations. It is the only coverage of the `archx.programming` frontend.

**CACTI7 is host-specific.** `cacti7.py` looks for `cacti-<system>-<machine>` (`cacti-Linux-x86_64`, `cacti-Darwin-arm64` are tracked) and falls back to `make all` when the host's binary is absent. There is no Windows binary and its makefile is GCC-specific, which is why CI covers Linux and macOS only. Any `test_mac_*` touches `sram` through `cacti7`.

**`graphviz` is optional.** `utils.draw_event_graph` logs an error and returns when it is missing; it is declared in `environment.yaml` but not `pyproject.toml`, so the PDF test skips under a bare pip install.

## The pipeline (7 stages)

`main.py` and every test follow the same sequence, one subpackage per stage:

1. **architecture** → `create_architecture_dict`. Parses the architecture YAML into a *flat* dict keyed by module name. Modules may `path:` to other YAML files. Global `attribute` defaults (technology, frequency, interface) are merged into each module's `query`; `tag` lists and `instance` lists propagate down.
2. **metric** → `create_metric_dict`. Each metric declares a `unit` and an `aggregation` mode. Exactly three are legal: **`module`** (count each distinct module once, for area and leakage), **`summation`** (scale by per-edge call counts, the default, for energy), **`specified`** (taken straight from a performance-model output, for cycle count and runtime). Tag aggregation allows only the first two.
3. **workload** → `create_workload_dict`. **One workload per file**: a `name` and a `configuration` dict. A file may instead carry `path:`, which is followed recursively until a file with a `name` is reached. This FLAT schema is repo-wide; performance models read `workload_dict['configuration']` directly.
4. **event** → `create_event_graph`. Builds the `ArchxGraph`: event nodes carrying a `performance:` file, then leaf module nodes, then `event → subevent` edges.
5. **metric population** → `create_event_metrics` calls `query_interface` per leaf module and writes results in through `set_module_data`.
6. **performance** → `simulate_performance_all_events` runs each event's model, setting per-edge `count`/`operation`/`factor` and per-node `specified` metrics.
7. **aggregation** → `aggregate_event_metric`, `aggregate_tag_metric`, `query_module_metric` walk the graph and return `{value, unit}`.

Checkpoints are plain JSON via `save_json`/`load_json`, the file passed to `-c`.

### The Rust core (`src/archx/rust-core/src/`)

- `lib.rs`: the `ArchxGraph` PyO3 class and **the entire Python-to-Rust contract**. Changing a signature here means updating callers in `event/`, `metric/`, and `performance/`.
- `graph.rs`: `ArchxGraphInner`, a petgraph `DiGraph` plus a name-to-index map. Node payloads hold metrics, instance counts, tags; edge payloads hold count, aggregation, operation, factor.
- `metric.rs`: `MetricValue` is either `Single{value,unit}` or `MultiOp{op -> SingleMetric}` for multi-operation modules such as SRAM read vs write.
- `aggregate.rs`: the algorithms. `aggregate_summation_multiop` is an internal helper for multi-op metrics, **not** a user-facing aggregation mode.
- `paths.rs`: path enumeration.

**Logging bridge:** Rust logs through the `log` crate, `pyo3_log` forwards to stdlib `logging`, and `archx/__init__.py` installs an `InterceptHandler` that redirects that into **loguru**. Rust and Python therefore share one stream, filtered by loguru sinks. When adding Rust traces use `log::debug!/info!/warn!` and keep wording aligned with the Python side.

### Interfaces (`src/archx/interface/`)

`query_interface` imports `interface/<name>/<name>.py` and calls its `query(module, interface, query, input_dir, output_dir)`. A module picks one with `interface:` in its `query`, usually inherited from the architecture `attribute`.

Each CSV backend hardcodes its data directory relative to its own `__file__`; the `input_dir` parameter is accepted everywhere and read nowhere. The backends are split **by technology library, not by design**: `csv_cmos` (45 nm and 7 nm), `csv_cmos_32nm`, `csv_cmos_asplos_2026_ae` (a second 45 nm library), plus `cacti7`, `chiplet_cmos`, `csv_sc`, `csv_h200` (measured GPU kernels), `csv_riscv`.

### Caching

Two persistent caches under `~/.cache/archx/`, both keyed so a changed input invalidates the entry:

- **interface**: keyed on the query plus a fingerprint of **every file in the interface directory**, since interfaces are mostly data (one `.py` among hundreds of CSVs).
- **performance**: keyed on the model file plus **every project module it imports, transitively**, so editing a shared helper invalidates dependents. Library modules are excluded.

Both fingerprint by mtime and size, memoized per process, so edits mid-run are not seen. Relocate with `ARCHX_INTERFACE_CACHE_DIR` / `ARCHX_PERFORMANCE_CACHE_DIR`, or bypass the second with `ARCHX_DISABLE_PERFORMANCE_CACHE=1`. Each cache carries a version integer for invalidating entries when the caching logic itself changes.

### Frontend (`src/archx/programming/`)

`graph/agraph.py::AGraph` drives sweeps using **OR-Tools CP-SAT** to enumerate valid combinations under constraints (`direct_constraint`, `conditional_constraint`). A user `description.py` defines `description(path)`, builds an `AGraph`, and emits `configurations.csv`. `programming/object/` holds the `Architecture`/`Event`/`Metric`/`Workload` builders.

Two API shapes bite: `add_module` returns a **name-keyed** dict for a list `name` and a flat one for a string; query parameters nest under `['query']`; and `add_parameter` returns `{'parameter': ...}`, not a name-keyed dict.

## Design zoos (`zoo/`)

Each is a copy of another branch, kept faithful to it, so **do not tidy `zoo/` as part of unrelated work**: byte-fidelity to the source branch is the property that makes them auditable.

- `zoo/agraph/`: 13 A-Graph designs (FFT, systolic, TNN, stochastic computing, MUGI, RISC-V, three GPU), from `micro_agraph`.
- `zoo/chiplet4ai/`: a chiplet study for Llama, from `rust_fix`.
- `zoo/llm/`: five accelerators swept over LLM workloads, from `asplos_2026_ae`. `scripts/generate_dicts.py` fans templates out into `*/generated/`, which is gitignored and rebuilt, never committed. Its workload **templates are intentionally NESTED**: they are model catalogues read as plain YAML by the generator, not through `create_workload_dict`.

## CLI

```bash
archx -r <run_dir> -a arch.yaml -m metric.yaml -w workload.yaml -e event.yaml -c out.json [-t] [-s]
```
`-t` mirrors logs to the terminal (a logfile always lands in the run dir); `-s` dumps the parsed dicts as YAML; `-p` adds a directory to `sys.path` for performance-model imports.

```bash
archx -r <dir> -compile description.py [-full] [-ff]   # -> configurations.csv
archx -r <dir> -extract configurations.csv             # -> runs.txt
archx -r <dir> -x runs.txt                             # batch execute
archx -ireg -iname <name> -idir <dir>                  # register an interface
```
`bash src/archx/bin/run_archx.sh runs.txt` fans runs across cores and logs failures to `failed_runs.txt`.

## Conventions

- YAML is read with `yamlordereddictloader` to preserve order; dicts flow as `OrderedDict`.
- Logging is **loguru**; stages emit `logger.success` banners and assertions use `logger.error(...)` as the message.
- Resolve paths through `archx.utils.get_path`, not raw paths.
- `utils/__init__.py` is `from .utils import *`, so everything in `utils.py` is public API even when nothing in-tree calls it.
