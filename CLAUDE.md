# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Archx is

Archx is a cost-modeling framework for computer-system design-space exploration, built around the **A-Graph** abstraction. It computes hardware metrics (area, power, energy, cycle count, runtime, …) for an architecture running a workload by:
1. building a directed graph of **events** and **hardware modules** (nodes) connected by **subevent** edges,
2. populating leaf (module) nodes with metrics queried from pluggable hardware **interfaces** (CACTI7, CMOS synthesis CSVs, stochastic-computing CSVs, …),
3. running user-supplied Python **performance models** that set per-edge call counts, and
4. **aggregating** metrics up the graph to answer queries.

The graph engine is implemented in Rust (PyO3) and exposed to Python as `archx._core.ArchxGraph`. Everything else is Python.

## Build & development

This is a **maturin/PyO3 hybrid**: pure-Python source under `src/archx/`, plus a Rust extension compiled from `src/archx/rust-core/` into the `archx._core` module.

```bash
conda env create -f environment.yaml   # provides python, maturin, graphviz, etc.
conda activate archx
pip install -e . --no-deps              # editable install; compiles the Rust extension
```

- **Requires the Rust toolchain** (`rustup`) on PATH to build.
- **After editing any `.rs` file you must recompile** - Python changes are live (editable install) but Rust changes are not. Rebuild with `pip install -e . --no-deps`, or `maturin develop` run from the repo root (NOT `maturin develop -m src/archx/rust-core/Cargo.toml`, which bypasses `pyproject.toml` and builds a stray `archx_core` package instead of `archx._core`). Forgetting this means you are running stale Rust.
- `module-name = "archx._core"` and `python-source = "src"` are set in `pyproject.toml`; the Rust manifest is `src/archx/rust-core/Cargo.toml`.
- `ortools` is installed via the `pip:` subsection of `environment.yaml`: it is not on conda-forge under that name (the conda package is `ortools-python`).

## Tests

```bash
pytest                                  # full suite (testpaths = tests/)
pytest tests/test_mac_1_cycle.py        # single file
pytest tests/test_mac_1_cycle.py::test_area   # single test
```

Most test files (e.g. `tests/test_mac_1_cycle.py`) execute the full pipeline at **module import time** against the `examples/mac_1_cycle/` inputs, then assert in individual `test_*` functions. A `test_cleanup()` at the end `shutil.rmtree`s the run output dir, so the output directory only exists mid-run. There are no Rust unit tests; the Rust engine is exercised through the Python suite.

`tests/test_numerical_equivalence.py` is the regression pin for the Rust aggregation engine: it builds a `mac_1_cycle`-shaped A-Graph directly through the `archx._core` API (no hardware interface) and checks every aggregation mode against hand-derived values, plus the ported design-error guards. Run it anywhere the extension builds; it does not need CACTI.

**Platform caveat (CACTI7):** the `cacti7` interface ships a prebuilt `cacti` binary that is an **x86-64 Linux ELF**. On other platforms (e.g. arm64 macOS) it cannot execute, and the interface only recompiles when the binary is *absent* - and its makefile is GCC-specific (`-gstabs+`), so it will not build under clang without patching. Any `tests/test_mac_*` run touches `sram` via `cacti7` and will fail at that step on such platforms. This is an interface/environment limitation, independent of the Rust core. On those platforms the CACTI-free tests still run: `test_numerical_equivalence.py` (the engine), the parser-stage unit tests `test_architecture.py` / `test_event.py` / `test_workload.py`, and `test_interface.py` (which exercises `csv_cmos`, not `cacti7`).

## CI & publishing (`.github/workflows/`)

- `ci.yaml` - on push/PR to `main`, builds via `maturin develop` and runs `pytest` across a matrix of `{ubuntu, macos, windows} x Python {3.9-3.12}`. Note this runs the full suite, so it exercises `cacti7` on the x86-64 Linux runners (where the bundled binary is native) but the macOS runners hit the platform caveat above.
- `publish.yaml` - on a published GitHub **release**, builds platform-specific wheels with `PyO3/maturin-action` for Ubuntu (x86_64 + aarch64), macOS (x86_64 + aarch64), and Windows (x86_64), plus an sdist, then uploads to PyPI via **trusted publishing** (OIDC, no stored token). Because the crate is built `abi3` (`pyo3/abi3-py39` in `pyproject.toml`), each platform yields a single `cp39-abi3` wheel covering Python 3.9+. The release version comes from `pyproject.toml`'s `version`. A smoke-test step (`import archx; import archx._core`) gates the natively-runnable targets before publish; the two cross-compiled wheels (linux/aarch64, macos/x86_64) build-and-upload without an import test since they can't execute on the runner.

## Running the CLI

The package installs an `archx` console command (`archx.main:main`). Two modes:

**Single run** - needs all five inputs plus a `.json` checkpoint path:
```bash
archx -r <run_dir> -a arch.yaml -m metric.yaml -w workload.yaml -e event.yaml -c out.json [-t] [-s]
```
`-t` mirrors logs to the terminal (otherwise only a logfile in the run dir); `-s` also dumps the parsed dicts as YAML.

**Frontend / sweeping** - drive design-space exploration from a Python description file:
```bash
archx -r <run_dir> -compile description.py [-full] [-ff]   # generate configurations.csv (+ runs.txt, +GUI)
archx -r <run_dir> -extract configurations.csv             # csv -> runs.txt
archx -r <run_dir> -f configurations.csv                   # Tkinter GUI to filter -> runs.txt
archx -r <run_dir> -x runs.txt                             # batch-execute runs via bin/run_archx.sh
```
`-full` chains compile→extract→execute; `bin/run_archx.sh` fans the runs out across cores (`nproc`) and logs failures to `failed_runs.txt`.

**Interface management:**
```bash
archx -ireg  -iname <name> -idir <dir>   # register (copy a dir into archx/interface/<name>)
archx -iureg -iname <name>               # unregister
archx -icopy -iname <name> -idir <dir>   # copy an existing interface out
```

## Pipeline architecture (the 7 steps)

`main.py` (single-run) and the test files both follow the same sequence. Each stage lives in its own subpackage under `src/archx/`:

1. **architecture** (`architecture/architecture.py`) → `create_architecture_dict`. Parses the architecture YAML into a *flattened* dict keyed by unique module name. Modules can `path:` to other YAML files (recursive include). Propagates `attribute` (global defaults like technology/frequency/interface), `tag` lists, and `instance` lists down the hierarchy. Each leaf module carries a `query` dict.
2. **metric** (`metric/metric.py`) → `create_metric_dict`. Each metric declares a `unit` and an `aggregation` mode: **`module`** (sum once over distinct modules - for area/leakage), **`summation`** (sum scaled by per-edge call counts - for energy), or **`specified`** (taken from a performance-model output, not from modules - for cycle_count/runtime). Default is `summation`.
3. **workload** (`workload/workload.py`) → `create_workload_dict`. Per-workload `configuration` knobs (e.g. GEMM m/k/n), also supports `path:` includes.
4. **event** (`event/event.py`) → `create_event_graph`. Builds the `ArchxGraph`: event nodes (with an attached `performance:` Python file), then leaf module nodes, then `event → subevent` edges.
5. **metric population** (`create_event_metrics` in metric.py) → for each leaf module, calls `query_interface(...)` and writes results into the graph via `ArchxGraph.set_module_data`.
6. **performance** (`performance/performance.py`) → `simulate_performance_all_events`. Runs each event's performance-model function, setting per-edge `count`/`operation`/`factor` and per-node `specified` metrics on the graph.
7. **aggregation / query** - `aggregate_event_metric`, `aggregate_tag_metric`, `query_module_metric` walk the graph to produce final `{value, unit}` results.

### The Rust core (`src/archx/rust-core/src/`)

- `lib.rs` - the `ArchxGraph` PyO3 class (the entire Python-facing API: node/edge construction, metric setters/getters, the three aggregation entry points, and `save_json`/`load_json` checkpointing). **This is the Python↔Rust contract; changing a method signature here means updating its Python callers in `event/`, `metric/`, `performance/`.**
- `graph.rs` - `ArchxGraphInner`, a petgraph `DiGraph` with a name→index map; node payloads hold metrics, instance count, tags; edge payloads hold count/aggregation/operation/factor.
- `metric.rs` - `MetricValue` is either `Single{value,unit}` or `MultiOp{op → SingleMetric}` (multi-op metrics, e.g. SRAM read vs write energy, keyed by operation name); plus per-node `specified_metrics`.
- `aggregate.rs` - the aggregation algorithms (`module`, `summation`, `summation_multiop`, `specified`) and `compute_path_count` / `aggregate_event_count` which scale by edge counts along workload→event paths.
- `paths.rs` - path enumeration over the graph.

Checkpoints are plain JSON (`save_json`/`load_json`), the `.json` file passed via `-c`.

**Logging bridge (Rust → loguru):** the Rust core logs through the `log` crate; `pyo3_log::try_init()` (in the `_core` `#[pymodule]`) forwards those records to Python's stdlib `logging`, and `archx/__init__.py` installs an `InterceptHandler` that redirects stdlib logging into **loguru**. So Rust and Python share one log stream, and level filtering is done by loguru's sinks (set by the CLI `--log_level` in `main.py`), identical to the original pure-Python tracing. When adding Rust trace logs, use `log::debug!/info!/warn!` and keep the message wording matching the Python original. (Minor cosmetic note: Rust's `{}` prints integer-valued floats without a trailing `.0`, unlike Python's f-strings.)

### Interfaces (`src/archx/interface/`)

Pluggable cost models. `interface.py::query_interface` dynamically imports `interface/<name>/<name>.py` and calls its `query(module, interface, query, input_dir, output_dir)`. A module's `query` dict selects its interface via the `interface:` key (often inherited from architecture `attribute`). Existing interfaces:
- `cacti7/` - wraps the CACTI7 C++ memory model (SRAM/DRAM); has bundled C++ source under `include/cacti7/` and `.cfg` templates.
- `csv_cmos/` - interpolates area/power/energy from CMOS synthesis & place-and-route CSVs (`syn_pnr_csv/`); `csv_cmos_extract.py` generates those CSVs from synthesis reports (see its README).
- `csv_sc/` - stochastic-computing module CSVs.

### Frontend programming (`src/archx/programming/`)

`graph/agraph.py::AGraph` is the sweeping/DSE front-end. It uses **Google OR-Tools CP-SAT** (`cp_model`) to enumerate valid parameter combinations under constraints (`direct_constraint`, `conditional_constraint`, …). A user `description.py` defines a `description(path)` function that builds an `AGraph` and emits `configurations.csv`; `_generate_runs` turns that into `runs.txt`, and `_gui` is a Tkinter filter. `programming/object/` holds the `Architecture`/`Event`/`Metric`/`Workload`/`ParameterEnumerator` builder objects.

## Examples & zoo

- `examples/mac_1_cycle/` - the canonical minimal example (a MAC array + SRAM running GEMM); the tests run against it. `examples/systolic_array/` has a `description.py` sweep example.
- `zoo/llm/` - larger real configurations (CARAT, systolic, tensor, SIMD, MUGI accelerators for LLM workloads) with generation scripts under `zoo/llm/scripts/`.

## Conventions

- YAML inputs are read with `yamlordereddictloader` to preserve order; dicts flow as `OrderedDict` throughout.
- Logging is via **loguru**; pipeline stages emit `logger.success` banners. Assertions use `logger.error(...)` as the message.
- Path resolution goes through `archx.utils.get_path` (resolves relative to repo/cwd); prefer it over raw paths in pipeline code.
- The git branch `rust` is the active line of work that ported the graph engine from `graph-tool` to the Rust `_core` extension with no changes to the Python-facing pipeline API.
