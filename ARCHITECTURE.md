# Repository layout

- `src/archx/`: the Python package. Pipeline stages live in `architecture/`, `metric/`, `workload/`, `event/`, and `performance/`; `interface/` holds the cost models; `programming/` is the sweep frontend.
- `src/archx/rust-core/`: the Rust A-Graph engine, compiled into `archx._core`.
- `src/archx/bin/run_archx.sh`: the batch runner, one `archx` invocation per line of an argument file, fanned out across CPU cores.
- `examples/`: small end-to-end examples ([`examples/README.md`](examples/README.md)).
- `zoo/agraph/`: A-Graph designs (FFT, systolic, TNN, stochastic computing, MUGI, RISC-V, GPU kernels).
- `zoo/chiplet4ai/`: a chiplet-based accelerator study for Llama models.
- `zoo/llm/`: five accelerators (CARAT, MUGI, SIMD, systolic, tensor) swept over LLM workloads, with the query and figure scripts under `zoo/llm/results/`.
- `tests/`: the pytest suite.

## The A-Graph pipeline

A run moves through seven stages, each reading the output of the last:

1. `architecture/` parses the architecture YAML into a flat module dict, resolving `path:` includes and merging global `attribute` defaults into each module's `query`.
2. `metric/` parses the metric YAML and records each metric's unit and aggregation mode.
3. `workload/` parses the workload YAML, following `path:` chains to the file that carries the `name` and `configuration`.
4. `event/` builds the A-Graph in Rust from the event YAML: events and hardware modules as nodes, subevent relations as edges.
5. `metric/` populates each module node by querying its hardware interface.
6. `performance/` runs the Python performance model attached to each event, setting per-edge call counts.
7. `metric/` aggregates up the graph to answer a query.

## Interfaces

An interface is a directory `src/archx/interface/<name>/` containing `<name>.py` with a `query(name, interface, query, input_dir, output_dir)` function. `interface.py` imports it by path and caches results in memory and on disk, keyed on the query and a fingerprint of every file in the interface directory. See [`src/archx/interface/README.md`](src/archx/interface/README.md) for the contract.

## Caching

Two independent caches persist under `~/.cache/archx/`, and both are keyed so that changing an input invalidates the entry:

- `interface/`: query results, keyed on the query, the interface directory fingerprint, and a version integer.
- `performance/`: model outputs and their traced dependencies, keyed on the model file plus every project module it imports, transitively.

Set `ARCHX_INTERFACE_CACHE_DIR` or `ARCHX_PERFORMANCE_CACHE_DIR` to relocate them, or `ARCHX_DISABLE_PERFORMANCE_CACHE=1` to bypass the second.
