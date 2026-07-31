# Archx

An event-based cost-modeling framework for computer-system design-space exploration, built around the **A-Graph** abstraction.

Archx models across the system stack, separating into 4 levels. Each level is described by one of the four inputs to a run (detailed under [Describing a design](#describing-a-design)).

## Application

A given workload (e.g. LLM, Signal processing, Error correction, etc). Each workload is defined with parameters to sweep through multiple configurations.

## Software

An event graph that details how the application decomposes into architecture events. Each event has an isolated, python-based **performance model** attached that translates the workload configuration into per-subevent call counts.

## Architecture

The micro-architecture or blocks that build the overall architecture. Such blocks can be implemented at any granularity.

## Circuit

The physical costs of each module, or **metrics**. Each metric, (e.g., area, power, energy, cycle count, runtime, or any user-defined quantity), details its aggregation up the graph; pluggable hardware **interfaces** (the CACTI7 memory model, CMOS synthesis CSVs, ...) supply the circuit-level values per module.

Archx computes arbitrary hardware metrics for an architecture running a workload. It does this by:

1. building a directed graph (the A-Graph) of **events** and hardware **modules** connected by **subevent** edges,
2. populating each hardware module with costs queried from a pluggable hardware **interface** (the CACTI7 memory model, CMOS synthesis CSVs, ...),
3. running user-supplied Python **performance models** that set per-edge call counts, and
4. aggregating metrics up the graph to answer queries such as "total energy of a GEMM on this accelerator".

The graph engine is implemented in Rust and exposed to Python as `archx._core`; everything else is Python.

## Installation
All installation methods provide the `archx` CLI command and the `import archx` Python module.

### Prerequisites
- [Anaconda](https://www.anaconda.com/) for managing the environment

### Option 1: conda + pip install from PyPI (recommended)
Installs all dependencies via conda and the Archx package from PyPI.

```bash
conda env create -f environment.yaml   # edit `name: archx` to rename
conda activate archx
pip install archx
```

### Option 2: source installation (developer mode)
Editable install from source: live code changes are reflected without reinstalling.

Requires the [Rust toolchain](https://rustup.rs/) to compile the Rust extension via [Maturin](https://www.maturin.rs/):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"   # add cargo/rustc to PATH in the current shell
rustc --version             # verify
```

```bash
git clone https://github.com/UnaryLab/archx.git && cd archx
conda env create -f environment.yaml   # edit `name: archx` to rename
conda activate archx
pip install -e . --no-deps             # editable install; Rust extension is compiled here
```

After editing any Rust source under `src/archx/rust-core/`, rerun `pip install -e . --no-deps` to recompile; Python changes are live without reinstalling.

### Validate
```bash
archx -h
python -c "import archx"
```

## Describing a design

A run is described by four YAML files plus one or more Python performance models.

### Architecture (`-a`)
The hardware: a flattened set of named modules. `attribute` holds global defaults (technology, frequency, default interface) that are merged into each module's `query`; the `query` dict is what gets sent to the hardware interface to price the module. Modules can `path:` to other architecture YAML files for hierarchical descriptions, carry `tag:` lists for group queries, and `instance:` lists for arrays of identical units.

```yaml
architecture:
  attribute:
    technology: 45        # nm
    frequency: 400        # MHz
    interface: csv_cmos   # default hardware interface
  module:
    mac_array:
      path: mac_array.architecture.yaml   # include another file
    sram:
      tag: [memory, onchip]
      query:
        interface: cacti7
        class: sram
        bank: 32
        width: 64          # bits
        depth: 1024
```

### Metric (`-m`)
The metrics to compute. Each metric declares a `unit` and an `aggregation` mode:

| mode | meaning | typical metrics |
|---|---|---|
| `module` | sum once over distinct modules | area, leakage power |
| `summation` | sum scaled by per-edge call counts (default) | dynamic energy |
| `specified` | taken directly from a performance-model output | cycle count, runtime |

```yaml
metric:
  area:
    unit: mm^2
    aggregation: module
  dynamic_energy:
    unit: nJ                # aggregation defaults to summation
  runtime:
    unit: ms
    aggregation: specified
```

### Workload (`-w`)
One workload per file: a `name` and a `configuration` dict of knobs the performance models read. A file may instead `path:` to another workload file.

```yaml
workload:
  name: llama_3_70b
  configuration:
    batch_size: 32
    dim: 8192
    layers: 80
```

### Event (`-e`)
The A-Graph structure: each event lists its `subevent`s (other events, or leaf hardware modules from the architecture) and the Python file holding its performance model.

```yaml
event:
  gemm:
    subevent: [mac_array, sram_rd, sram_wr]
    performance: performance/example.performance.py
  sram_rd:
    subevent: [sram]
    performance: performance/example.performance.py
```

### Performance models
For each event, a Python function with the same name as the event:

```python
def gemm(architecture_dict, workload_dict=None):
    cfg = workload_dict['configuration']
    macs = cfg['m'] * cfg['k'] * cfg['n']
    return OrderedDict({
        'subevent': OrderedDict({
            'mac_array': OrderedDict({'count': macs / array_size}),
            'sram_rd':   OrderedDict({'count': reads}),
        }),
    })
```

It receives the parsed architecture and workload dicts and returns, per subevent, the call `count` (and optionally an `operation` such as `read`/`write` for multi-operation modules like SRAM). It may also return `specified` metrics directly (e.g. `{'cycle_count': {'value': 1., 'unit': 'cycles'}}`). See `examples/mac_1_cycle/input/performance/example.performance.py` for a complete model.

## Artifact Evaluation (HPCA 2027)

This branch (`agraph_hpca_2027`) is the artifact for our HPCA 2027 submission. It reproduces every case-study result from source inside a Docker container: nothing is pre-generated. The image builds Archx from source, registers the hardware interfaces, runs all case-study designs under `agraph/designs/`, and regenerates the figures and validation tables.

### Requirements
- Docker. The artifact is CPU-only; no GPU or CUDA is required.
- An x86-64 Linux host is recommended (the CACTI7 interface compiles from C++ source inside the image).

### Build
```bash
docker build -t archx-agraph .
```

This installs a pinned Rust toolchain, compiles the Rust A-Graph core via Maturin, installs the Python dependencies (framework deps from `pyproject.toml`, plus `agraph/requirements.txt` for the figure/table scripts), and copies the case studies in. The case-study runs happen at container run time, not during the build.

### Run
```bash
mkdir -p out/figures out/tables
docker run --rm \
    -v "$PWD/out/figures:/opt/archx/agraph/res/figures" \
    -v "$PWD/out/tables:/opt/archx/agraph/res/tables" \
    archx-agraph
```

The container executes `agraph/agraph.sh`, which:

1. registers the hardware-characterization interfaces (`agraph/interface/`) into Archx,
2. compiles and runs every design under `agraph/designs/`, writing per-run results, and
3. regenerates the figures and validation tables.

Only the two output directories are mounted, so the plotting scripts baked into the image stay intact. Generated figures land in `out/figures/`, validation tables in `out/tables/`.

The script runs under `set -e` and must complete with exit code `0`. Any non-zero exit is a genuine reproduction failure, not an expected warning.

### Reproduced outputs

| Case study | Output |
| --- | --- |
| FFT (coarse / fine grain) | `out/figures/fft_metrics_comparison.pdf` |
| TNN (coarse / fine grain) | `out/figures/tnn_metrics.pdf` |
| Systolic array (coarse / fine grain) | `out/figures/systolic_metrics_comparison.pdf` |
| Stochastic-computing FIR | `out/figures/fir_metrics_comparison.pdf` |
| Modeling-runtime comparison | `out/figures/runtime_comparison.pdf` |
| RISC-V GEMM validation | `out/tables/riscv_gemm.txt` |
| GPU GPT-2 validation | `out/tables/gpu_gpt2.txt` |

Each figure is also written as a `.png` alongside the `.pdf`. The entry point for the whole flow is `agraph/agraph.sh`; see the `agraph/` tree for the design descriptions, performance models, interface bundles, and plotting scripts.

## Hardware interfaces

Interfaces are the pluggable cost models that price each module (`src/archx/interface/`):

- `cacti7`: SRAM/DRAM area, power, and energy via the CACTI7 memory model (bundled C++ source; the prebuilt binary is x86-64 Linux).
- `csv_cmos`: interpolates logic-module costs from CMOS synthesis and place-and-route CSVs.
- `chiplet_cmos`: CMOS CSV costs for chiplet-based designs.
- `csv_sc`: stochastic-computing module CSVs.

See [`src/archx/interface/README.md`](src/archx/interface/README.md) for the query contract and how to add your own.

## Repository layout

- `src/archx/`: the Python package (pipeline stages `architecture/`, `metric/`, `workload/`, `event/`, `performance/`; `interface/` cost models; `programming/` sweep frontend).
- `src/archx/rust-core/`: the Rust A-Graph engine, compiled into `archx._core`.
- `examples/`: small end-to-end examples ([`examples/README.md`](examples/README.md)).
- `zoo/llm/`: larger real accelerator configurations (CARAT, systolic, tensor, SIMD, MUGI) for LLM workloads.
- `chiplet4ai/`: chiplet-based accelerator study for Llama models, built on Archx.
- `tests/`: pytest suite.

## Running tests
```bash
pytest
```

Note: tests touching SRAM run CACTI7, whose bundled binary is an x86-64 Linux executable; on other platforms those tests cannot run. `tests/test_numerical_equivalence.py` pins the Rust aggregation engine against hand-derived values and runs anywhere the extension builds.
