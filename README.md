# Archx

A cost-modeling framework for computer-system design-space exploration, built around the **A-Graph** abstraction.

Archx models across the system stack, separating into 4 levels. Each level is described by one of the four inputs to a run (detailed under [Describing a design](#describing-a-design)).

# Application

What the system executes: the **workload**. A workload YAML (`-w`) names the application (a GEMM, an LLM, ...) and gives its `configuration` knobs (matrix dimensions, batch size, layer counts, ...) that the levels below consume.

# Software

How the application decomposes into work on the hardware: the **events**. An event YAML (`-e`) defines the A-Graph structure, breaking the application into a hierarchy of events down to leaf hardware modules, and attaches to each event a Python **performance model** that translates the workload configuration into per-subevent call counts, operations, and timing.

# Architecture

The hardware organization the software runs on: the **architecture**. An architecture YAML (`-a`) declares the modules (compute units, memories, interconnect), their hierarchy and instance counts, their tags for group queries, and the query parameters (technology, frequency, sizes) used to price each module.

# Circuit

The physical costs of each module: arbitrary **metrics**. A metric YAML (`-m`) declares which metrics to compute (area, power, energy, cycle count, runtime, or any user-defined quantity) and how each aggregates up the graph; pluggable hardware **interfaces** (the CACTI7 memory model, CMOS synthesis CSVs, ...) supply the circuit-level values per module.

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

## Running

### Single run
```bash
archx -r <run_dir> \
      -a arch.yaml -m metric.yaml -w workload.yaml -e event.yaml \
      -c <run_dir>/graph.json \
      [-t] [-s] [-d] [-l DEBUG] [-p <dir>]
```

- `-c` is the checkpoint the populated A-Graph is saved to (must end in `.json`).
- `-t` mirrors the log to the terminal (a logfile is always written into the run dir).
- `-s` also dumps the parsed architecture/metric/workload/event dicts as YAML into the run dir.
- `-d` deletes the run dir first if it exists.
- `-p` adds a directory to `sys.path` so performance models can import local helpers.

### Querying results
Load the checkpoint and aggregate any metric at any event:

```python
from archx.event import load_event_graph
from archx.metric import create_metric_dict, aggregate_event_metric

event_graph = load_event_graph('<run_dir>/graph.json')
metric_dict = create_metric_dict('metric.yaml')

result = aggregate_event_metric(event_graph=event_graph, metric_dict=metric_dict,
                                metric='dynamic_energy', workload='gemm', event='gemm')
# -> OrderedDict({'value': ..., 'unit': 'nJ'})
```

`aggregate_tag_metric` aggregates over all modules sharing an architecture `tag`, and `query_module_metric` reads a single module's raw metrics.

### Design-space sweeps
A Python *description file* defining a `description(path)` function (built on `archx.programming`, which uses OR-Tools CP-SAT to enumerate valid configurations under constraints) drives batch exploration:

```bash
archx -r <run_dir> -compile description.py        # generate configurations.csv
archx -r <run_dir> -extract configurations.csv    # csv -> runs.txt
archx -r <run_dir> -f configurations.csv          # Tkinter GUI to filter -> runs.txt
archx -r <run_dir> -x runs.txt                    # execute all runs in parallel
```

`-compile ... -full` chains all of the above in one command (add `-ff` to insert the GUI filter step). `-x` fans the runs out across all CPU cores; failing runs are collected in `failed_runs.txt`. See `examples/systolic_array/description.py` for a description file.

### Interface management
```bash
archx -ireg  -iname <name> -idir <dir>   # register a new hardware interface
archx -iureg -iname <name>               # unregister
archx -icopy -iname <name> -idir <dir>   # copy an installed interface out
```

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
