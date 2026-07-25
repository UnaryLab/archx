# Hardware interfaces

An interface is a pluggable cost model that prices a hardware module: given a module's `query` dict from the architecture YAML, it returns the module's metrics (area, leakage power, per-operation dynamic energy, ...).

A module selects its interface with the `interface:` key inside its `query` (usually inherited from the architecture's global `attribute`). At metric-population time, Archx imports `interface/<name>/<name>.py` and calls its `query` function.

## Bundled interfaces

- `cacti7/`: wraps the CACTI7 C++ memory model for SRAM/DRAM. Bundles the CACTI7 source under `include/cacti7/` plus `.cfg` templates. Binaries are stored per host as `cacti-<system>-<machine>` (`cacti-Linux-x86_64`, `cacti-Darwin-arm64`); the one matching the host is used, and built from source when absent.
- `csv_cmos/`: interpolates area/power/energy of logic modules from CMOS synthesis and place-and-route CSVs under `include/csv/`, characterized at 45 nm and 7 nm. `csv_cmos_extract.py` generates those CSVs from synthesis reports; see `csv_cmos/README.md`.
- `csv_cmos_32nm/`: the same lookup over a 32 nm library under `include/csv/`.
- `csv_cmos_asplos_2026_ae/`: the same lookup over the 45 nm library the `asplos_2026_ae` branch characterized, used by `zoo/llm`.
- `chiplet_cmos/`: CMOS CSV lookup for chiplet-based designs (CSVs under `csv/`).
- `csv_sc/`: stochastic-computing module CSVs.
- `csv_h200/`: measured H200 power and runtime per GPU kernel.
- `csv_riscv/`: per-stage CSVs for a RISC-V core, reporting dynamic energy per instruction where a class lists several.

## Writing an interface

An interface is a directory `<name>/` containing `<name>.py` that defines:

```python
def query(name: str, interface: str, query: OrderedDict, input_dir=None, output_dir=None):
    ...
    return metric_dict
```

- `name`: the module name from the architecture.
- `query`: the module's query dict (interface key already stripped; global attributes such as `technology` and `frequency` merged in).
- Returns an OrderedDict of metrics. Each metric is either a single `{'value': ..., 'unit': ...}` entry, or a per-operation dict (e.g. separate `read`/`write` dynamic energy for an SRAM); the performance model then selects the operation via the `operation` key on a subevent edge.

Query results are cached in memory and on disk (under `$ARCHX_INTERFACE_CACHE_DIR`, defaulting to `~/.cache/archx/interface`), so repeated queries with identical parameters do not rerun the underlying model.

## Managing interfaces

```bash
archx -ireg  -iname <name> -idir <dir>   # register: copy <dir> into archx/interface/<name>
archx -iureg -iname <name>               # unregister: remove archx/interface/<name>
archx -icopy -iname <name> -idir <dir>   # copy an installed interface out to <dir>
```

Registration copies the directory into the installed package, so the interface travels with the `archx` installation rather than the calling project.
