# Xeon Tulsa on gem5

An archx description of the dual-core Intel Xeon Tulsa from McPAT's `Xeon.xml`, run on gem5 as two O3 x86 cores with a Ruby MESI three-level hierarchy and a 64x64x64 double GEMM in SE mode.

- `xeon_tulsa.architecture.yaml`: modules `core`, `l1i`, `l1d`, `l2`, `l3`, `mc`. Each `query` is the hardware description; each `gem5` block holds settings only gem5 reads.
- `xeon_tulsa.workload.yaml`: guest binary and its arguments (`m k n`).
- `gemm.c`, `build.sh`: the guest kernel. The multiply runs between `m5_work_begin` and `m5_work_end`.
- `run_gem5.py`: runs the design through the `gem5` interface (`archx.interface.query_interface`).

## Run

Requires a gem5 build at `$GEM5_PATH/build/ALL/gem5.opt` and `libm5.a` (`cd $GEM5_PATH/util/m5 && scons build/x86/out/m5`). From the archx repo root:

```bash
export GEM5_PATH=/path/to/gem5
zoo/agraph/designs/riscv_mcpat/build.sh
python zoo/agraph/designs/riscv_mcpat/run_gem5.py
```

The script prints the stats file in the run directory `$ARCHX_GEM5_CACHE_DIR/<hash>/` (default `~/.cache/archx/gem5/<hash>/`). The run directory holds the generated `config.py`, gem5 output in `m5out/`, and `gem5.stdout`/`gem5.stderr`. A repeated run with the same description and binaries reuses the run without calling gem5.

`m5out/stats.txt` holds two dumps: the first is the region of interest (the multiply); the second runs from the start of the region to program exit.
