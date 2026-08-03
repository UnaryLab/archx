# Archx: A-Graph Case Studies (HPCA 2027 Artifact)

This branch (`agraph_hpca_2027`) is the artifact for our HPCA 2027 submission. It
reproduces every case-study result from source inside a Docker container.
The image builds Archx (an event-based cost-modeling framework
built around the **A-Graph** abstraction) from source, registers the hardware
interfaces, runs all case-study designs under `agraph/designs/`, and regenerates
the figures and validation tables.

## Requirements

- Docker. The artifact is CPU-only; no GPU or CUDA is required.
- An x86-64 Linux host is recommended (the CACTI7 interface compiles from C++
  source inside the image).

## Install
```bash
git clone https://github.com/UnaryLab/archx.git
cd archx
git switch -c agraph_hpca_2027 --track origin/agraph_hpca_2027
```

## Build

```bash
docker build -t archx-agraph .
```

This installs a pinned Rust toolchain, compiles the Rust A-Graph core via Maturin,
installs the Python dependencies (framework deps from `pyproject.toml`, plus
`agraph/requirements.txt` for the figure/table scripts), and copies the case
studies in. The case-study runs happen at container run time, not during the build.

## Run

```bash
docker run --rm \
    -v "$PWD/out/figures:/opt/archx/agraph/res/figures" \
    -v "$PWD/out/tables:/opt/archx/agraph/res/tables" \
    archx-agraph
```

The `-v` flag auto-creates the `out/figures` and `out/tables` host directories if
they don't exist, so no `mkdir` is needed beforehand.

The container executes `agraph/agraph.sh`, which:

1. registers the hardware-characterization interfaces (`agraph/interface/`) into Archx,
2. compiles and runs every design under `agraph/designs/`, writing per-run results, and
3. regenerates the figures and validation tables.

Only the two output directories are mounted, so the plotting scripts baked into
the image stay intact. Generated figures land in `out/figures/`, validation tables
in `out/tables/`.

The script runs under `set -e` and must complete with exit code `0`. Any non-zero
exit is a genuine reproduction failure, not an expected warning.

## Reproduced outputs

| Case study | Output |
| --- | --- |
| FFT (coarse / fine grain) | `out/figures/fft_metrics_comparison.pdf` |
| TNN (coarse / fine grain) | `out/figures/tnn_metrics.pdf` |
| Systolic array (coarse / fine grain) | `out/figures/systolic_metrics_comparison.pdf` |
| Stochastic-computing FIR | `out/figures/fir_metrics_comparison.pdf` |
| Modeling-runtime comparison | `out/figures/runtime_comparison.pdf` |
| RISC-V GEMM validation | `out/tables/riscv_gemm.txt` |
| GPU GPT-2 validation | `out/tables/gpu_gpt2.txt` |

Each figure is written as a `.pdf`. The entry point for
the whole flow is `agraph/agraph.sh`; see the `agraph/` tree for the design
descriptions, performance models, interface bundles, and plotting scripts.
