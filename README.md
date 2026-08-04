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
studies in.

## Run

```bash
docker run --rm \
    -v "$PWD/out/figures:/opt/archx/agraph/res/figures" \
    -v "$PWD/out/tables:/opt/archx/agraph/res/tables" \
    -v "$PWD/out/csv:/opt/archx/agraph/res/csv" \
    archx-agraph
```

The docker container will generate each case study's result at runtime.

The container executes `agraph/agraph.sh`, which:

## Reproduced outputs

| Case study | Target | Paper Figure / Table | Output |
| Figures |
| --- | --- | --- | --- |
| Runtime                  | CMOS            | Figure 8  | `out/figures/runtime_comparison.pdf`  |
| FIR Bitwidth             | Superconducting | Figure 9  | `out/figures/fir_validation.pdf`      |
| TNN                      | Neuromorphic    | Figure 10 | `out/figures/tnn_validation.pdf`      |
| FFT                      | CMOS            | Figure 11 | `out/figures/fft_validation.pdf`      |
| Systolic array           | CMOS            | Figure 12 | `out/figures/systolic_validation.pdf` |
| Tables |
| FIR Power                | Superconducting | Table 3   | `out/csv/fir_validation.txt`          |
| CNN                      | Superconducting | Table 4   | `out/csv/cnn_validation.txt`          |
| CNN breakdown            | Superconducting | Table 5   | `out/csv/cnn_breakdown.txt`           |
| Systolic array breakdown | CMOS            | Table 6   | `out/csv/systolic_breakdown.txt`      |
| RISC-V                   | CMOS            | Table 7   | `out/csv/riscv_validation.txt`        |
| GPU GPT-2                | CMOS            | Table 8   | `out/csv/gpu_gpt2_validation.txt`     |