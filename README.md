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
git switch -c hpca_2027_ae --track origin/hpca_2027_ae
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
    archx-agraph
```

The docker container will generate each case study's result at runtime.

The container executes `agraph/agraph.sh`, which:

## Reproduced outputs

All casestudy figures and tables are reproduced in this artifact. Tables have two outputs, one matching the LaTex code to compile the table, and a csv output for readability.

<table>
  <tr>
    <th>Case study</th>
    <th>Target</th>
    <th>Paper Figure / Table</th>
    <th>Output</th>
  </tr>

  <tr>
    <th colspan="4">Figures</th>
  </tr>
  <tr>
    <td>Runtime</td>
    <td>CMOS</td>
    <td>Figure 8</td>
    <td><code>out/figures/runtime_comparison.pdf</code></td>
  </tr>
  <tr>
    <td>FIR Bitwidth</td>
    <td>Superconducting</td>
    <td>Figure 9</td>
    <td><code>out/figures/fir_validation.pdf</code></td>
  </tr>
  <tr>
    <td>TNN</td>
    <td>Neuromorphic</td>
    <td>Figure 10</td>
    <td><code>out/figures/tnn_validation.pdf</code></td>
  </tr>
  <tr>
    <td>FFT</td>
    <td>CMOS</td>
    <td>Figure 11</td>
    <td><code>out/figures/fft_validation.pdf</code></td>
  </tr>
  <tr>
    <td>Systolic array</td>
    <td>CMOS</td>
    <td>Figure 12</td>
    <td><code>out/figures/systolic_validation.pdf</code></td>
  </tr>

  <tr>
  <th colspan="4">Tables</th>
</tr>
<tr>
  <td>FIR Power</td>
  <td>Superconducting</td>
  <td>Table 3</td>
  <td>
    <code>out/tables/txt/fir_validation.txt</code><br>
    <code>out/tables/csv/fir_validation.csv</code>
  </td>
</tr>
<tr>
  <td>CNN</td>
  <td>Superconducting</td>
  <td>Table 4</td>
  <td>
    <code>out/tables/txt/cnn_validation.txt</code><br>
    <code>out/tables/csv/cnn_validation.csv</code>
  </td>
</tr>
<tr>
  <td>CNN breakdown</td>
  <td>Superconducting</td>
  <td>Table 5</td>
  <td>
    <code>out/tables/txt/cnn_breakdown.txt</code><br>
    <code>out/tables/csv/cnn_breakdown.csv</code>
  </td>
</tr>
<tr>
  <td>Systolic array breakdown</td>
  <td>CMOS</td>
  <td>Table 6</td>
  <td>
    <code>out/tables/txt/systolic_breakdown.txt</code><br>
    <code>out/tables/csv/systolic_breakdown.csv</code>
  </td>
</tr>
<tr>
  <td>RISC-V</td>
  <td>CMOS</td>
  <td>Table 7</td>
  <td>
    <code>out/tables/txt/riscv_validation.txt</code><br>
    <code>out/tables/csv/riscv_validation.csv</code>
  </td>
</tr>
<tr>
  <td>GPU GPT-2</td>
  <td>CMOS</td>
  <td>Table 8</td>
  <td>
    <code>out/tables/txt/gpu_gpt2_validation.txt</code><br>
    <code>out/tables/csv/gpu_gpt2_validation.csv</code>
  </td>
</tr>
</table>
