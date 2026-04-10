# Archx
A cost modeling framework to explore the system design space based on A-Graph.

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
Editable install from source — live code changes are reflected without reinstalling.

Requires the [Rust toolchain](https://rustup.rs/) to compile the Rust extension via [Maturin](https://www.maturin.rs/):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version  # verify
```

```bash
git clone https://github.com/UnaryLab/archx.git && cd archx
conda env create -f environment.yaml   # edit `name: archx` to rename
conda activate archx
pip install -e . --no-deps             # editable install; Rust extension is compiled here
```

### Validate
```bash
archx -h
python -c "import archx"
```

### Running tests
```bash
pytest
```
