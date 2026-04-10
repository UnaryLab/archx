# Archx
A cost modeling framework to explore the system design space based on A-Graph.

## Installation
All installation methods provide the `archx` CLI command and the `import archx` Python module.

### Prerequisites
- Python >= 3.9
- [Rust toolchain](https://rustup.rs/) — required to compile the Rust extension via [Maturin](https://www.maturin.rs/)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustc --version  # verify
  ```
- (Recommended) [Anaconda](https://www.anaconda.com/) for managing the full environment

### Option 1: conda + pip install from PyPI (recommended)
Installs core dependencies via conda and the Archx package from PyPI.

```bash
conda env create -f environment.yaml   # edit `name: archx` to rename
conda activate archx
pip install archx
```

### Option 2: source installation (developer mode)
Editable install from source — live code changes are reflected without reinstalling.

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
