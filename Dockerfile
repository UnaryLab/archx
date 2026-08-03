# Reproducible artifact image for the A-Graph case studies.
# CPU-only cost modeling; no CUDA or GPU required.
#
# Build:  docker build -t archx-agraph .
# Run:    docker run --rm \
#             -v "$PWD/out/figures:/opt/archx/agraph/res/figures" \
#             -v "$PWD/out/tables:/opt/archx/agraph/res/tables" \
#             -v "$PWD/out/csv:/opt/archx/agraph/res/csv" \
#             archx-agraph
#         (-v auto-creates the host output dirs if missing; mount only the
#          output subdirs so the baked-in res/scripts stay intact)
FROM python:3.11-slim-bookworm

# System deps:
#   build-essential -> g++/make; the cacti7 interface compiles CACTI at run time
#   curl, git       -> fetch and bootstrap the Rust toolchain
#   libtk8.6        -> Tk runtime (libtk8.6.so) so the tkinter / PIL.ImageTk
#                      imports at the top of archx/programming/graph/agraph.py
#                      resolve; pulls libtcl8.6 + X11 libs as dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

# archx is a maturin/PyO3 package built from source, so it needs a Rust
# toolchain. Pin it for reproducibility.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.82.0 --profile minimal

WORKDIR /opt/archx
COPY . .

# Build + install archx (compiles the Rust core; deps come from pyproject.toml),
# then the extra libraries the case-study figure/table scripts need.
RUN pip install --no-cache-dir "maturin>=1.5" \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir -r agraph/requirements.txt

# Reproduce the artifact: register interfaces, run every design, emit
# results, figures, and tables under agraph/.
CMD ["bash", "agraph/agraph.sh"]
