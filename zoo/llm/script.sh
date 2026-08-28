#!/bin/bash
# Build + run every LLM-zoo design. Each design is a single AGraph description.py
# (compiled internally into its config set + runs.txt); there is no more
# generate_dicts/generate_runs step.
set -e

echo "Setting up environment."
export PYTHONPATH="$(pwd)/zoo:$(pwd):$PYTHONPATH"
# mugi's shared common model does workload_config.update(...) under architecture=='mugi',
# which the performance cache's read-only tracer cannot handle. Disable the cache.
export ARCHX_DISABLE_PERFORMANCE_CACHE=1

RUNS="zoo/llm/runs.txt"
: > "$RUNS"

# Select designs to build: names passed as arguments, else every design.
# Usage: llm_script.sh [design ...]   e.g.  llm_script.sh mugi systolic
if [ "$#" -gt 0 ]; then
    designs=("$@")
else
    designs=()
    for dir in zoo/llm/designs/*/; do
        [ -f "${dir}description.py" ] || continue
        designs+=("$(basename "$dir")")
    done
fi

echo "Compiling designs (config set + runs.txt per design): ${designs[*]}"
for name in "${designs[@]}"; do
    dir="zoo/llm/designs/$(basename "$name")/"
    if [ ! -f "${dir}description.py" ]; then
        echo "  ! skipping '${name}': ${dir}description.py not found" >&2
        continue
    fi
    echo "  - ${dir}description.py"
    archx -compile "${dir}description.py" -r "${dir}description"
    archx -r "${dir}description" -extract "${dir}description/configurations.csv"
    cat "${dir}description/runs.txt" >> "$RUNS"
done

echo "Simulating all runs across cores."
# archx -x runs the batch in one process pool with a tqdm progress bar
# (failed runs are collected in failed_runs.txt); -r only holds the batch log
archx -x "$RUNS" -r zoo/llm/log

echo "Query performance models and generate figures."
python -m zoo.llm.results.figure_generation
