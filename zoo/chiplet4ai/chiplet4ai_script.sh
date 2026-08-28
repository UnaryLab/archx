#!/bin/bash
# Build + run every chiplet4ai design. Each design is a single AGraph
# description.py (compiled internally into its config set + runs.txt), the
# same flow as zoo/llm/llm_script.sh.
set -e

echo "Setting up environment."
export PYTHONPATH="$(pwd)/zoo:$(pwd):$PYTHONPATH"

RUNS="zoo/chiplet4ai/runs.txt"
: > "$RUNS"

# Select designs to build: names passed as arguments, else every design.
# Usage: chiplet4ai_script.sh [design ...]   e.g.  chiplet4ai_script.sh llama
if [ "$#" -gt 0 ]; then
    designs=("$@")
else
    designs=()
    for dir in zoo/chiplet4ai/designs/*/; do
        [ -f "${dir}description.py" ] || continue
        designs+=("$(basename "$dir")")
    done
fi

echo "Compiling designs (config set + runs.txt per design): ${designs[*]}"
for name in "${designs[@]}"; do
    dir="zoo/chiplet4ai/designs/$(basename "$name")/"
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
# archx -x runs the batch in one process pool with a tqdm progress bar; -r only
# holds the batch log. A partially failed batch must not stop figure generation.
status=0
archx -x "$RUNS" -r zoo/chiplet4ai/log || status=$?
if [ "${status}" -eq 2 ]; then
    echo "  ! some runs failed, see zoo/chiplet4ai/log/failed_runs.txt" >&2
elif [ "${status}" -ne 0 ]; then
    echo "  ! archx -x failed (exit ${status})" >&2
    exit "${status}"
fi

echo "Query performance models and generate figures."
python -m zoo.chiplet4ai.results.figure_generation
