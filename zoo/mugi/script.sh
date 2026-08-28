#!/bin/bash
# Build + run every mugi-zoo design. Each design is a single AGraph description.py
# (compiled internally into its config set + runs.txt); there is no more
# generate_dicts/generate_runs step.
set -e

echo "Setting up environment."
export PYTHONPATH="$(pwd)/zoo:$(pwd):$PYTHONPATH"

# Select designs to build: names passed as arguments, else every design.
# Usage: script.sh [design ...]   e.g.  script.sh mugi systolic
if [ "$#" -gt 0 ]; then
    designs=("$@")
else
    designs=()
    for dir in zoo/mugi/designs/*/; do
        [ -f "${dir}description.py" ] || continue
        designs+=("$(basename "$dir")")
    done
fi

echo "Building and simulating designs: ${designs[*]}"
for name in "${designs[@]}"; do
    dir="zoo/mugi/designs/$(basename "$name")/"
    if [ ! -f "${dir}description.py" ]; then
        echo "  ! skipping '${name}': ${dir}description.py not found" >&2
        continue
    fi
    echo "  - ${dir}description.py"
    archx -compile "${dir}description.py" -r "${dir}description"
    archx -r "${dir}description" -extract "${dir}description/configurations.csv"
    # archx -x runs this design's batch in one process pool; -r only holds the
    # batch log. A failed batch must not stop the remaining designs.
    status=0
    archx -x "${dir}description/runs.txt" -r "${dir}description" || status=$?
    if [ "${status}" -eq 2 ]; then
        echo "  ! design '${name}': some runs failed, see ${dir}description/failed_runs.txt" >&2
    elif [ "${status}" -ne 0 ]; then
        echo "  ! design '${name}': archx -x failed (exit ${status})" >&2
    fi
done

echo "Query performance models and generate figures."
python -m zoo.mugi.results.figure_generation
