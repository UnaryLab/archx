#!/bin/bash

# Run the first script to generate dictionary
echo "Generating yaml files."
python3 -m zoo.llm.scripts.generate_dicts

# Run the second script to generate runs
echo "Generating runs."
python3 -m zoo.llm.scripts.generate_runs

# Call the run_script.sh
echo "Simulate performance models."
bash ./run_script.sh zoo/llm/scripts/runs.txt

echo "Query performance models and generate figures."
python -m zoo.llm.results.figure_generation