#!/usr/bin/env bash
set -e
python3 -m pip install -e . --no-deps
start=`date +%s`
bash zoo/llm/llm_script.sh
end=`date +%s`
runtime=$((end-start))
echo "Total runtime: $runtime seconds."