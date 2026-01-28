#!/bin/bash
# This script runs the configs specified in the input file
# Usage: bash ./run_script.sh <input_file>
# Each line in the input file should be the config for a single run
# Be sure to add end of line character at the end of the file
# Runs that throw errors will be logged in failed_runs.txt

start=`date +%s`
ncore=$(nproc --all)
counter=0
error_log="failed_runs.txt"

# Clear error log at start
> $error_log

while IFS= read -r line; do
    # Run archx and check its exit status
    archx $line &
    pid=$!
    pids[$counter]=$pid
    cmds[$counter]=$line
    
    echo "Launched $line"
    echo ""
    
    counter=$((counter+1))
    if [ $counter -eq $ncore ]; then
        # Wait for current batch and check exit status
        for i in ${!pids[@]}; do
            if ! wait ${pids[$i]}; then
                echo "${cmds[$i]}" >> $error_log
            fi
        done
        counter=0
        pids=()
        cmds=()
    fi
done < "$1"

# Check remaining processes
for i in ${!pids[@]}; do
    if ! wait ${pids[$i]}; then
        echo "${cmds[$i]}" >> $error_log
    fi
done

end=`date +%s`
runtime=$((end-start))

echo "Total runtime: $runtime seconds."

if [ -s "$error_log" ]; then
    echo "Some runs failed. Check $error_log for failed commands. Ran while away."
else
    echo "All runs completed successfully"
    rm $error_log
fi