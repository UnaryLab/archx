set -e
for dir in zoo/agraph/designs/*/; do
    # A partially failed batch must not stop this design's query or later designs.
    status=0
    archx -compile $dir/description.py -r $dir/description -full || status=$?
    if [ "${status}" -eq 2 ]; then
        echo "  ! design '${dir}': some runs failed, see $dir/description/failed_runs.txt" >&2
    elif [ "${status}" -ne 0 ]; then
        echo "  ! design '${dir}': archx -compile failed (exit ${status})" >&2
        exit "${status}"
    fi
    python $dir/query.py
done

for script in zoo/agraph/res/scripts/*.py; do
    echo "Running $script..."
    python $script
done