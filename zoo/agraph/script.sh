set -e

# Usage: sh zoo/agraph/script.sh [design]
# With no design, runs every design in zoo/agraph/designs/ and then the figure
# scripts. With a design name (e.g. eyeriss), runs only that design.
#
# zoo/agraph/runtime/ holds the external EDA measurements and the recorded
# A-Graph runtimes; both are static inputs to res/scripts/runtime_fig.py and are
# not regenerated here.
design=${1:-*}
for dir in zoo/agraph/designs/$design/; do
    if [ ! -d "$dir" ]; then
        echo "  ! unknown design '$1', choose from: $(ls zoo/agraph/designs | tr '\n' ' ')" >&2
        exit 1
    fi
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

[ -z "$1" ] || exit 0

for script in zoo/agraph/res/scripts/*.py; do
    echo "Running $script..."
    python $script
done
