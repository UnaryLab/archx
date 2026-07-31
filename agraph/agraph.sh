set -e

# Register the interface bundles. The framework ships no interface data; each
# design queries these characterization libraries at run time, so they must be
# registered into archx first. Skip any interface that is already registered.
iface_root=$(python -c "import os, archx.interface.interface as m; print(os.path.dirname(m.__file__))")
for iface in agraph/interface/*/; do
    name=$(basename "$iface")
    if [ -d "$iface_root/$name" ]; then
        echo "Interface $name already registered, skipping."
    else
        archx -ireg -iname "$name" -idir "$iface"
    fi
done

# The runtime figure (res/scripts/runtime_fig.py) plots the A-Graph modeling
# time per config. Timing a design's compile measures the whole config sweep,
# so divide by the arch-config count to get the per-config runtime. Only these
# five designs are plotted; the .txt names are the figure's legacy keys.
declare -A runtime_txt=(
    [fft_cg]=fft_course_grain
    [fft_fg]=fft_fine_grain
    [systolic_cg]=systolic_course_grain
    [systolic_fg]=systolic_fine_grain
    [tnn_fg]=tnn
)

for dir in agraph/designs/*/; do
    name=$(basename "$dir")
    start=$(date +%s.%N)
    archx -compile $dir/description.py -r $dir/description -full
    end=$(date +%s.%N)
    python $dir/query.py

    txt=${runtime_txt[$name]:-}
    if [ -n "$txt" ]; then
        nconfig=$(ls -d $dir/description/runs/*/arch_* 2>/dev/null | wc -l)
        [ "$nconfig" -gt 0 ] || nconfig=1
        per_config=$(python -c "print(($end - $start) / $nconfig)")
        echo "Total runtime: $per_config seconds" > agraph/runtime/$txt.txt
    fi
done

for script in agraph/res/scripts/*.py; do
    echo "Running $script..."
    python $script
done