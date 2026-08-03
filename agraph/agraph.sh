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

# agraph/runtime/agraph_runtime.csv is a static input to the runtime figure
# (res/scripts/runtime_fig.py) and is not regenerated here.
for dir in agraph/designs/*/; do
    archx -compile $dir/description.py -r $dir/description -full
    python $dir/query.py
done

for script in agraph/res/scripts/*.py; do
    echo "Running $script..."
    python $script
done