set -e

# Register the interface bundles. The framework ships no interface data; each
# design queries these characterization libraries at run time, so they must be
# registered into archx first. Re-running is safe (existing ones are skipped).
for iface in agraph/interface/*/; do
    name=$(basename "$iface")
    archx -ireg -iname "$name" -idir "$iface"
done

for dir in agraph/designs/*/; do
    archx -compile $dir/description.py -r $dir/description -full
    python $dir/query.py
done

for script in agraph/res/scripts/*.py; do
    echo "Running $script..."
    python $script
done