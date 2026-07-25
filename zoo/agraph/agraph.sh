set -e
for dir in zoo/agraph/designs/*/; do
    archx -compile $dir/description.py -r $dir/description -full
    python $dir/query.py
done

for script in zoo/agraph/res/scripts/*.py; do
    echo "Running $script..."
    python $script
done