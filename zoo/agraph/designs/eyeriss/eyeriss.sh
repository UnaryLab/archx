#!/bin/sh
# Compile, query and plot figures for both eyeriss variants (cg, fg). Run from the archx base dir.
set -e
dir=zoo/agraph/designs/eyeriss
for v in cg fg; do
    archx -compile $dir/$v/description.py -r $dir/$v/description -full
    python $dir/$v/query.py
    for f in $dir/$v/figure_generation/*.py; do python "$f"; done
done
