#!/bin/sh
# Compile and query the eyeriss design. Run from the archx base dir.
set -e
dir=zoo/agraph/designs/eyeriss
archx -compile $dir/description.py -r $dir/description -full
python $dir/query.py
