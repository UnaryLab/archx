#!/bin/sh
# Compile and query the eyeriss_fg design. Run from the archx base dir.
set -e
dir=zoo/agraph/designs/eyeriss_fg
archx -compile $dir/description.py -r $dir/description -full
python $dir/query.py
