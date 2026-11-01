import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__)) + "/"
# run as `python <path>/query.py`, so common/ is put on the import path
sys.path.insert(0, BASE + "../common")
import query_shared as shared

# the PE module name, as PE_MODULE in performance.py (not importable here: it imports through the zoo package)
PE_MODULE = "multiplier"

# after pe, so their modules are already listed and each PE group is a single total line
FG_TAGS = shared.TAGS[:3] + ["spad", "fifo", "mac", "control"] + shared.TAGS[3:]

runs = shared.load_runs("eyeriss fg", BASE)
csv_path = BASE + "../results/fg/"
# the pe_ breakdown splits the PE into its submodule groups
shared.write_breakdowns(
    runs,
    csv_path,
    PE_MODULE,
    {"": FG_TAGS, "pe_": ["pe", "spad", "fifo", "mac", "control"]},
)
shared.write_layer_csv(runs, csv_path + "layer.csv", PE_MODULE)
