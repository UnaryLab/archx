import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__)) + "/"
# run as `python <path>/query.py`, so common/ is put on the import path
sys.path.insert(0, BASE + "../common")
import query_shared as shared

# the PE module name, as PE_MODULE in performance.py (not importable here: it imports through the zoo package)
PE_MODULE = "pe"

runs = shared.load_runs("eyeriss cg", BASE)
csv_path = BASE + "../results/cg/"
os.makedirs(csv_path, exist_ok=True)
shared.write_breakdowns(runs, csv_path, PE_MODULE, {"": shared.TAGS})
shared.write_layer_csv(runs, csv_path + "layer.csv", PE_MODULE)
