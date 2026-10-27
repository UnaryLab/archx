"""Run this design on gem5 once through the archx gem5 interface; run from the archx repo root."""

from archx.architecture import create_architecture_dict
from archx.interface import query_interface
from archx.workload import create_workload_dict

design_dir = "zoo/agraph/designs/riscv_mcpat/"

result = query_interface(
    "xeon_tulsa",
    {
        "interface": "gem5",
        "architecture": create_architecture_dict(
            design_dir + "xeon_tulsa.architecture.yaml"
        ),
        "workload": create_workload_dict(design_dir + "xeon_tulsa.workload.yaml"),
    },
)
print(result["stats_file"])
