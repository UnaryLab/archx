from archx.metric import (
    aggregate_event_metric,
    aggregate_tag_metric,
    aggregate_event_count,
)
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from archx.workload import load_workload_dict
from loguru import logger
import pandas as pd
import importlib.util
import os
import sys

logger.remove()

configs = ["arch_0"]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
# area and power rows: each tag's total, then its modules not already listed;
# the first tag is the whole the percentage is taken of, and no tag here holds dram
tags = [
    "onchip",
    "pe",
    "filter_spad",
    "glb",
    "ifmap_noc",
    "filt_noc",
    "psum_noc",
    # after the three nocs, so its buses are already listed and it is a single total line
    "noc",
    "codec",
]


def row_metric(event_graph, metric_dict, metric, workload, tag, module):
    # a module row queries the module itself: the pe module shares its name with the pe tag, which also holds filter_spad
    if module:
        return aggregate_event_metric(
            event_graph=event_graph,
            metric_dict=metric_dict,
            metric=metric,
            workload=workload,
            event=module,
        )["value"]
    return aggregate_tag_metric(
        event_graph=event_graph,
        metric_dict=metric_dict,
        metric=metric,
        workload=workload,
        tag=tag,
    )["value"]


def tag_rows(event_graph, tags):
    # a tag row leaves module blank
    # each module is listed once, under the first tag after the whole that holds it, so the whole lists only the rest
    listed = set()
    rows = []
    for tag in tags[1:] + tags[:1]:
        modules = [m for m in event_graph.get_tag_modules(tag) if m not in listed]
        listed.update(modules)
        rows += [(tag, "")] + [(tag, module) for module in modules]
    return rows


def write_breakdown(df, tags, path):
    # the last column is the total the percentage is taken of
    value = df.columns[-1]
    whole = df[(df["tag"] == tags[0]) & (df["module"] == "")].set_index("arch")[value]
    df[f"percent_of_{tags[0]}"] = df[value] / df["arch"].map(whole) * 100
    # tags keep their order, each with its total first and then its modules from largest to smallest
    df["tag_order"] = df["tag"].map(tags.index)
    df["is_module"] = df["module"] != ""
    df.sort_values(
        ["arch", "tag_order", "is_module", value], ascending=[True, True, True, False]
    ).drop(columns=["tag_order", "is_module"]).to_csv(path, index=False)


layer_df = None
# area rows go to results/ only; power and count rows per layer: the workload's go to results/, each layer's to results/<layer>/
area_rows = []
power_rows = {}
count_rows = {}

base_path = "zoo/agraph/designs/eyeriss/"
run_path = f"{base_path}description/runs/alexnet/"

spec = importlib.util.spec_from_file_location(
    "performance", f"{base_path}performance.py"
)
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)

# zoo/agraph/script.sh runs with set -e, so exit 0 when no run exists
missing = [
    config
    for config in configs
    if not os.path.exists(f"{run_path}{config}/config_0/checkpoint.json")
]
if missing:
    print(
        f"eyeriss: no compiled run found for {missing} under {run_path}; "
        f"skipping query (run `archx -compile` for this design first)."
    )
    sys.exit(0)

csv_path = f"{base_path}results/"
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

layer_path = csv_path + "layer.csv"

for config in configs:
    path = f"{run_path}{config}/config_0"
    event_graph = load_event_graph(f"{path}/checkpoint.json")
    architecture = load_architecture_dict(f"{path}/architecture.yaml")
    metric = load_metric_dict(f"{path}/metric.yaml")
    workload_dict = load_workload_dict(f"{path}/workload.yaml")

    array_dim = architecture["pe"]["instance"]
    array_dim_str = "x".join([str(dim) for dim in array_dim])

    workload = "alexnet"

    assert (
        metric["dynamic_energy"]["unit"],
        metric["runtime"]["unit"],
        metric["leakage_power"]["unit"],
    ) == ("nJ", "ms", "mW")

    if layer_df is None:
        layer_df = pd.DataFrame(
            columns=[
                "arch",
                "layer",
                "macs",
                f'cycle_count {metric["cycle_count"]["unit"]}',
                f'total_runtime {metric["runtime"]["unit"]}',
                f'processing_runtime {metric["runtime"]["unit"]}',
                "glb_accesses MB",
                "dram_accesses MB",
                "dynamic_energy nJ",
                "dynamic_power mW",
                "leakage_power mW",
                "total_power mW",
            ]
        )

    macs = {
        layer: performance._layer(workload_dict, index)["macs"]
        for index, layer in enumerate(layers)
    }
    macs[workload] = sum(macs.values())
    # glb and dram events are counted in words of the memory's width
    glb_event_word_mb = {
        event: architecture[module]["query"]["width"] / 8 / 1e6
        for event, module in [
            ("ifmap_psum_glb_read", "ifmap_psum_glb"),
            ("ifmap_psum_glb_write", "ifmap_psum_glb"),
            ("filt_glb_read", "filt_glb"),
        ]
    }
    dram_word_mb = architecture["dram"]["query"]["width"] / 8 / 1e6

    rows = tag_rows(event_graph, tags)

    # area does not change across layers, so it is only queried for the workload
    for tag, module in rows:
        area = row_metric(event_graph, metric, "area", workload, tag, module)
        area_rows.append([array_dim_str, tag, module, area])

    for layer in layers + [workload]:
        cycle_count = aggregate_event_metric(
            event_graph=event_graph,
            metric_dict=metric,
            metric="cycle_count",
            workload=workload,
            event=layer,
        )
        runtime = aggregate_event_metric(
            event_graph=event_graph,
            metric_dict=metric,
            metric="runtime",
            workload=workload,
            event=layer,
        )
        # the dram events are sequential, so removing their runtime leaves the on-chip runtime
        dram_runtime = sum(
            aggregate_event_metric(
                event_graph=event_graph,
                metric_dict=metric,
                metric="runtime",
                workload=layer,
                event=e,
            )["value"]
            for e in ["dram_read", "dram_write"]
        )
        glb_accesses_mb = sum(
            aggregate_event_count(event_graph=event_graph, workload=layer, event=e)
            * word_mb
            for e, word_mb in glb_event_word_mb.items()
        )
        dram_accesses = sum(
            aggregate_event_count(event_graph=event_graph, workload=layer, event=e)
            for e in ["dram_read", "dram_write"]
        )
        for tag, module in rows:
            # a layer passed as the workload limits the total to that layer
            energy = row_metric(
                event_graph, metric, "dynamic_energy", layer, tag, module
            )
            leakage = row_metric(
                event_graph, metric, "leakage_power", layer, tag, module
            )
            # nJ / ms = uW, so scale by 1e-3 for mW
            dynamic_power = energy / runtime["value"] * 1e-3
            power_rows.setdefault(layer, []).append(
                [
                    array_dim_str,
                    tag,
                    module,
                    dynamic_power,
                    leakage,
                    dynamic_power + leakage,
                ]
            )

            if tag == "onchip" and module == "":
                layer_df.loc[len(layer_df)] = [
                    array_dim_str,
                    "total" if layer == workload else layer,
                    macs[layer],
                    cycle_count["value"],
                    runtime["value"],
                    runtime["value"] - dram_runtime,
                    glb_accesses_mb,
                    dram_accesses * dram_word_mb,
                    energy,
                    dynamic_power,
                    leakage,
                    dynamic_power + leakage,
                ]
        # count is the only output that includes dram
        for module in architecture:
            count = aggregate_event_count(
                event_graph=event_graph, workload=layer, event=module
            )
            count_rows.setdefault(layer, []).append([array_dim_str, module, count])

layer_df.to_csv(layer_path, index=False)
write_breakdown(
    pd.DataFrame(
        area_rows, columns=["arch", "tag", "module", f'area {metric["area"]["unit"]}']
    ),
    tags,
    f"{csv_path}area.csv",
)
for layer in layers + [workload]:
    out_path = csv_path if layer == workload else f"{csv_path}{layer}/"
    os.makedirs(out_path, exist_ok=True)
    write_breakdown(
        pd.DataFrame(
            power_rows[layer],
            columns=[
                "arch",
                "tag",
                "module",
                "dynamic_power mW",
                "leakage_power mW",
                "total_power mW",
            ],
        ),
        tags,
        f"{out_path}power.csv",
    )
    pd.DataFrame(count_rows[layer], columns=["arch", "module", "count"]).astype(
        {"count": float}
    ).to_csv(f"{out_path}count.csv", index=False, float_format="%.3e")
