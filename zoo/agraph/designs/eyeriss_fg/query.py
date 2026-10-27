from archx.metric import (
    aggregate_event_metric,
    aggregate_tag_metric,
    aggregate_event_count,
)
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.metric import load_metric_dict
from loguru import logger
import pandas as pd
import os
import sys

logger.remove()

configs = ["arch_0"]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
# area and power breakdowns by file prefix: each tag's total, then its modules not already listed;
# the first tag is the whole the percentage is taken of, and no tag here holds dram
breakdowns = {
    "": [
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
    ],
    "pe_": ["pe", "spad", "fifo", "mac", "control"],
}


def tag_metric(event_graph, metric_dict, metric, workload, tag):
    return aggregate_tag_metric(
        event_graph=event_graph,
        metric_dict=metric_dict,
        metric=metric,
        workload=workload,
        tag=tag,
    )["value"]


def tag_rows(event_graph, tags):
    # a tag row leaves module blank; a module's name is one of its tags, so its row is a tag query too
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


# area rows per prefix go to results/ only; power rows per (layer, prefix) and count rows per layer:
# the workload's go to results/, each layer's to results/<layer>/
area_rows = {}
power_rows = {}
count_rows = {}

base_path = "zoo/agraph/designs/eyeriss_fg/"
run_path = f"{base_path}description/runs/alexnet/"

# zoo/agraph/script.sh runs with set -e, so exit 0 when no run exists
missing = [
    config
    for config in configs
    if not os.path.exists(f"{run_path}{config}/config_0/checkpoint.json")
]
if missing:
    print(
        f"eyeriss_fg: no compiled run found for {missing} under {run_path}; "
        f"skipping query (run `archx -compile` for this design first)."
    )
    sys.exit(0)

csv_path = f"{base_path}results/"

for config in configs:
    path = f"{run_path}{config}/config_0"
    event_graph = load_event_graph(f"{path}/checkpoint.json")
    architecture = load_architecture_dict(f"{path}/architecture.yaml")
    metric = load_metric_dict(f"{path}/metric.yaml")

    array_dim = architecture["multiplier"]["instance"]
    array_dim_str = "x".join([str(dim) for dim in array_dim])

    workload = "alexnet"

    assert (
        metric["dynamic_energy"]["unit"],
        metric["runtime"]["unit"],
        metric["leakage_power"]["unit"],
    ) == ("nJ", "ms", "mW")

    rows = {prefix: tag_rows(event_graph, tags) for prefix, tags in breakdowns.items()}

    # area does not change across layers, so it is only queried for the workload
    for prefix in breakdowns:
        for tag, module in rows[prefix]:
            area = tag_metric(event_graph, metric, "area", workload, module or tag)
            area_rows.setdefault(prefix, []).append([array_dim_str, tag, module, area])

    for layer in layers + [workload]:
        runtime = aggregate_event_metric(
            event_graph=event_graph,
            metric_dict=metric,
            metric="runtime",
            workload=workload,
            event=layer,
        )["value"]
        for prefix in breakdowns:
            for tag, module in rows[prefix]:
                # a layer passed as the workload limits the total to that layer
                energy = tag_metric(
                    event_graph, metric, "dynamic_energy", layer, module or tag
                )
                leakage = tag_metric(
                    event_graph, metric, "leakage_power", layer, module or tag
                )
                # nJ / ms = uW, so scale by 1e-3 for mW
                dynamic_power = energy / runtime * 1e-3
                power_rows.setdefault((layer, prefix), []).append(
                    [
                        array_dim_str,
                        tag,
                        module,
                        dynamic_power,
                        leakage,
                        dynamic_power + leakage,
                    ]
                )
        # count is the only output that includes dram
        for module in architecture:
            count = aggregate_event_count(
                event_graph=event_graph, workload=layer, event=module
            )
            count_rows.setdefault(layer, []).append([array_dim_str, module, count])

for layer in layers + [workload]:
    out_path = csv_path if layer == workload else f"{csv_path}{layer}/"
    os.makedirs(out_path, exist_ok=True)
    for prefix, tags in breakdowns.items():
        write_breakdown(
            pd.DataFrame(
                power_rows[layer, prefix],
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
            f"{out_path}{prefix}power.csv",
        )
    pd.DataFrame(count_rows[layer], columns=["arch", "module", "count"]).astype(
        {"count": float}
    ).to_csv(f"{out_path}count.csv", index=False, float_format="%.3e")

for prefix, tags in breakdowns.items():
    write_breakdown(
        pd.DataFrame(
            area_rows[prefix],
            columns=["arch", "tag", "module", f'area {metric["area"]["unit"]}'],
        ),
        tags,
        f"{csv_path}{prefix}area.csv",
    )
