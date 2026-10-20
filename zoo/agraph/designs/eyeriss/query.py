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
tags = ["onchip", "pe", "filter_spad", "glb", "noc", "codec"]
counted = [
    "pe",
    "ifmap_psum_glb",
    "filt_glb",
    "dram",
    "filter_spad",
    "bus_gin_ifmap_y",
    "bus_gin_ifmap_x",
    "bus_gin_filt_y",
    "bus_gin_filt_x",
    "bus_gin_psum_y",
    "bus_gin_psum_x",
    "bus_gon_psum_y",
    "bus_gon_psum_x",
    "rlc_decoder",
    "rlc_encoder",
    "relu",
]


def add_percent_of_onchip(df, value):
    onchip = df[df["tag"] == "onchip"].set_index("arch")[value]
    df["percent_of_onchip"] = df[value] / df["arch"].map(onchip) * 100
    return df


def tag_metric(event_graph, metric_dict, metric, workload, tag):
    # filter_spad has no tag of its own, so query it by module name
    if tag == "filter_spad":
        result = aggregate_event_metric(
            event_graph=event_graph,
            metric_dict=metric_dict,
            metric=metric,
            workload=workload,
            event=tag,
        )
    else:
        result = aggregate_tag_metric(
            event_graph=event_graph,
            metric_dict=metric_dict,
            metric=metric,
            workload=workload,
            tag=tag,
        )
    return result["value"]


layer_df = None
area_df = None
leakage_df = None
dynamic_df = None
energy_df = None
power_df = None
count_df = None

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
area_path = csv_path + "area.csv"
leakage_path = csv_path + "leakage_power.csv"
dynamic_path = csv_path + "dynamic_power.csv"
energy_path = csv_path + "energy.csv"
power_path = csv_path + "power.csv"
count_path = csv_path + "count.csv"

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
                "tag",
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
                "percent_of_onchip",
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
        rows = []
        for tag in tags:
            # a layer passed as the workload limits the tag total to that layer
            energy = tag_metric(event_graph, metric, "dynamic_energy", layer, tag)
            leakage = tag_metric(event_graph, metric, "leakage_power", layer, tag)
            dynamic_power = energy / runtime["value"] * 1e-3
            rows.append([tag, energy, dynamic_power, leakage, dynamic_power + leakage])
        onchip_power = rows[0][4]

        for tag, energy, dynamic_power, leakage, total_power in rows:
            layer_df.loc[len(layer_df)] = [
                array_dim_str,
                "total" if layer == workload else layer,
                tag,
                macs[layer],
                cycle_count["value"],
                runtime["value"],
                runtime["value"] - dram_runtime,
                glb_accesses_mb,
                dram_accesses * dram_word_mb,
                energy,
                dynamic_power,
                leakage,
                total_power,
                total_power / onchip_power * 100,
            ]

    if area_df is None:
        area_df = pd.DataFrame(
            columns=["arch", "tag", f'area {metric["area"]["unit"]}']
        )

    if leakage_df is None:
        leakage_df = pd.DataFrame(
            columns=["arch", "tag", f'leakage_power {metric["leakage_power"]["unit"]}']
        )

    if dynamic_df is None:
        dynamic_df = pd.DataFrame(columns=["arch", "tag", "dynamic_power mW"])

    if energy_df is None:
        energy_df = pd.DataFrame(
            columns=[
                "arch",
                "tag",
                "dynamic_energy nJ",
                "leakage_energy nJ",
                "total_energy nJ",
            ]
        )

    if power_df is None:
        power_df = pd.DataFrame(
            columns=[
                "arch",
                "tag",
                "dynamic_power mW",
                "leakage_power mW",
                "total_power mW",
            ]
        )

    workload_runtime = aggregate_event_metric(
        event_graph=event_graph,
        metric_dict=metric,
        metric="runtime",
        workload=workload,
        event=workload,
    )["value"]

    for tag in tags:
        area = tag_metric(event_graph, metric, "area", workload, tag)
        leakage_power = tag_metric(event_graph, metric, "leakage_power", workload, tag)
        dynamic_energy = tag_metric(
            event_graph, metric, "dynamic_energy", workload, tag
        )

        area_df.loc[len(area_df)] = [array_dim_str, tag, area]
        # mW * ms = uJ, so scale by 1e3 for nJ
        leakage_energy = leakage_power * workload_runtime * 1e3
        energy_df.loc[len(energy_df)] = [
            array_dim_str,
            tag,
            dynamic_energy,
            leakage_energy,
            dynamic_energy + leakage_energy,
        ]
        leakage_df.loc[len(leakage_df)] = [array_dim_str, tag, leakage_power]
        # nJ / ms = uW, so scale by 1e-3 for mW
        dynamic_power = dynamic_energy / workload_runtime * 1e-3
        dynamic_df.loc[len(dynamic_df)] = [array_dim_str, tag, dynamic_power]
        power_df.loc[len(power_df)] = [
            array_dim_str,
            tag,
            dynamic_power,
            leakage_power,
            dynamic_power + leakage_power,
        ]

    if count_df is None:
        count_df = pd.DataFrame(columns=["arch", "module", "count"])

    for module in counted:
        count = aggregate_event_count(
            event_graph=event_graph, workload=workload, event=module
        )
        count_df.loc[len(count_df)] = [array_dim_str, module, count]

layer_df.to_csv(layer_path, index=False)
add_percent_of_onchip(area_df, area_df.columns[2]).sort_values(
    "percent_of_onchip", ascending=False
).to_csv(area_path, index=False)
add_percent_of_onchip(leakage_df, leakage_df.columns[2]).sort_values(
    "percent_of_onchip", ascending=False
).to_csv(leakage_path, index=False)
add_percent_of_onchip(dynamic_df, "dynamic_power mW").sort_values(
    "percent_of_onchip", ascending=False
).to_csv(dynamic_path, index=False)
add_percent_of_onchip(energy_df, "total_energy nJ").sort_values(
    "percent_of_onchip", ascending=False
).to_csv(energy_path, index=False)
add_percent_of_onchip(power_df, "total_power mW").sort_values(
    "percent_of_onchip", ascending=False
).to_csv(power_path, index=False)
count_df.to_csv(count_path, index=False)
