"""Query parts shared by the cg and fg Eyeriss variants: run loading and the area, power, count and layer CSVs."""

from archx.metric import (
    aggregate_event_metric,
    aggregate_tag_metric,
    aggregate_event_count,
    load_metric_dict,
)
from archx.architecture import load_architecture_dict
from archx.event import load_event_graph
from archx.workload import load_workload_dict
from loguru import logger
import pandas as pd
import os
import sys
import performance_shared as performance

logger.remove()

CONFIGS = ["arch_0"]
LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5"]
WORKLOAD = "alexnet"
# area and power rows: each tag's total, then its modules not already listed;
# the first tag is the whole the percentage is taken of, and no tag here holds dram
TAGS = [
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
    # a module row queries the module itself: the cg pe module shares its name with the pe tag, which also holds filter_spad
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


def load_runs(name, base_path):
    """Return (config_path, event_graph, architecture, metric) per config compiled under base_path."""
    run_path = f"{base_path}description/runs/{WORKLOAD}/"

    # zoo/agraph/script.sh runs with set -e, so exit 0 when no run exists
    missing = [
        config
        for config in CONFIGS
        if not os.path.exists(f"{run_path}{config}/config_0/checkpoint.json")
    ]
    if missing:
        print(
            f"{name}: no compiled run found for {missing} under {run_path}; "
            f"skipping query (run `archx -compile` for this design first)."
        )
        sys.exit(0)

    runs = []
    for config in CONFIGS:
        path = f"{run_path}{config}/config_0"
        event_graph = load_event_graph(f"{path}/checkpoint.json")
        architecture = load_architecture_dict(f"{path}/architecture.yaml")
        metric = load_metric_dict(f"{path}/metric.yaml")
        assert (
            metric["dynamic_energy"]["unit"],
            metric["runtime"]["unit"],
            metric["leakage_power"]["unit"],
        ) == ("nJ", "ms", "mW")
        runs.append((path, event_graph, architecture, metric))
    return runs


def array_dim_str(architecture, pe_module):
    return "x".join([str(dim) for dim in architecture[pe_module]["instance"]])


def write_breakdowns(runs, csv_path, pe_module, breakdowns):
    """Write {prefix}area.csv, and per layer {prefix}power.csv and count.csv, for each breakdown.

    breakdowns maps a file prefix to its tag list; pe_module is the module whose instance is the PE array.
    """
    # area rows per prefix go to results/ only; power rows per (layer, prefix) and count rows per layer:
    # the workload's go to results/, each layer's to results/<layer>/
    area_rows = {}
    power_rows = {}
    count_rows = {}

    for _, event_graph, architecture, metric in runs:
        arch = array_dim_str(architecture, pe_module)
        rows = {
            prefix: tag_rows(event_graph, tags) for prefix, tags in breakdowns.items()
        }

        # area does not change across layers, so it is only queried for the workload
        for prefix in breakdowns:
            for tag, module in rows[prefix]:
                area = row_metric(event_graph, metric, "area", WORKLOAD, tag, module)
                area_rows.setdefault(prefix, []).append([arch, tag, module, area])

        for layer in LAYERS + [WORKLOAD]:
            runtime = aggregate_event_metric(
                event_graph=event_graph,
                metric_dict=metric,
                metric="runtime",
                workload=WORKLOAD,
                event=layer,
            )["value"]
            for prefix in breakdowns:
                for tag, module in rows[prefix]:
                    # a layer passed as the workload limits the total to that layer
                    energy = row_metric(
                        event_graph, metric, "dynamic_energy", layer, tag, module
                    )
                    leakage = row_metric(
                        event_graph, metric, "leakage_power", layer, tag, module
                    )
                    # nJ / ms = uW, so scale by 1e-3 for mW
                    dynamic_power = energy / runtime * 1e-3
                    power_rows.setdefault((layer, prefix), []).append(
                        [
                            arch,
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
                count_rows.setdefault(layer, []).append([arch, module, count])

    for layer in LAYERS + [WORKLOAD]:
        out_path = csv_path if layer == WORKLOAD else f"{csv_path}{layer}/"
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


def write_layer_csv(runs, path, pe_module):
    """Per layer and in total: cycles, runtime, GLB/DRAM traffic and on-chip power (Table V validation)."""
    layer_df = None
    workload = WORKLOAD
    for config_path, event_graph, architecture, metric in runs:
        workload_dict = load_workload_dict(f"{config_path}/workload.yaml")
        arch = array_dim_str(architecture, pe_module)

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
            for index, layer in enumerate(LAYERS)
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

        for layer in LAYERS + [workload]:
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
            # a layer passed as the workload limits the total to that layer
            energy = row_metric(
                event_graph, metric, "dynamic_energy", layer, "onchip", ""
            )
            leakage = row_metric(
                event_graph, metric, "leakage_power", layer, "onchip", ""
            )
            # nJ / ms = uW, so scale by 1e-3 for mW
            dynamic_power = energy / runtime["value"] * 1e-3
            layer_df.loc[len(layer_df)] = [
                arch,
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
    layer_df.to_csv(path, index=False)
