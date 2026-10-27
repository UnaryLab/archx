from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

design_path = "zoo/agraph/designs/eyeriss/"
pdf_path = design_path + "figures/energy_breakdown.pdf"

theirs = pd.read_csv(
    design_path + "validation/accelergy_energy_breakdown.csv",
    skipinitialspace=True,
    index_col="design",
)
# over the whole workload every tag runs for the same time, so its share of power is its share of energy
power = pd.read_csv(design_path + "results/power.csv", keep_default_na=False)
# tag totals; the pe tag also holds filter_spad, so pe here is the pe module's own row
ours = power[power["module"] == ""].set_index("tag")["percent_of_onchip"].drop("onchip")
ours["pe"] = power[power["module"] == "pe"]["percent_of_onchip"].item()

# Bottom to top: PE array, GLB, NoC, Other; each merged segment is the sum of its CSV columns.
theirs_segments = {
    "pe_array": ["pe_array"],
    "glb": ["sharedbuffer", "weightsbuffer"],
    "noc": ["psumrdnoc", "psumwrnoc", "ifmapnoc", "weightsnoc"],
}
ours_segments = {
    "pe_array": ["pe"],
    "filter_spad": ["filter_spad"],
    "glb": ["glb"],
    "noc": ["ifmap_noc", "filt_noc", "psum_noc"],
    "other": ["codec"],
}
theirs_order = list(theirs_segments)
ours_order = list(ours_segments)

colors = {
    "pe_array": "#4697bd",
    "filter_spad": "#9cc9e0",
    "glb": "#e38a3c",
    "noc": "#3bac3b",
    "other": "#8e6bbf",
}
legend_labels = [
    ("pe_array", "PE array"),
    ("filter_spad", "Filter spad"),
    ("glb", "GLB"),
    ("noc", "NoC"),
    ("other", "Other"),
]

bars_spec = [
    (
        "Ground\ntruth",
        theirs_order,
        [
            theirs.loc["ground_truth", cols].sum() / 100
            for cols in theirs_segments.values()
        ],
    ),
    (
        "Accelergy",
        theirs_order,
        [
            theirs.loc["accelergy", cols].sum() / 100
            for cols in theirs_segments.values()
        ],
    ),
    ("Ours", ours_order, [ours[cols].sum() / 100 for cols in ours_segments.values()]),
]
for (_, _, heights), row in zip(bars_spec[:2], ["ground_truth", "accelergy"]):
    assert np.isclose(
        heights[1],
        (theirs.loc[row, "sharedbuffer"] + theirs.loc[row, "weightsbuffer"]) / 100,
    )
    assert np.isclose(
        heights[2],
        theirs.loc[row, ["psumrdnoc", "psumwrnoc", "ifmapnoc", "weightsnoc"]].sum()
        / 100,
    )
assert np.isclose(bars_spec[1][2][0], theirs.loc["accelergy", "pe_array"] / 100)
assert abs(sum(bars_spec[2][2]) - 1.0) < 1e-6, sum(bars_spec[2][2])

width = 252 / 72
height = width / 1.6
bar_width = 0.5
x = [0, 1, 2]

fig, ax = plt.subplots(1, 1, figsize=(width, height))
ax.tick_params(axis="x", labelsize=8, pad=0.5)
ax.tick_params(axis="y", labelsize=8, pad=0.5)
ax.grid(axis="y", linestyle=(0, (2, 7)), linewidth=0.4)
ax.set_axisbelow(True)

edges = {}
for xi, (_, order, heights) in zip(x, bars_spec):
    bars = ax.bar(
        [xi] * len(order),
        heights,
        bar_width,
        bottom=np.cumsum([0] + heights[:-1]),
        color=[colors[c] for c in order],
        edgecolor="white",
        linewidth=0.3,
    )
    for c, h, bar in zip(order, heights, bars):
        assert np.isclose(bar.get_height(), h)
        edges[(xi, c)] = (bar.get_y(), bar.get_y() + bar.get_height())

# Accelergy pe_array spans our pe + filter_spad: link the bottom edges and the top edges.
x_left, x_right = x[1] + bar_width / 2, x[2] - bar_width / 2
for y_left, y_right in [
    (edges[(x[1], "pe_array")][0], edges[(x[2], "pe_array")][0]),
    (edges[(x[1], "pe_array")][1], edges[(x[2], "filter_spad")][1]),
]:
    ax.plot([x_left, x_right], [y_left, y_right], color="#333333", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([name for name, _, _ in bars_spec], fontsize=8)
ax.set_xlim(-0.5, 2.5)
# start below 0 so the bottom connector line isn't hidden under the x-axis
ax.set_ylim(-0.04, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:g}"))
ax.set_ylabel("Normalized energy", fontsize=9, labelpad=2)

handles = [
    Patch(facecolor=colors[c], edgecolor="white", linewidth=0.3, label=lbl)
    for c, lbl in legend_labels
]
ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(handles),
    fontsize=8,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    handlelength=1.0,
    handleheight=0.8,
    handletextpad=0.4,
    columnspacing=1.0,
)

plt.tight_layout(pad=1.0)
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
