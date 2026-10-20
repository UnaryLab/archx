from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

design_path = "zoo/agraph/designs/eyeriss/"
pdf_path = design_path + "figures/area_breakdown.pdf"

eyeriss = pd.read_csv(
    design_path + "validation/eyeriss_area_core_breakdown.csv", skipinitialspace=True
).iloc[0]
ours = pd.read_csv(design_path + "results/area.csv", index_col="tag")[
    "percent_of_onchip"
].drop("onchip")

# Bottom to top: PE array, GLB, NoC, Other; each merged segment is the sum of its CSV columns.
eyeriss_segments = {
    "pe_array": ["pe_array"],
    "glb": ["glb"],
    "noc": ["noc"],
    "other": ["RLC_ReLU", "control", "config_scan_chain"],
}
ours_segments = {
    "pe": ["pe"],
    "filter_spad": ["filter_spad"],
    "glb": ["glb"],
    "noc": ["noc"],
    "other": ["codec"],
}
eyeriss_order = list(eyeriss_segments)
ours_order = list(ours_segments)

colors = {
    "pe_array": "#4697bd",
    "pe": "#2b6f99",
    "filter_spad": "#9cc9e0",
    "glb": "#e38a3c",
    "noc": "#3bac3b",
    "other": "#8e6bbf",
}
legend_labels = [
    ("pe_array", "PE array"),
    ("pe", "PE"),
    ("filter_spad", "Filter spad"),
    ("glb", "GLB"),
    ("noc", "NoC"),
    ("other", "Other"),
]

eyeriss_heights = [eyeriss[cols].sum() / 100 for cols in eyeriss_segments.values()]
ours_heights = [ours[cols].sum() / 100 for cols in ours_segments.values()]
assert np.isclose(
    eyeriss_heights[-1],
    (eyeriss["RLC_ReLU"] + eyeriss["control"] + eyeriss["config_scan_chain"]) / 100,
)
assert abs(sum(ours_heights) - 1.0) < 1e-6, sum(ours_heights)

width = 252 / 72
height = width / 2.2
bar_width = 0.5
x = [0, 1]

fig, ax = plt.subplots(1, 1, figsize=(width, height))
ax.tick_params(axis="x", labelsize=8, pad=0.5)
ax.tick_params(axis="y", labelsize=8, pad=0.5)
ax.grid(axis="y", linestyle=(0, (2, 7)), linewidth=0.4)
ax.set_axisbelow(True)

bottoms = {}
for xi, order, heights in [
    (x[0], eyeriss_order, eyeriss_heights),
    (x[1], ours_order, ours_heights),
]:
    bars = ax.bar(
        [xi] * len(order),
        heights,
        bar_width,
        bottom=np.cumsum([0] + heights[:-1]),
        color=[colors[c] for c in order],
        edgecolor="white",
        linewidth=0.3,
    )
    for c, bar in zip(order, bars):
        assert np.isclose(bar.get_height(), heights[order.index(c)])
        bottoms[(xi, c)] = (bar.get_y(), bar.get_y() + bar.get_height())

# pe_array spans our pe + filter_spad: link the bottom edges and the top edges.
x_left, x_right = x[0] + bar_width / 2, x[1] - bar_width / 2
for y_left, y_right in [
    (bottoms[(x[0], "pe_array")][0], bottoms[(x[1], "pe")][0]),
    (bottoms[(x[0], "pe_array")][1], bottoms[(x[1], "filter_spad")][1]),
]:
    ax.plot([x_left, x_right], [y_left, y_right], color="#333333", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(["Eyeriss", "Ours"], fontsize=8)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:g}"))
ax.set_ylabel("Normalized area", fontsize=8, labelpad=2)

handles = [
    Patch(facecolor=colors[c], edgecolor="white", linewidth=0.3, label=lbl)
    for c, lbl in legend_labels
]
ax.legend(
    handles=handles[::-1],
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=6,
    frameon=False,
    handlelength=1.0,
    handleheight=0.8,
    handletextpad=0.4,
    labelspacing=0.25,
)

plt.tight_layout(pad=1.0)
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
