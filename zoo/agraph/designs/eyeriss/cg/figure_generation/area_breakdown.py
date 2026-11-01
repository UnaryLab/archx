"""Eyeriss coarse-grain area breakdown: Eyeriss paper (validation) vs our model, as stacked bars.

One pair of bars: PE array, filter spad (Ours only), GLB, and one merged NoC/ReLU/CODEC/Control
segment, all as percent of on-chip area. Same layout, label logic and colors as the fg figures.
Merged segment: Eyeriss = noc + RLC_ReLU + control + config_scan_chain; Ours = noc + codec.
Ours is read directly from the percent_of_onchip column of results/cg/area.csv. Our pe tag
holds the pe module plus filter_spad, so the PE array segment is the pe module row and the
filter spad is its own segment; the Eyeriss PE array includes the filter spad (noted in the
figure).
Writes figures/cg/area_breakdown.png in the design directory.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "..", "..")
VALIDATION = os.path.join(DESIGN, "validation")
RESULTS = os.path.join(DESIGN, "results", "cg")
OUT = os.path.join(DESIGN, "figures", "cg", "area_breakdown.png")

BAR_WIDTH = 0.45
X = [0, 1]  # Eyeriss, Ours
INSIDE_MIN = 5.0  # segments at least this tall (percent) get their label inside the bar
LABEL_GAP = 5.0  # minimum vertical spacing (percent) between outside labels on one bar
LABEL_DX = 0.12  # horizontal offset of outside labels from the bar edge

# Categories bottom to top: (key, legend name, short name for outside labels, color).
# Colors match the fg figures: PE parts in the blue ramp, GLB orange, merged segment red.
CATS = [
    ("pe", "PE array", "PE array", "#08519C"),
    ("filter_spad", "Filter spad", "Filter spad", "#3182BD"),
    ("glb", "GLB", "GLB", "#F3B94D"),
    ("rest", "NoC/ReLU/CODEC/Control", "NoC/...", "#D9807A"),
]
DARK = {"pe", "filter_spad"}  # segments that need white inside labels

# Eyeriss (paper numbers); the paper has no separate filter spad, so that segment is 0.
core = pd.read_csv(
    os.path.join(VALIDATION, "eyeriss_area_core_breakdown.csv"), skipinitialspace=True
).iloc[0]
EYERISS = {
    "pe": core["pe_array"],
    "filter_spad": 0.0,
    "glb": core["glb"],
    "rest": core["noc"]
    + core["RLC_ReLU"]
    + core["control"]
    + core["config_scan_chain"],
}

# Ours (simulated), read directly from percent_of_onchip
area = pd.read_csv(os.path.join(RESULTS, "area.csv"), keep_default_na=False)
onchip = area[area["module"] == ""].set_index("tag")["percent_of_onchip"]
pe_module = area[(area["tag"] == "pe") & (area["module"] == "pe")][
    "percent_of_onchip"
].item()
# Skip ifmap/filt/psum_noc (inside noc); the filter_spad tag row is inside the pe tag row.
assert abs(onchip[["pe", "glb", "noc", "codec"]].sum() - onchip["onchip"]) < 1e-6
assert abs(pe_module + onchip["filter_spad"] - onchip["pe"]) < 1e-6
OURS = {
    "pe": pe_module,
    "filter_spad": onchip["filter_spad"],
    "glb": onchip["glb"],
    "rest": onchip["noc"] + onchip["codec"],
}
assert abs(sum(OURS.values()) - 100.0) < 1e-6
assert abs(sum(EYERISS.values()) - 100.0) < 1e-6


def outside_positions(centers):
    """Label y positions for small segments, bottom-up, at least LABEL_GAP apart."""
    pos = []
    for c in centers:
        pos.append(c if not pos else max(c, pos[-1] + LABEL_GAP))
    return pos


def draw_bar(ax, xi, values, side):
    """Stack one bar. side = -1 labels left, +1 labels right. Zero segments are skipped."""
    bottom = 0.0
    small = []
    for key, _, short, color in CATS:
        h = values[key]
        if h == 0:
            continue
        ax.bar(
            xi,
            h,
            BAR_WIDTH,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if h >= INSIDE_MIN:
            ax.text(
                xi,
                bottom + h / 2,
                f"{h:.1f}%",
                ha="center",
                va="center",
                color="white" if key in DARK else "black",
            )
        else:
            small.append((short, bottom + h / 2, h))
        bottom += h
    assert abs(bottom - 100.0) < 1e-6, bottom
    edge_x = xi + side * BAR_WIDTH / 2
    text_x = edge_x + side * LABEL_DX
    # place labels top-down so the topmost label stays at its segment and lower ones move down
    centers = [100.0 - s[1] for s in reversed(small)]
    ys = [100.0 - y for y in outside_positions(centers)]
    for (short, yc, h), yl in zip(reversed(small), ys):
        ax.plot(
            [edge_x, edge_x + side * LABEL_DX * 0.5, text_x - side * 0.01],
            [yc, yl, yl],
            color="#333333",
            linewidth=0.8,
        )
        ax.text(
            text_x,
            yl,
            f"{short} {h:.1f}%",
            ha="left" if side > 0 else "right",
            va="center",
        )


fig, ax = plt.subplots(figsize=(7, 4.5))
draw_bar(ax, X[0], EYERISS, side=-1)
draw_bar(ax, X[1], OURS, side=+1)
ax.set_xticks(X)
ax.set_xticklabels(["Eyeriss", "Ours"])
ax.set_xlim(-0.95, 1.95)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel("Percent of on-chip area (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    handles=[Patch(facecolor=c, label=n) for _, n, _, c in CATS],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    handlelength=2.0,
    handleheight=1.0,
    columnspacing=1.5,
)
NOTE = "Eyeriss PE array includes the filter spad; ours splits it out"
ax.text(0.5, -0.11, NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=8)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {os.path.normpath(OUT)}")
for name, vals in (("Eyeriss", EYERISS), ("Ours", OURS)):
    print(name, " ".join(f"{k}={vals[k]:.2f}" for k, *_ in CATS))
