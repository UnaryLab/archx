"""Eyeriss coarse-grain energy breakdown: ground truth and Accelergy vs our model, as stacked bars.

Three stacked bars (Ground truth, Accelergy, Ours) with the same colors as the fg figures:
PE array, Filter spad (Ours only), GLB, and one merged NoC/ReLU/CODEC/Control segment.
Ground truth and Accelergy come from validation/accelergy_energy_breakdown.csv:
PE array = pe_array, GLB = sharedbuffer + weightsbuffer, merged = the four NoC columns (the
paper has no codec, so the merged segment is NoC only for these two bars). Their PE array
includes the filter spad. The published rows do not sum to exactly 100 (ground truth 100.1,
Accelergy 99.5), so each bar is renormalized to 100% by dividing by its own sum.
Ours is read from the percent_of_onchip column of results/cg/power.csv: PE array = the pe
module row (pe without filter_spad), Filter spad, GLB, merged = noc + codec (tag rows).
Writes figures/cg/energy_breakdown.png in the design directory.
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
OUT = os.path.join(DESIGN, "figures", "cg", "energy_breakdown.png")

BAR_WIDTH = 0.45
X = [0.0, 1.4, 2.8]  # Ground truth, Accelergy, Ours
INSIDE_MIN = 5.0  # segments at least this tall (percent) get their label inside the bar
LABEL_GAP = 5.0  # minimum vertical spacing (percent) between outside labels on one bar
LABEL_DX = 0.12  # horizontal offset of outside labels from the bar edge

# Categories bottom to top: (key, legend name, short name for outside labels, color).
CATS = [
    ("pe", "PE array", "PE array", "#08519C"),
    ("filter_spad", "Filter spad", "Filter spad", "#3182BD"),
    ("glb", "GLB", "GLB", "#F3B94D"),
    ("rest", "NoC/ReLU/CODEC/Control", "NoC/...", "#D9807A"),
]
DARK = {"pe", "filter_spad"}  # segments that need white inside labels
NOC_COLS = ["ifmapnoc", "weightsnoc", "psumwrnoc", "psumrdnoc"]


def theirs_values(row):
    """One row of the Accelergy paper breakdown, renormalized to 100."""
    p = pd.read_csv(
        os.path.join(VALIDATION, "accelergy_energy_breakdown.csv"),
        skipinitialspace=True,
        index_col="design",
    ).loc[row]
    raw = {
        "pe": p["pe_array"],
        "filter_spad": 0.0,  # inside pe_array in the paper data
        "glb": p["sharedbuffer"] + p["weightsbuffer"],
        "rest": p[NOC_COLS].sum(),
    }
    total = sum(raw.values())
    assert abs(total - p.sum()) < 1e-6, (total, p.sum())
    return {k: 100.0 * v / total for k, v in raw.items()}, total


def ours_values():
    # Every tag runs for the whole workload, so its share of power is its share of energy.
    df = pd.read_csv(os.path.join(RESULTS, "power.csv"), keep_default_na=False)
    onchip = df[df["module"] == ""].set_index("tag")["percent_of_onchip"]
    pe_module = df[(df["tag"] == "pe") & (df["module"] == "pe")]["percent_of_onchip"]
    pe_module = pe_module.item()
    # The pe tag row holds pe + filter_spad; plot them apart. Skip ifmap/filt/psum_noc (inside noc).
    assert abs(pe_module + onchip["filter_spad"] - onchip["pe"]) < 1e-6
    return {
        "pe": pe_module,
        "filter_spad": onchip["filter_spad"],
        "glb": onchip["glb"],
        "rest": onchip["noc"] + onchip["codec"],
    }


GT, GT_RAW = theirs_values("ground_truth")
ACC, ACC_RAW = theirs_values("accelergy")
OURS = ours_values()
BARS = [("Ground truth", GT), ("Accelergy", ACC), ("Ours", OURS)]
for _, vals in BARS:
    assert abs(sum(vals.values()) - 100.0) < 1e-6, sum(vals.values())


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
        if h == 0.0:
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


fig, ax = plt.subplots(figsize=(8.5, 4.5))
for xi, (_, vals), side in zip(X, BARS, (-1, +1, +1)):
    draw_bar(ax, xi, vals, side)
ax.set_xticks(X)
ax.set_xticklabels([name for name, _ in BARS])
ax.set_xlim(-1.2, 3.9)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel("Percent of on-chip energy (%)")
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
NOTE = "Ground truth / Accelergy PE array includes the filter spad; ours splits it out"
ax.text(0.5, -0.12, NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=8)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {os.path.normpath(OUT)}")
print(f"raw sums: ground_truth={GT_RAW:.2f} accelergy={ACC_RAW:.2f}")
for name, vals in BARS:
    print(name, " ".join(f"{k}={vals[k]:.2f}" for k, *_ in CATS))
