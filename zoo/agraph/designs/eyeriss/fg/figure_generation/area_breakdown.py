"""Eyeriss fine-grain area breakdown: Eyeriss paper (validation) vs our model, as stacked bars.

One pair of bars. The PE array is split into its parts (spads, FIFOs, MAC, PE control), stacked
under GLB and one merged NoC/ReLU/CODEC/Control segment, all as percent of on-chip area.
Merged segment: Eyeriss = noc + RLC_ReLU + control + config_scan_chain; Ours = noc + codec.
Ours is read directly from the percent_of_onchip column of results/fg/area.csv (tag rows).
Eyeriss PE parts are given as percent of PE in the paper data, so they are scaled by the
paper's pe_array share to get percent of on-chip area.
Writes figures/fg/area_breakdown.png in the design directory.
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
RESULTS = os.path.join(DESIGN, "results", "fg")
OUT = os.path.join(DESIGN, "figures", "fg", "area_breakdown.png")

BAR_WIDTH = 0.45
X = [0, 1]  # Eyeriss, Ours
INSIDE_MIN = 5.0  # segments at least this tall (percent) get their label inside the bar
LABEL_GAP = 5.0  # minimum vertical spacing (percent) between outside labels on one bar
LABEL_DX = 0.12  # horizontal offset of outside labels from the bar edge


def load(path, col):
    df = pd.read_csv(path, keep_default_na=False)
    return df[df["module"] == ""].set_index("tag")[col]


def eyeriss_csv(name):
    return pd.read_csv(os.path.join(VALIDATION, name), skipinitialspace=True).iloc[0]


# Categories bottom to top: (key, legend name, color). PE parts use a blue ramp so they read
# as one group; GLB is a light orange and the merged segment a muted light red.
CATS = [
    ("spad", "Spads", "#08519C"),
    ("fifo", "FIFOs", "#3182BD"),
    ("mac", "MAC", "#6BAED6"),
    ("control", "PE control", "#C6DBEF"),
    ("glb", "GLB", "#F3B94D"),
    ("rest", "NoC/ReLU/CODEC/Control", "#D9807A"),
]
DARK = {"spad", "fifo"}  # segments that need white inside labels
PE_PARTS = ["spad", "fifo", "mac", "control"]

# Eyeriss (paper numbers)
core = eyeriss_csv("eyeriss_area_core_breakdown.csv")
pe_core = eyeriss_csv("eyeriss_area_pe_breakdown.csv")
pe_core_pct = {
    "spad": pe_core["spad"],
    "fifo": pe_core["fifo"],
    "mac": pe_core["multiplier_adder"],
    "control": pe_core["pe_control"],
}
EYERISS = {k: core["pe_array"] * v / 100 for k, v in pe_core_pct.items()}
EYERISS.update(
    glb=core["glb"],
    rest=core["noc"] + core["RLC_ReLU"] + core["control"] + core["config_scan_chain"],
)
assert abs(sum(EYERISS[k] for k in PE_PARTS) - core["pe_array"]) < 1e-6

# Ours (simulated), read directly from percent_of_onchip
onchip = load(os.path.join(RESULTS, "area.csv"), "percent_of_onchip")
# Skip ifmap/filt/psum_noc (inside noc) and filter_spad (inside spad and pe).
assert abs(onchip[["pe", "glb", "noc", "codec"]].sum() - onchip["onchip"]) < 1e-6
OURS = {k: onchip[k] for k in PE_PARTS}
OURS.update(glb=onchip["glb"], rest=onchip["noc"] + onchip["codec"])
assert abs(sum(OURS[k] for k in PE_PARTS) - onchip["pe"]) < 1e-6
assert abs(sum(OURS.values()) - 100.0) < 1e-6
assert abs(sum(EYERISS.values()) - 100.0) < 1e-6


def outside_positions(centers):
    """Label y positions for small segments, bottom-up, at least LABEL_GAP apart."""
    pos = []
    for c in centers:
        pos.append(c if not pos else max(c, pos[-1] + LABEL_GAP))
    return pos


def draw_bar(ax, xi, values, side):
    """Stack one bar. side = -1 labels left, +1 labels right."""
    bottom = 0.0
    small = []
    for key, name, color in CATS:
        h = values[key]
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
            small.append((name, bottom + h / 2, h))
        bottom += h
    assert abs(bottom - 100.0) < 1e-6, bottom
    edge_x = xi + side * BAR_WIDTH / 2
    text_x = edge_x + side * LABEL_DX
    # place labels top-down so the topmost label stays at its segment and lower ones move down
    centers = [100.0 - s[1] for s in reversed(small)]
    ys = [100.0 - y for y in outside_positions(centers)]
    for (name, yc, h), yl in zip(reversed(small), ys):
        ax.plot(
            [edge_x, edge_x + side * LABEL_DX * 0.5, text_x - side * 0.01],
            [yc, yl, yl],
            color="#333333",
            linewidth=0.8,
        )
        ax.text(
            text_x,
            yl,
            f"{h:.1f}%",
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
    handles=[Patch(facecolor=c, label=n) for _, n, c in CATS],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    handlelength=2.0,
    handleheight=1.0,
    columnspacing=1.5,
)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {os.path.normpath(OUT)}")
