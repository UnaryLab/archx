"""Eyeriss fine-grain power breakdown for conv1 and conv5: Eyeriss paper (Fig. 16) vs our model.

Two groups (conv1, conv5) of two stacked bars (Eyeriss, Ours), with the same six categories and
colors as area_breakdown.py: spads, FIFOs, MAC, PE control, GLB, and one merged
NoC/ReLU/CODEC/Control segment. Merged segment: Eyeriss = noc + top-level control;
Ours = noc + codec.
Ours is read directly from the percent_of_onchip column of results/fg/<layer>/power.csv (tag
rows). Our model has no clock network or I/O pads, so the Eyeriss "clock network" and "i/o"
shares are excluded and the remaining Eyeriss categories are renormalized to 100% (divided by
their own sum, which equals total - clock network - i/o; the paper's conv5 shares are rounded
and total 99.8, not 100). The figure states the excluded shares in a note under the x axis.
Writes figures/fg/power_breakdown.png in the design directory.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "..", "..")
VALIDATION = os.path.join(DESIGN, "validation")
RESULTS = os.path.join(DESIGN, "results", "fg")
OUT = os.path.join(DESIGN, "figures", "fg", "power_breakdown.png")

LAYERS = ["conv1", "conv5"]
BAR_WIDTH = 0.45
GROUP_X = {"conv1": (0.0, 1.0), "conv5": (3.4, 4.4)}  # (Eyeriss, Ours) bar centers
INSIDE_MIN = 5.0  # segments at least this tall (percent) get their label inside the bar
LABEL_GAP = 5.0  # minimum vertical spacing (percent) between outside labels on one bar
LABEL_DX = 0.12  # horizontal offset of outside labels from the bar edge

# Categories bottom to top: (key, legend name, short name for outside labels, color).
CATS = [
    ("spad", "Spads", "Spads", "#08519C"),
    ("fifo", "FIFOs", "FIFOs", "#3182BD"),
    ("mac", "MAC", "MAC", "#6BAED6"),
    ("control", "PE control", "PE control", "#C6DBEF"),
    ("glb", "GLB", "GLB", "#F3B94D"),
    ("rest", "NoC/ReLU/CODEC/Control", "NoC/...", "#D9807A"),
]
DARK = {"spad", "fifo"}  # segments that need white inside labels
PE_PARTS = ["spad", "fifo", "mac", "control"]


def load(path, col):
    df = pd.read_csv(path, keep_default_na=False)
    return df[df["module"] == ""].set_index("tag")[col]


def eyeriss_csv(name):
    return pd.read_csv(os.path.join(VALIDATION, name), skipinitialspace=True).iloc[0]


def eyeriss_values(layer):
    """Paper Fig. 16 shares without clock network and I/O, renormalized to 100."""
    p = eyeriss_csv(f"eyeriss_{layer}_power_breakdown.csv")
    # Paper shares are rounded to 0.1; conv5 sums to 99.8, so allow rounding slack here.
    assert abs(p.sum() - 100.0) < 0.5, p.sum()
    raw = {
        "spad": p["spad"],
        "fifo": p["pe_fifo"],
        "mac": p["multiplier_adder"],
        "control": p["pe_control"],
        "glb": p["glb"],
        "rest": p["noc"] + p["top-level control"],
    }
    modeled = sum(raw.values())  # = total - clock network - i/o
    assert abs(p.sum() - p["clock network"] - p["i/o"] - modeled) < 1e-6
    return {k: 100.0 * v / modeled for k, v in raw.items()}, p


def ours_values(layer):
    onchip = load(os.path.join(RESULTS, layer, "power.csv"), "percent_of_onchip")
    # Skip ifmap/filt/psum_noc (inside noc) and filter_spad (inside spad and pe).
    assert abs(onchip[PE_PARTS].sum() - onchip["pe"]) < 1e-6
    assert abs(onchip[["pe", "glb", "noc", "codec"]].sum() - onchip["onchip"]) < 1e-6
    vals = {k: onchip[k] for k in PE_PARTS}
    vals.update(glb=onchip["glb"], rest=onchip["noc"] + onchip["codec"])
    return vals


EYERISS, PAPER, OURS = {}, {}, {}
for layer in LAYERS:
    EYERISS[layer], PAPER[layer] = eyeriss_values(layer)
    OURS[layer] = ours_values(layer)
    for vals in (EYERISS[layer], OURS[layer]):
        assert abs(sum(vals.values()) - 100.0) < 1e-6


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
    for key, _, short, color in CATS:
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


fig, ax = plt.subplots(figsize=(9, 4.5))
ticks, ticklabels = [], []
for layer in LAYERS:
    xe, xo = GROUP_X[layer]
    draw_bar(ax, xe, EYERISS[layer], side=-1)
    draw_bar(ax, xo, OURS[layer], side=+1)
    ticks += [xe, xo]
    ticklabels += ["Eyeriss", "Ours"]
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
trans = blended_transform_factory(ax.transData, ax.transAxes)
for layer in LAYERS:
    ax.text(
        np.mean(GROUP_X[layer]),
        -0.11,
        layer,
        transform=trans,
        ha="center",
        va="top",
        fontweight="bold",
    )
ax.set_xlim(-1.2, 5.6)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel("Percent of modeled power (%)")
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
clock = " / ".join(f"{PAPER[l]['clock network']:.1f}%" for l in LAYERS)
io = " / ".join(f"{PAPER[l]['i/o']:.1f}%" for l in LAYERS)
NOTE = (
    f"Eyeriss clock network ({clock}) and I/O ({io}) for {' / '.join(LAYERS)} "
    "excluded and rest renormalized (not modeled)"
)
ax.text(0.5, -0.2, NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=8)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {os.path.normpath(OUT)}")
for layer in LAYERS:
    for name, vals in (("Eyeriss", EYERISS[layer]), ("Ours", OURS[layer])):
        print(layer, name, " ".join(f"{k}={vals[k]:.2f}" for k, *_ in CATS))
