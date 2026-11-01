"""Eyeriss fine-grain area and power breakdown pies (structure of Figs. 15/16 in the Eyeriss paper).

Writes onchip_area.pdf, pe_area.pdf and conv1_power.pdf ... conv5_power.pdf to the design's figures/fg/.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results", "fg")
FIGURES = os.path.join(HERE, "..", "..", "figures", "fg")

NAMES = {
    "pe": "PE array",
    "spad": "Spads",
    "fifo": "PE FIFOs",
    "mac": "MAC",
    "control": "PE control",
    "glb": "GLB",
    "noc": "NoC",
    "codec": "Codec",
}
# Fixed slice order shared by every chart; each chart uses its subset in this order.
ORDER = list(NAMES)
TAB10 = plt.get_cmap("tab10").colors
COLORS = {cat: TAB10[i] for i, cat in enumerate(ORDER)}

LABEL_X = 1.32  # label text x (data units, pie radius = 1)
ELBOW_X = 1.25  # where the leader line turns horizontal
MIN_GAP = 0.36  # minimum vertical spacing between labels on one side
Y_LIMIT = 1.25  # labels are kept within [-Y_LIMIT, Y_LIMIT]


def load(path, col):
    df = pd.read_csv(path, keep_default_na=False)
    return df[df["module"] == ""].set_index("tag")[col]


def spread(ys):
    """Return label y positions (same order as ys) at least MIN_GAP apart, within Y_LIMIT."""
    idx = sorted(range(len(ys)), key=lambda i: -ys[i])
    pos = [min(ys[i], Y_LIMIT) for i in idx]
    for k in range(1, len(pos)):  # top-down: push labels down
        pos[k] = min(pos[k], pos[k - 1] - MIN_GAP)
    if pos and pos[-1] < -Y_LIMIT:  # bottom-up: push back up if we fell off
        pos[-1] = -Y_LIMIT
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = max(pos[k], pos[k + 1] + MIN_GAP)
    out = [0.0] * len(ys)
    for k, i in enumerate(idx):
        out[i] = pos[k]
    return out


def pie(slices, title, total_text, out_name):
    cats = [c for c in ORDER if c in slices]
    values = [slices[c] for c in cats]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    wedges, _ = ax.pie(
        values,
        colors=[COLORS[c] for c in cats],
        startangle=90,
        counterclock=False,
        radius=1.0,
        wedgeprops={"linewidth": 0},
    )

    mids = [math.radians((w.theta1 + w.theta2) / 2) for w in wedges]
    sides = {+1: [], -1: []}
    for i, a in enumerate(mids):
        sides[+1 if math.cos(a) >= 0 else -1].append(i)

    for side, members in sides.items():
        ys = spread([1.1 * math.sin(mids[i]) for i in members])
        for i, y in zip(members, ys):
            a = mids[i]
            x0, y0 = math.cos(a), math.sin(a)
            ax.plot(
                [x0, side * ELBOW_X, side * (LABEL_X - 0.03)],
                [y0, y, y],
                color="gray",
                linewidth=0.8,
            )
            ax.text(
                side * LABEL_X,
                y,
                f"{NAMES[cats[i]]}\n{100 * values[i] / total:.1f}%",
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=12,
                linespacing=1.2,
            )

    ax.set_title(f"{title}\n{total_text}", fontsize=14)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(FIGURES, out_name), bbox_inches="tight")
    plt.close(fig)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6)


os.makedirs(FIGURES, exist_ok=True)

# On-chip area
area = load(os.path.join(RESULTS, "area.csv"), "area mm^2")
s = {c: area[c] for c in ["pe", "glb", "noc", "codec"]}
assert close(sum(s.values()), area["onchip"]), (sum(s.values()), area["onchip"])
pie(s, "On-chip area", f"{sum(s.values()):.2f} mm²", "onchip_area.pdf")

# PE area
pe_area = load(os.path.join(RESULTS, "pe_area.csv"), "area mm^2")
s = {c: pe_area[c] for c in ["spad", "fifo", "mac", "control"]}
assert close(sum(s.values()), pe_area["pe"]), (sum(s.values()), pe_area["pe"])
pie(s, "PE area", f"{sum(s.values()):.2f} mm²", "pe_area.pdf")

# Per-layer power (codec excluded)
for n in range(1, 6):
    d = os.path.join(RESULTS, f"conv{n}")
    pe_pw = load(os.path.join(d, "pe_power.csv"), "total_power mW")
    pw = load(os.path.join(d, "power.csv"), "total_power mW")
    s = {c: pe_pw[c] for c in ["spad", "fifo", "mac", "control"]}
    s.update({c: pw[c] for c in ["glb", "noc"]})
    expect = pw["onchip"] - pw["codec"]
    assert close(sum(s.values()), expect), (n, sum(s.values()), expect)
    pie(s, f"conv{n} power", f"{sum(s.values()):.1f} mW", f"conv{n}_power.pdf")
