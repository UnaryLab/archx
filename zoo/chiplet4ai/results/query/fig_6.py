"""fig_6: compute cycles across the array ASPECT-RATIO grid, at each model's max context.

fig_1 walks the DIAGONAL of the array design space -- 32x32, 64x64, ... 512x512 -- so every
point doubles the reduction depth and the output width together, and the figure can only
say how much array helps, never which array. fig_6 keeps both sides free and plots the same
5x5 grid TWICE, once per direction:

  left column   x = array columns (output width N), one line per array rows K
  right column  x = array rows (reduction depth K), one line per array columns N

Same numbers, two readings, and the point of drawing both is that a FLAT line names a
saturated dimension: it says that adding array along that axis buys nothing at that setting
of the other. A curve that keeps descending says the opposite. Reading the two columns
against each other is what separates the two dimensions, which a single diagonal cannot do.

Square arrays -- the slice fig_1 plots -- are marked with a ringed point on every curve, so
fig_6 can be lined up against fig_1 panel (c) directly.
"""

import math
import os

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# ONE BATCH, and why it has to be one. Both plot axes are already spent on the array's two
# dimensions, so batch cannot also be a series here the way it is in fig_1. 512 is the
# largest batch the sweep carries and the one that streams the most rows per weight load,
# which is the regime where the array's SHAPE -- rather than how starved it is -- decides
# the cycle count. fig_6_query writes every batch, so this is a one-line change.
BATCH_SIZE = 512

# Full LaTeX textwidth (~480pt). Four rows (one per model) x two columns (the two
# directions), so a panel is about 2.9in wide -- enough for five labelled log ticks.
FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 1.5
LEGEND_HEADROOM_IN = 0.62  # reserved at the top of the figure for the shared legend
FIG_HEIGHT = 4 * PANEL_HEIGHT + LEGEND_HEADROOM_IN

CSV_PATH = 'zoo/chiplet4ai/results/csv/array_shape_performance_metrics.csv'
FIG_OUT = 'zoo/chiplet4ai/results/figs/fig_6.pdf'

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')

# One entry per workload root event, with the labels and shade families fig_1 and fig_3
# use, so a model keeps the same colour across every figure in the set. Within a panel the
# shades run dark to light across the OTHER array dimension.
model_styles = {
    'llama_3_1_8b': {
        'label': 'Llama 3.1 8B',
        'context': '128K',
        'shades': ["#1045b8", "#4d80dd", "#41a9ee", "#7bcaee", "#a7dcec"],  # blues
    },
    'llama_3_1_70b': {
        'label': 'Llama 3.1 70B',
        'context': '128K',
        'shades': ["#d67627", "#e08e46", "#eea862", "#e0a975", "#e9c5a5"],  # oranges
    },
    'llama_3_1_405b': {
        'label': 'Llama 3.1 405B',
        'context': '128K',
        'shades': ["#1e7a34", "#3f9b53", "#66b878", "#92d0a0", "#bde4c6"],  # greens
    },
    'deepseek_v4': {
        'label': 'DeepSeek V4 Pro 1.6T',
        'context': '1M',
        'shades': ["#7d2fa0", "#9a55bb", "#b57cd2", "#cda4e3", "#e2c9f0"],  # purples
    },
}

dims = [32, 64, 128, 256, 512]

# LEGEND KEYS ARE GREY, not one model's blues. Every panel uses the same dark-to-light
# ordering over `dims`, so the legend is about that ORDERING and applies to all four
# models; drawing its keys in any one model's family would read as if it described that
# model alone.
legend_shades = ["#333333", "#5f5f5f", "#8a8a8a", "#b0b0b0", "#d2d2d2"]

views = [
    ('array_n', 'array_m', 'Array columns', 'Array rows', 'upper right'),
    ('array_m', 'array_n', 'Array rows', 'Array columns', 'upper left'),
]

# ONE FREQUENCY, and why the figure does not show the other. fig_6_query's CSV carries
# both 1000 and 2000 MHz, but the metric plotted here is `llama_array` -- the compute-only
# view -- which is exactly frequency-invariant, so the 2000 MHz series would be drawn on
# top of the 1000 MHz one. The frequency axis is real, but it lives in the CSV's
# `llama_cycle_count` and `runtime_ms` columns, not in these panels.
FREQUENCY_MHZ = 1000

df = pd.read_csv(CSV_PATH)
df = df[(df['batch_size'] == BATCH_SIZE) & (df['frequency'] == FREQUENCY_MHZ)]
if df.empty:
    raise SystemExit(f'fig_6: no rows at batch {BATCH_SIZE}, {FREQUENCY_MHZ} MHz '
                     f'in {CSV_PATH}')

# sharey='row': the two panels of a row are the SAME 25 numbers read two ways, so putting
# them on different y-axes would invite reading a difference that is not there.
fig, axes = plt.subplots(len(model_styles), len(views), sharex='col', sharey='row',
                         figsize=(FIG_WIDTH, FIG_HEIGHT))

for row, (model, style) in enumerate(model_styles.items()):
    sub = df[df['model'] == model]

    for col, (x_column, series_column, x_label, series_name, key_corner) in enumerate(views):
        ax = axes[row][col]
        ax.grid(True, color='lightgrey', linewidth=0.5, zorder=0)

        if sub.empty:
            ax.set_axis_off()
            continue

        for index, series_value in enumerate(dims):
            line = sub[sub[series_column] == series_value].sort_values(x_column)
            if line.empty:
                continue
            color = style['shades'][index % len(style['shades'])]
            ax.plot(line[x_column], line['cycle_count'], marker='o', color=color,
                    linewidth=0.7, markersize=2.2, zorder=3)

            # the square array on this curve -- fig_1's design point, for cross-reference
            square = line[line[x_column] == series_value]
            if not square.empty:
                ax.plot(square[x_column], square['cycle_count'], marker='o', color=color,
                        markersize=5, markerfacecolor='none', markeredgewidth=0.8,
                        linestyle='none', zorder=4)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator(dims))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(d) for d in dims]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        # WHOLE DECADES ONLY. The 2x and 5x subdivisions are still drawn as unlabelled
        # minor ticks below, so the scale is readable without the axis carrying three
        # labels per decade. Same treatment as fig_1.
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, pos: f'$10^{{{int(round(math.log10(y)))}}}$' if y > 0 else ''))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis='y', pad=0.5, labelsize=6)
        ax.tick_params(axis='y', which='minor', length=2)
        ax.margins(x=0.08)
        ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)

        # No column titles: the x-axis label on the bottom row already names the sweep,
        # and repeating a truncated form of it at the top only competes with the legend.
        # WHAT A LINE IS, stated once per column on the top row. The shared legend above
        # gives the five sizes but not which dimension they index, and that differs between
        # the columns -- rows on the left, columns on the right -- which is exactly the
        # thing a reader has to get right for the figure to mean anything. Borderless, and
        # each key sits in whichever corner its own column leaves empty: the left panel's
        # curves descend away from the top right, the right panel's rise away from the top
        # left.
        if row == 0:
            ax.legend([Line2D([], [], marker='o', color='0.35', linewidth=0.7,
                              markersize=2.2)],
                      [series_name], loc=key_corner, frameon=False,
                      fontsize=6, handlelength=1.6, handletextpad=0.5, borderpad=0.2)

        if row == len(model_styles) - 1:
            ax.set_xlabel(x_label)
        if col == 0:
            ax.set_ylabel(f"{style['label']}\n{style['context']} context", fontsize=7)

# ONE LEGEND, TWO MEANINGS. The five shades index the array's OTHER dimension, which is
# rows in the left column and columns in the right one -- the same five sizes either way,
# so a single ramp legend covers both and the column titles say which is which.
handles = [Line2D([], [], marker='o', color=shade, linewidth=0.7, markersize=2.2)
           for shade in legend_shades]
labels = [str(size) for size in dims]
handles = handles + [Line2D([], [], marker='o', color='0.35', markersize=5,
                            markerfacecolor='none', markeredgewidth=0.8, linestyle='none')]
labels = labels + ['square array']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .95),
           ncol=len(handles), fontsize=6, columnspacing=0.9, handlelength=1.6,
           title=f'batch {BATCH_SIZE}', title_fontsize=6)

fig.supylabel('Aggregate compute cycles', fontsize=8, x=0.005)

fig.tight_layout(rect=(0.01, 0, 1, 1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))
fig.savefig(FIG_OUT)
