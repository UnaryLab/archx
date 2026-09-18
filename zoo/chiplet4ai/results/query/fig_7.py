"""fig_7: decode tokens/s across the array ASPECT-RATIO grid, at each model's max context.

The same grid, panels and styling as fig_6, read from fig_6_query's CSV, as generated
tokens over generation time:

  tokens_per_s = batch_size * (max_seq_len - prefill_seq_len) * frequency_MHz * 1e6
                 / decode_cycle_count

`decode_cycle_count` is the array_compute lane over the DECODE GEMMs only; prefill time is
excluded. Same cost model as fig_6: perfect memory.

Usage: python fig_7.py [output.pdf]
"""

import math
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from chiplet4ai.results.query.utils import FIG_BATCH_SIZE

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# fig_6.py is a script (importing it redraws fig_6), so these are kept in step by hand:
# FREQUENCY_MHZ, dims and the shade lists match fig_6.py. The batch comes from utils.py.
BATCH_SIZE = FIG_BATCH_SIZE
FREQUENCY_MHZ = 1000

FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 1.5
LEGEND_HEADROOM_IN = 0.62
FIG_HEIGHT = 4 * PANEL_HEIGHT + LEGEND_HEADROOM_IN

CSV_PATH = 'zoo/chiplet4ai/results/csv/array_shape_performance_metrics.csv'
FIG_OUT = sys.argv[1] if len(sys.argv) > 1 else 'zoo/chiplet4ai/results/figs/fig_7.pdf'

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

model_styles = {
    'llama_3_1_8b': {
        'label': 'Llama 3.1 8B',
        'context': '128K',
        'shades': ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#86bfe0", "#a4cfe9", "#bcdcf0"],
    },
    'llama_3_1_70b': {
        'label': 'Llama 3.1 70B',
        'context': '128K',
        'shades': ["#7f2704", "#a63603", "#d94801", "#f16913", "#fd8d3c", "#fdac67", "#fdc28e", "#fdd4ae"],
    },
    'llama_3_1_405b': {
        'label': 'Llama 3.1 405B',
        'context': '128K',
        'shades': ["#00441b", "#006d2c", "#238b45", "#41ab5d", "#74c476", "#93d08f", "#aedfa8", "#c7e9c0"],
    },
    'deepseek_v4': {
        'label': 'DeepSeek V4 Pro 1.6T',
        'context': '1M',
        'shades': ["#3f007d", "#54278f", "#6a51a3", "#807dba", "#9e9ac8", "#b4b1d6", "#c7c4e2", "#dadaeb"],
    },
}

dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]

legend_shades = ["#000000", "#252525", "#424242", "#636363", "#858585", "#a3a3a3", "#bdbdbd", "#d4d4d4"]

views = [
    ('array_n', 'array_m', 'Array columns', 'Array rows', 'upper left'),
    ('array_m', 'array_n', 'Array rows', 'Array columns', 'upper left'),
]

df = pd.read_csv(CSV_PATH)
df = df[(df['batch_size'] == BATCH_SIZE) & (df['frequency'] == FREQUENCY_MHZ)].copy()
if df.empty:
    raise SystemExit(f'fig_7: no rows at batch {BATCH_SIZE}, {FREQUENCY_MHZ} MHz '
                     f'in {CSV_PATH}')
df['tokens_per_s'] = (df['batch_size'] * (df['max_seq_len'] - df['prefill_seq_len'])
                      * df['frequency'] * 1e6 / df['decode_cycle_count'])

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
            color = style['shades'][index]
            ax.plot(line[x_column], line['tokens_per_s'], marker='o', color=color,
                    linewidth=0.7, markersize=2.2, zorder=3)

            square = line[line[x_column] == series_value]
            if not square.empty:
                ax.plot(square[x_column], square['tokens_per_s'], marker='o', color=color,
                        markersize=5, markerfacecolor='none', markeredgewidth=0.8,
                        linestyle='none', zorder=4)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator(dims))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(d) for d in dims]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis='x', labelsize=6)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, pos: f'$10^{{{int(round(math.log10(y)))}}}$' if y > 0 else ''))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis='y', pad=0.5, labelsize=6)
        ax.tick_params(axis='y', which='minor', length=2)
        ax.margins(x=0.08)
        ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)

        if row == 0:
            ax.legend([Line2D([], [], marker='o', color='0.35', linewidth=0.7,
                              markersize=2.2)],
                      [series_name], loc=key_corner, frameon=False,
                      fontsize=6, handlelength=1.6, handletextpad=0.5, borderpad=0.2)

        if row == len(model_styles) - 1:
            ax.set_xlabel(x_label)
        if col == 0:
            ax.set_ylabel(f"{style['label']}\n{style['context']} context", fontsize=7)

handles = [Line2D([], [], marker='o', color=shade, linewidth=0.7, markersize=2.2)
           for shade in legend_shades]
labels = [str(size) for size in dims]
handles = handles + [Line2D([], [], marker='o', color='0.35', markersize=5,
                            markerfacecolor='none', markeredgewidth=0.8, linestyle='none')]
labels = labels + ['square array']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .95),
           ncol=len(handles), fontsize=6, columnspacing=0.9, handlelength=1.6,
           title=f'batch {BATCH_SIZE}', title_fontsize=6)

fig.supylabel('Decode tokens/s', fontsize=8, x=0.005)

fig.tight_layout(rect=(0.01, 0, 1,
                       1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))
fig.savefig(FIG_OUT)
