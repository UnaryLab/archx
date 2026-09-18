"""fig_5: stall-free array utilization against context length, one panel per model.

Weight loading and operand streaming are free here (perfect memory), so the compute lane
alone is the denominator and what is left is pure mapping efficiency: the tiling loss
from a reduction that does not fill the array's rows, an output that does not fill its
columns, and partial edge tiles.

Every point is the utilization AT that context, not aggregated up to it -- one decode
step at that sequence length, except the 2K point, which is the prefill of the prompt.
Batch is 512 throughout and the lines are square array shapes, so the figure reads as
"how well does this GEMM shape fill this array as the context grows".

Only DeepSeek is configured to 1048576; the Llama lines stop at their 131072 context.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from chiplet4ai.results.query.utils import CORE_ARRAY_SIZES, FIG_BATCH_SIZE

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# Full LaTeX textwidth (~480pt), four stacked panels sharing the context x-axis and the
# same linear 0-1 utilization axis.
FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 1.6
LEGEND_HEADROOM_IN = 0.5  # reserved at the top of the figure for the shared legend
FIG_HEIGHT = 4 * PANEL_HEIGHT + LEGEND_HEADROOM_IN

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')

# One panel per model, smallest first.
panels = [
    ('llama_3_1_8b', '(a) Llama 3.1 8B'),
    ('llama_3_1_70b', '(b) Llama 3.1 70B'),
    ('llama_3_1_405b', '(c) Llama 3.1 405B'),
    ('deepseek_v4', '(d) DeepSeek V4 Pro 1.6T'),
]

# Shades run light to dark with the array side, so the large arrays -- the ones whose
# utilization collapses once attention dominates -- are the ones that read darkest.
array_shades = {
    32: "#a7dcec",
    64: "#7bcaee",
    128: "#41a9ee",
    256: "#4d80dd",
    512: "#1045b8",
}

def context_label(tokens):
    """'128K' / '1M' for a token count, the form fig_1 and fig_6 label their panels with."""
    if tokens >= 2**20:
        return f'{tokens // 2**20}M'
    return f'{tokens // 2**10}K'

df = pd.read_csv('zoo/chiplet4ai/results/csv/array_utilization_nostall_metrics.csv')
df['array_dim_int'] = df['array_dim'].apply(lambda x: int(x.split('x')[0]))
df = df[df['array_dim_int'].isin(CORE_ARRAY_SIZES)]

# The contexts are decades apart, so they are placed evenly and labelled rather than
# drawn to scale: the figure compares shapes at four named operating points, and a log
# axis would crowd 2K against 4K and strand 1M at the far edge.
contexts = sorted(df['seq_len'].unique())
positions = {context: index for index, context in enumerate(contexts)}

fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(FIG_WIDTH, FIG_HEIGHT))

for ax, (model, title) in zip(axes, panels):
    df_model = df[df['model'] == model]
    ax.grid(True, color='lightgrey', linewidth=0.5, zorder=0)

    for array_size in CORE_ARRAY_SIZES:
        sub = df_model[df_model['array_dim_int'] == array_size].sort_values('seq_len')
        ax.plot([positions[context] for context in sub['seq_len']], sub['utilization'],
                marker='o', color=array_shades[array_size], linewidth=0.8,
                label=f'{array_size}x{array_size}', markersize=2.4, zorder=3)

    # A little headroom above 1: the small arrays tile perfectly and sit exactly at 1.0,
    # which the top spine would otherwise hide.
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y:.2f}'))
    ax.tick_params(axis='y', pad=0.5)
    ax.tick_params(axis='y', which='minor', length=2)
    ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)
    ax.margins(x=0.07)
    ax.set_title(title, loc='left', fontsize=8, pad=3)

axes[-1].set_xticks(list(positions.values()))
# The first point is the prefill and the rest are single decode steps; the axis says so
# once rather than leaving the reader to infer it from the shape of the curve.
axes[-1].set_xticklabels([f'{context_label(context)}\n(prefill)' if index == 0
                          else context_label(context)
                          for index, context in enumerate(contexts)])
axes[-1].set_xlabel(f'Context length at which the step runs (batch {FIG_BATCH_SIZE})')
# Say which cycles are in the denominator rather than leaving this looking like a
# system-level utilization; memory never enters it.
fig.supylabel('Array utilization (useful MACs / compute PE-cycles)', fontsize=8, x=0.005)

# Shared legend above the figure. Every panel draws the same array shapes, so the
# handles are collected from the first axes only.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=5, fontsize=6, columnspacing=0.8, handlelength=1.6,
           title='Systolic-array dimensions', title_fontsize=6)

fig.tight_layout(rect=(0, 0, 1, 1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))
fig.savefig('zoo/chiplet4ai/results/figs/fig_5.pdf')
