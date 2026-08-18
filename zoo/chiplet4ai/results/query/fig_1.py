import math

import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# Full LaTeX textwidth (~480pt). Single panel: all 4 models share one log-scale
# y-axis. Each model's own cycle-count range spans ~1.8-1.95 decades and the
# four models together span ~3.15 decades (deepseek_v4 lowest to
# llama_3_1_405b highest), so the shared log axis still shows each curve's own
# slope clearly instead of flattening the smaller models.
FIG_WIDTH = 480 / 72.27
FIG_HEIGHT = 2.6
LEGEND_HEADROOM_IN = 0.55  # reserved at the top of the figure for the shared legend

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')

df = pd.read_csv('zoo/chiplet4ai/results/csv/array_performance_metrics.csv')

# One entry per workload root event; shades run dark to light across batch sizes.
model_styles = {
    'llama_3_1_8b': {
        'label': 'Llama 3.1 8B',
        'shades': ["#1045b8", "#4d80dd", "#41a9ee", "#7bcaee", "#a7dcec"],  # blues
    },
    'llama_3_1_70b': {
        'label': 'Llama 3.1 70B',
        'shades': ["#d67627", "#e08e46", "#eea862", "#e0a975", "#e9c5a5"],  # oranges
    },
    'llama_3_1_405b': {
        'label': 'Llama 3.1 405B',
        'shades': ["#1e7a34", "#3f9b53", "#66b878", "#92d0a0", "#bde4c6"],  # greens
    },
    'deepseek_v4': {
        'label': 'DeepSeek V4 Pro 1.6T',
        'shades': ["#7d2fa0", "#9a55bb", "#b57cd2", "#cda4e3", "#e2c9f0"],  # purples
    },
}

# Only keep square array_dim in 32, 64, 128, 256, 512
valid_dims = [32, 64, 128, 256, 512]
df = df[df['array_dim'].apply(lambda x: x.split('x')[0] == x.split('x')[1])].copy()
df['array_dim_int'] = df['array_dim'].apply(lambda x: int(x.split('x')[0]))
df = df[df['array_dim_int'].isin(valid_dims)]
df = df.sort_values('array_dim_int')

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax.grid(True, color='lightgrey', linewidth=0.5, zorder=0)

for model, style in model_styles.items():
    df_model = df[df['model'] == model]
    for i, batch_size in enumerate(sorted(df_model['batch_size'].unique())):
        sub = df_model[df_model['batch_size'] == batch_size]
        color = style['shades'][i % len(style['shades'])]
        label = f'{style["label"]} Batch {batch_size}' if batch_size in (32, 512) else '_nolegend_'
        ax.plot(sub['array_dim_int'], sub['cycle_count'], marker='o', color=color, linewidth=0.5,
                 label=label, markersize=1.8, zorder=3)

ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='y', pad=0.5)
ax.margins(x=0.07)
# Shared y-range across all 4 models (deepseek_v4 lowest to llama_3_1_405b
# highest data), with a modest log-space pad so extreme markers aren't
# clipped by the frame.
data_min = df['cycle_count'].min()
data_max = df['cycle_count'].max()
pad_factor = 1.3
ax.set_ylim(data_min / pad_factor, data_max * pad_factor)
# Major ticks/labels only at whole-decade powers of 10 that fall inside the
# (now tight) y-limits; minor sub-ticks (2-9x each decade) for scale context.
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda y, pos: f'$10^{{{int(round(math.log10(y)))}}}$' if y > 0 else ''))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(axis='y', which='minor', length=2)
ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)

ax.set_xticks(valid_dims)
ax.set_xticklabels([f'{d}x{d}' for d in valid_dims])
ax.set_xlabel('Systolic-array dimensions')
ax.set_ylabel('Aggregate compute cycles')

# Shared legend above the plot, collected from the single axes.
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=4, fontsize=6, columnspacing=0.8, handlelength=1.6,
           title='Workload', title_fontsize=6)

fig.tight_layout(rect=(0, 0, 1, 1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))
fig.savefig('zoo/chiplet4ai/results/figs/fig_1.pdf')
