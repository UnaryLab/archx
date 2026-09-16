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

# Full LaTeX textwidth (~480pt). Three stacked panels, one per max_seq_len slice
# fig_1_query writes, sharing the array-dimension x-axis.
#
# Each panel scales its own log y-axis. The three slices sit at very different
# heights (4096 starts near 2e11, the mixed slice reaches 4e17, 6.3 decades end
# to end) while each slice internally spans only ~3.2-3.7 decades, so one shared
# axis would leave every panel's data filling about half its height and flatten
# the array-size slopes this figure is about. Per-panel limits keep each curve's
# slope legible; the decade tick labels carry the cross-panel comparison.
FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 1.9
LEGEND_HEADROOM_IN = 0.55  # reserved at the top of the figure for the shared legend
NOTE_HEIGHT_IN = 0.20      # reserved at the bottom for the cost-model note
FIG_HEIGHT = 3 * PANEL_HEIGHT + LEGEND_HEADROOM_IN + NOTE_HEIGHT_IN

# The cost model these cycles are measured under, stated on the figure rather than left to
# the caption: without it a reader takes the curves for wall-clock latency.
COST_NOTE = ('Cost model: array-compute cycles only -- weight loading and operand '
             'streaming are free (perfect memory).')

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')

# One panel per slice, in increasing context length, with the models each panel
# draws. The mixed slice holds every model at its own long-context setting, but
# only DeepSeek reaches 1048576 -- the Llamas would repeat their 131072 rows from
# the panel above -- so that panel is restricted to DeepSeek. `None` means every
# model in model_styles.
panels = [
    ('array_performance_metrics_seqlen_4096', '4k Context', None),
    ('array_performance_metrics_seqlen_131072', '128k Context', None),
    ('array_performance_metrics_seqlen_mixed',
     '1M Context', ['deepseek_v4']),
]

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

def load_slice(name):
    df = pd.read_csv(f'zoo/chiplet4ai/results/csv/{name}.csv')
    df = df[df['array_dim'].apply(lambda x: x.split('x')[0] == x.split('x')[1])].copy()
    df['array_dim_int'] = df['array_dim'].apply(lambda x: int(x.split('x')[0]))
    df = df[df['array_dim_int'].isin(valid_dims)]
    return df.sort_values('array_dim_int')

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(FIG_WIDTH, FIG_HEIGHT))

for ax, (name, title, panel_models) in zip(axes, panels):
    df = load_slice(name)
    if panel_models is not None:
        df = df[df['model'].isin(panel_models)]
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
    # Panel-local y-range with a modest log-space pad so extreme markers aren't
    # clipped by the frame.
    pad_factor = 1.3
    ax.set_ylim(df['cycle_count'].min() / pad_factor, df['cycle_count'].max() * pad_factor)
    # Major ticks/labels only at whole-decade powers of 10 that fall inside the
    # (now tight) y-limits; minor sub-ticks (2-9x each decade) for scale context.
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, pos: f'$10^{{{int(round(math.log10(y)))}}}$' if y > 0 else ''))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis='y', which='minor', length=2)
    ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)
    ax.set_title(title, loc='center', fontsize=8, pad=3)

axes[-1].set_xticks(valid_dims)
axes[-1].set_xticklabels([f'{d}x{d}' for d in valid_dims])
axes[-1].set_xlabel('Systolic-array dimensions')
fig.supylabel('Array compute cycles', fontsize=8, x=0.005)
fig.text(0.5, 0.012, COST_NOTE, ha='center', va='bottom', fontsize=6)

# Shared legend above the figure. Every panel draws the same model/batch series,
# so the handles are collected from the first axes only.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=4, fontsize=6, columnspacing=0.8, handlelength=1.6,
           title='Workload', title_fontsize=6)

fig.tight_layout(rect=(0, NOTE_HEIGHT_IN / FIG_HEIGHT, 1,
                       1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))
fig.savefig('zoo/chiplet4ai/results/figs/fig_1.pdf')
