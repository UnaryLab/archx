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

# fig_1's layout: full LaTeX textwidth (~480pt), three stacked panels, one per
# max_seq_len slice, sharing the array-dimension x-axis.
#
# Unlike fig_1's cycle counts, utilization is a bounded fraction, so every panel
# uses the SAME linear 0-1 axis. Autoscaling each panel would let a design that
# never exceeds 0.3 fill its panel and read as well utilized, and the distance
# below 1 is the whole point of the figure. The cost is that the largest arrays
# crowd the bottom of the axis -- switch set_ylim to a log scale if that band is
# what you need to resolve.
FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 1.9
LEGEND_HEADROOM_IN = 0.55  # reserved at the top of the figure for the shared legend
FIG_HEIGHT = 3 * PANEL_HEIGHT + LEGEND_HEADROOM_IN

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')

# One panel per slice, in increasing context length. The third entry of each tuple
# restricts the panel to a subset of models; `None` means every model in model_styles.
#
# Only DeepSeek reaches 1048576, and its mixed-slice rows are the only ones that differ
# from the 128K panel above -- the Llama rows would just repeat panel (b) -- so that
# panel is restricted to DeepSeek.
panels = [
    ('array_utilization_metrics_seqlen_4096', '(a) 4K context (max_seq_len = 4096)', None),
    ('array_utilization_metrics_seqlen_131072', '(b) 128K context (max_seq_len = 131072)', None),
    ('array_utilization_metrics_seqlen_mixed',
     '(c) Long context (DeepSeek at 1048576)', ['deepseek_v4']),
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

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(FIG_WIDTH, FIG_HEIGHT))

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
            ax.plot(sub['array_dim_int'], sub['utilization'], marker='o', color=color, linewidth=0.5,
                    label=label, markersize=1.8, zorder=3)

    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y:.2f}'))
    ax.tick_params(axis='y', pad=0.5)
    ax.tick_params(axis='y', which='minor', length=2)
    ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)
    ax.margins(x=0.07)
    ax.set_title(title, loc='left', fontsize=8, pad=3)

axes[-1].set_xticks(valid_dims)
axes[-1].set_xticklabels([f'{d}x{d}' for d in valid_dims])
axes[-1].set_xlabel('Systolic-array dimensions')
fig.supylabel('Array utilization (useful MACs / PE-cycles)', fontsize=8, x=0.005)

# Shared legend above the figure. Every panel draws the same model/batch series,
# so the handles are collected from the first axes only.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=4, fontsize=6, columnspacing=0.8, handlelength=1.6,
           title='Workload', title_fontsize=6)

fig.tight_layout(rect=(0, 0, 1, 1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))
fig.savefig('zoo/chiplet4ai/results/figs/fig_3.pdf')
