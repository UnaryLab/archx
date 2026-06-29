from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

FIG_WIDTH = 240 / 72.27
FIG_HEIGHT = FIG_WIDTH * .9

RESULTS_PATH = 'chiplet4ai/llama/results/bandwidth_performance_metrics.csv'
FIGURE_PATH = 'chiplet4ai/llama/figures/fig_2.png'


def bits_to_mib(bits):
    return bits / 8 / 2**20


df = pd.read_csv(RESULTS_PATH)

input_df = (
    df
    .groupby(['model', 'asram_size'], as_index=False)['asram_bandwidth']
    .mean()
    .sort_values(['model', 'asram_size'])
)
input_df['sram_size_mib'] = input_df['asram_size'].apply(bits_to_mib)

weight_df = (
    df
    .groupby(['model', 'wsram_size'], as_index=False)['wsram_bandwidth']
    .mean()
    .sort_values(['model', 'wsram_size'])
)
weight_df['sram_size_mib'] = weight_df['wsram_size'].apply(bits_to_mib)

model_styles = {
    'llama_3_8b': {'label': '8B', 'color': '#1045b8', 'marker': 'o'},
    'llama_3_70b': {'label': '70B', 'color': '#d67627', 'marker': 's'},
}

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.65, FIG_HEIGHT), sharex=True)

for model in sorted(input_df['model'].unique()):
    style = model_styles.get(model, {'label': model, 'color': None, 'marker': 'o'})
    sub = input_df[input_df['model'] == model]
    axes[0].plot(
        sub['sram_size_mib'],
        sub['asram_bandwidth'],
        linewidth=0.9,
        markersize=4,
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
    )

for model in sorted(weight_df['model'].unique()):
    style = model_styles.get(model, {'label': model, 'color': None, 'marker': 'o'})
    sub = weight_df[weight_df['model'] == model]
    axes[1].plot(
        sub['sram_size_mib'],
        sub['wsram_bandwidth'],
        linewidth=0.9,
        markersize=4,
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
    )

axes[0].set_title('Input SRAM')
axes[0].set_ylabel('Bandwidth (GiB/s)')
axes[0].set_xlabel('SRAM Size (MiB)')

axes[1].set_title('Weight SRAM')
axes[1].set_xlabel('SRAM Size (MiB)')

for ax in axes:
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.grid(True, which='major', linewidth=0.3, alpha=0.4)
    ax.margins(x=0.07)

axes[1].legend()
plt.tight_layout()

Path(FIGURE_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=300)
