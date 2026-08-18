from pathlib import Path
import sys

import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def warn(message):
    print(f'WARNING [fig_2]: {message}', file=sys.stderr)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')


FIG_WIDTH = 240 / 72.27
FIG_HEIGHT = FIG_WIDTH * .9

RESULTS_PATH = 'zoo/chiplet4ai/results/csv/bandwidth_performance_metrics.csv'
FIGURE_PATH = 'zoo/chiplet4ai/results/figs/fig_2.png'


def bits_to_mib(bits):
    return bits / 8 / 2**20


def byte_weighted_bandwidth(group, bytes_column, bandwidth_column, label=''):
    """Byte-weighted harmonic mean, sum(D) / sum(D / BW).

    This is the same aggregation fig_2_query applies within a row, so aggregating across
    the SRAM variants at one size stays a demand rate over the pooled bytes. An unweighted
    arithmetic mean of already-harmonic-meaned rates is not a rate over anything.
    """
    moved = group[bytes_column]
    bandwidth = group[bandwidth_column]
    usable = (moved > 0) & (bandwidth > 0)
    total_bytes = moved[usable].sum()
    total_time = (moved[usable] / bandwidth[usable]).sum()
    if total_bytes <= 0 or total_time <= 0:
        # No usable sample here. A 0.0 would be plotted as a real measurement of "no
        # bandwidth demanded"; NaN leaves a visible gap, which is the honest mark.
        warn(f'{label or bandwidth_column}: no usable ({bytes_column} > 0, '
             f'{bandwidth_column} > 0) sample in group of {len(group)} rows; point skipped')
        return float('nan')
    return total_bytes / total_time


def aggregate(df, size_column, bytes_column, bandwidth_column, keys=('model',)):
    keys = list(keys)
    rows = [
        {**dict(zip(keys, group_keys if isinstance(group_keys, tuple) else (group_keys,))),
         size_column: size,
         bandwidth_column: byte_weighted_bandwidth(group, bytes_column, bandwidth_column,
                                                   label=f'{group_keys} {size_column}={size}')}
        for (*group_keys, size), group in df.groupby(keys + [size_column])
        for group_keys in [tuple(group_keys)]
    ]
    out = pd.DataFrame(rows).sort_values(keys + [size_column])
    out['sram_size_mib'] = out[size_column].apply(bits_to_mib)
    return out


df = pd.read_csv(RESULTS_PATH)

# The GiB/s columns are per-frequency quantities (frequency is a swept axis, 1000/2000
# MHz); pooling the two halves would blend incompatible rates, so the figure shows the
# 1000 MHz reference slice. Traffic (bytes moved) is identical at both frequencies.
df = df[df['frequency'] == 1000]

# The remaining slice covers one array size (512x512) at 1000 MHz, all five batch sizes:
# 2,000 rows of the 4,000-row CSV. See the SWEEP FILTER comment in fig_2_query.py.
ARRAY_DIM = sorted(df['array_dim'].unique())
BATCH_SIZE = sorted(df['batch_size'].unique())
# The model double-buffers: only `active_fraction` of the declared capacity on the x-axis
# is credited with reuse.
ACTIVE_FRACTION = sorted(df['active_fraction'].unique())

# INPUT panel: per (model, batch) series. The all-or-nothing input law steps at
# SRAM = 2x working set, and the working sets scale with batch, so the staircase only
# shows when batches are kept apart -- aggregating across batches averages the cliffs
# into a slope.
input_df = aggregate(df, 'asram_size', 'input_data_moved', 'input_dram_bandwidth',
                     keys=('model', 'batch_size'))

# WEIGHT / OUTPUT panels: batch-512 slice, one series per model, preserving these
# panels' original story. Both are flat by construction (weights fetched once; output is
# write-through, independent of osram size), so per-batch families would add lines
# without adding signal.
df_512 = df[df['batch_size'] == 512]
weight_df = aggregate(df_512, 'wsram_size', 'weight_data_moved', 'weight_dram_bandwidth')
# Combined output movement (write-through: writes only, read stream structurally 0; see
# fig_2_query.py). osram_size == wsram_size on every swept config (description.py
# direct-constrains osram bank/depth to wsram's), so the x-axes of the weight and output
# panels coincide by construction.
output_df = aggregate(df_512, 'osram_size', 'output_data_moved', 'output_dram_bandwidth')

# 'color' is the model's line in the single-series panels; 'shades' (dark to light,
# fig_1's families) color the per-batch input-panel series.
model_styles = {
    'llama_3_1_8b': {'label': '8B', 'color': '#1045b8', 'marker': 'o',
                     'shades': ["#1045b8", "#4d80dd", "#41a9ee", "#7bcaee", "#a7dcec"]},
    'llama_3_1_70b': {'label': '70B', 'color': '#d67627', 'marker': 's',
                      'shades': ["#d67627", "#e08e46", "#eea862", "#e0a975", "#e9c5a5"]},
    'llama_3_1_405b': {'label': '405B', 'color': '#1e7a34', 'marker': '^',
                       'shades': ["#1e7a34", "#3f9b53", "#66b878", "#92d0a0", "#bde4c6"]},
    'deepseek_v4': {'label': 'DSv4', 'color': '#7d2fa0', 'marker': 'D',
                    'shades': ["#7d2fa0", "#9a55bb", "#b57cd2", "#cda4e3", "#e2c9f0"]},
}

fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH * 2.45, FIG_HEIGHT))

for model in sorted(input_df['model'].unique()):
    style = model_styles.get(model, {'label': model, 'color': None, 'marker': 'o',
                                     'shades': [None] * 5})
    sub_model = input_df[input_df['model'] == model]
    for i, batch in enumerate(sorted(sub_model['batch_size'].unique())):
        sub = sub_model[sub_model['batch_size'] == batch]
        # fig_1's legend idiom: label only the darkest and lightest batch per model.
        label = f'{style["label"]} B{batch}' if batch in (32, 512) else '_nolegend_'
        axes[0].plot(
            sub['sram_size_mib'],
            sub['input_dram_bandwidth'],
            linewidth=0.9,
            markersize=3,
            marker=style['marker'],
            color=style['shades'][i % len(style['shades'])],
            label=label,
        )

for model in sorted(weight_df['model'].unique()):
    style = model_styles.get(model, {'label': model, 'color': None, 'marker': 'o'})
    sub = weight_df[weight_df['model'] == model]
    axes[1].plot(
        sub['sram_size_mib'],
        sub['weight_dram_bandwidth'],
        linewidth=0.9,
        markersize=4,
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
    )

for model in sorted(output_df['model'].unique()):
    style = model_styles.get(model, {'label': model, 'color': None, 'marker': 'o'})
    sub = output_df[output_df['model'] == model]
    axes[2].plot(
        sub['sram_size_mib'],
        sub['output_dram_bandwidth'],
        linewidth=0.9,
        markersize=4,
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
    )

fraction_note = ', '.join(f'{value:g}' for value in ACTIVE_FRACTION)
x_label = f'Declared SRAM Size (MiB)\n(active fraction {fraction_note})'
y_label = 'Avg DRAM Demand Rate (GiB/s)'

axes[0].set_title('Input Movement')
axes[0].set_ylabel(y_label)
axes[0].set_xlabel(x_label)
axes[0].legend(ncol=2, fontsize=5, columnspacing=0.6, handlelength=1.2)

axes[1].set_title('Weight Movement')
axes[1].set_xlabel(x_label)
axes[1].set_ylabel(y_label)

axes[2].set_title('Output Movement')
axes[2].set_xlabel(x_label)
axes[2].set_ylabel(y_label)

def set_top_ylim(ax, series, label):
    """set_ylim(0, max) is the one place the NaN contract is unhandled.

    An all-NaN (or empty) panel makes `.max()` NaN, and set_ylim(0, nan) is a degenerate
    axis. Leave matplotlib's autoscale alone in that case - an empty panel is the honest
    mark - and say so.
    """
    top = series.max()
    if not pd.notna(top) or top <= 0:
        warn(f'{label}: no usable (non-NaN, positive) bandwidth in the panel; '
             f'leaving the y-axis on autoscale')
        return
    ax.set_ylim(0, top)


set_top_ylim(axes[0], input_df['input_dram_bandwidth'], 'input panel')
set_top_ylim(axes[1], weight_df['weight_dram_bandwidth'], 'weight panel')
set_top_ylim(axes[2], output_df['output_dram_bandwidth'], 'output panel')

for ax in axes:
    ax.set_xscale('linear')
    ax.grid(True, which='major', linewidth=0.3, alpha=0.4)
    ax.margins(x=0.07)

axes[2].legend()
plt.tight_layout(rect=(0, 0.09, 1, 1))

CAPTION = (
    f'Average DRAM demand rate (bytes moved / demand window), byte-weighted across configs. '
    f'Array {"/".join(ARRAY_DIM)} at 1000 MHz only: 2,000 of 4,960 swept configurations '
    f'(the 2000 MHz half of the sweep is in the CSV, not plotted). The input panel '
    f'shows one series per (model, batch {"/".join(str(b) for b in BATCH_SIZE)}), dark to '
    f'light: the all-or-nothing input law (SCALE-Sim reconciled) steps down where the '
    f'declared SRAM reaches 2x a GEMM working set, since only '
    f'active_fraction={fraction_note} of the capacity holds reused data (the other half is '
    f'the prefetch buffer). The weight and output panels are the batch-512 slice, flat by '
    f'construction: weight-stationary fetches every weight exactly once, and output is '
    f'write-through (every partial-sum update drains to DRAM, independent of osram size; '
    f'spill re-reads are structurally zero). osram bank/depth are constrained equal to '
    f'wsram, so the output panel x-axis coincides with the weight panel.'
)
fig.text(0.01, 0.005, CAPTION, fontsize=5, va='bottom', ha='left', wrap=True)

Path(FIGURE_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=300)
print(f'wrote {FIGURE_PATH}')
print(CAPTION)
