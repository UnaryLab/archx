from pathlib import Path
import sys

import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


# Full LaTeX textwidth, three stacked panels sharing the SRAM-capacity axis.
FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 1.55
LEGEND_HEADROOM_IN = 0.45  # reserved at the top for the shared legend
FIG_HEIGHT = 3 * PANEL_HEIGHT + LEGEND_HEADROOM_IN

RESULTS_PATH = 'zoo/chiplet4ai/results/csv/bandwidth_performance_metrics.csv'
FIGURE_PATH = 'zoo/chiplet4ai/results/figs/fig_2.pdf'

# The array and batch are per model and already applied by fig_2_query (DeepSeek on
# 256x256 at batch 256, the Llama models on 128x128 at batch 128), so this figure applies
# no design-point filter of its own.

# One size for every marker in the figure, legend included. The open circles carry the
# attention class, so the edge has to stay legible at this size.
MARKER_SIZE = 1.8
MARKER_EDGE_WIDTH = 0.5


def bits_to_mib(bits):
    return bits / 8 / 2**20


def pooled_bandwidth(group, bytes_column, label=''):
    """Pooled demand rate: total bytes over the total on-chip window they hide under.

    fig_2_query defines each row's rate as bytes / window_cycle_count, and reports that
    window as its own column. Pooling across the SRAM variants at one size is therefore a
    straight sum on both sides -- the same byte-weighted result as reconstructing each
    row's window from bytes/rate, but taken from the query's own number instead of
    inverted back out of it. An unweighted mean of per-row rates is not a rate over
    anything.
    """
    moved = group[bytes_column]
    window = group['window_cycle_count']
    usable = (moved > 0) & (window > 0)
    total_bytes = moved[usable].sum()
    total_cycles = window[usable].sum()
    if total_bytes <= 0 or total_cycles <= 0:
        # No usable sample here. A 0.0 would be plotted as a real measurement of "no
        # bandwidth demanded"; NaN leaves a visible gap, which is the honest mark.
        warn(f'{label or bytes_column}: no usable ({bytes_column} > 0, '
             f'window_cycle_count > 0) sample in group of {len(group)} rows; point skipped')
        return float('nan')
    seconds = total_cycles / (float(group['frequency'].iloc[0]) * 1e6)
    return (total_bytes / seconds) / 2**30


def aggregate(df, size_column, bytes_column, bandwidth_column, keys=('model',)):
    keys = list(keys)
    rows = [
        {**dict(zip(keys, group_keys if isinstance(group_keys, tuple) else (group_keys,))),
         size_column: size,
         bandwidth_column: pooled_bandwidth(group, bytes_column,
                                            label=f'{group_keys} {size_column}={size}')}
        for (*group_keys, size), group in df.groupby(keys + [size_column])
        for group_keys in [tuple(group_keys)]
    ]
    out = pd.DataFrame(rows).sort_values(keys + [size_column])
    out['sram_size_mib'] = out[size_column].apply(bits_to_mib)
    return out


df = pd.read_csv(RESULTS_PATH)

# The GB/s columns are per-frequency quantities (frequency is a swept axis, 1000/2000
# MHz); pooling the two halves would blend incompatible rates, so the figure shows the
# 1000 MHz reference slice. Traffic (bytes moved) is identical at both frequencies.
df = df[df['frequency'] == 1000]

# fig_2_query already scoped each model to its own (array, batch) design point -- DeepSeek
# on 256x256 at batch 256, the Llama models on 128x128 at batch 128 -- so every row here
# is already the point this figure reports and no batch filter is applied. Report what
# survived, since a model whose design point stopped being generated would otherwise just
# go missing from the legend.
if df.empty:
    raise SystemExit(f'no rows at frequency=1000 in {RESULTS_PATH}')
for model_name, model_rows in sorted(df.groupby('model')):
    points = sorted(model_rows['array_dim'].unique())
    batches = sorted(model_rows['batch_size'].unique())
    print(f'  {model_name}: array {",".join(points)} batch {batches}, '
          f'{model_rows["asram_size"].nunique()} SRAM sizes')

ARRAY_DIM = sorted(df['array_dim'].unique())
# The model double-buffers: only `active_fraction` of the declared capacity on the x-axis
# is credited with reuse.
ACTIVE_FRACTION = sorted(df['active_fraction'].unique())

# All three panels are the same slice, each against its own buffer's capacity, and each
# split by GEMM class. Attention and the dense layers respond to capacity in opposite
# ways -- the KV cache is streamed once per token and cannot be reused at any size, while
# projection and FFN traffic falls steeply -- so pooling them hides both.
KEYS = ('model', 'gemm_class')
input_df = aggregate(df, 'asram_size', 'input_data_moved', 'input_dram_bandwidth', keys=KEYS)
weight_df = aggregate(df, 'wsram_size', 'weight_data_moved', 'weight_dram_bandwidth', keys=KEYS)
# Combined output movement: the M*N finals, plus every K fold's partial tile wherever
# mapping.py found spilling cheaper than re-walking the weights. osram_size ==
# wsram_size on every swept config (description.py direct-constrains osram bank/depth to
# wsram's), so the x-axes of the weight and output panels coincide by construction.
output_df = aggregate(df, 'osram_size', 'output_data_moved', 'output_dram_bandwidth', keys=KEYS)

# Colour carries the model, line style the GEMM class. 'shades' is kept for a future
# multi-batch variant.
#
# SEPARATE AXES PER CLASS. The two classes differ by more than an order of magnitude on
# the IFMAP panel -- DeepSeek's attention lane alone pins the linear axis high enough to
# flatten every projection curve -- so each class gets its own linear scale: proj/FFN on
# the left spine, attention on the right. Both still start at zero, so the shape of each
# family is read against nothing rather than against the other's magnitude. The cost is
# that vertical position is no longer comparable ACROSS the two classes; only within one.
#
# The class is carried twice over, by the dash AND by whether the marker is filled, so a
# line is identifiable wherever either cue is hard to read -- a dashed segment between two
# close points, or an open circle at this size.
class_styles = {
    'proj/ffn': {'linestyle': '-', 'label': 'Proj/FFN (left)', 'axis': 'left',
                 'filled': True},
    'attention': {'linestyle': '--', 'label': 'Attention (right)', 'axis': 'right',
                  'filled': False},
}

# Labels match fig_1 and fig_3 word for word, so the same model reads as the same model
# across the figure set.
model_styles = {
    'llama_3_1_8b': {'label': 'Llama 3.1 8B', 'color': '#1045b8', 'marker': 'o',
                     'shades': ["#1045b8", "#4d80dd", "#41a9ee", "#7bcaee", "#a7dcec"]},
    'llama_3_1_70b': {'label': 'Llama 3.1 70B', 'color': '#d67627', 'marker': 's',
                      'shades': ["#d67627", "#e08e46", "#eea862", "#e0a975", "#e9c5a5"]},
    'llama_3_1_405b': {'label': 'Llama 3.1 405B', 'color': '#1e7a34', 'marker': '^',
                       'shades': ["#1e7a34", "#3f9b53", "#66b878", "#92d0a0", "#bde4c6"]},
    'deepseek_v4': {'label': 'DeepSeek V4 Pro 1.6T', 'color': '#7d2fa0', 'marker': 'D',
                    'shades': ["#7d2fa0", "#9a55bb", "#b57cd2", "#cda4e3", "#e2c9f0"]},
}

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(FIG_WIDTH, FIG_HEIGHT))

right_axes = []

PANELS = (
    (input_df, 'input_dram_bandwidth', 'IFMAP'),
    (weight_df, 'weight_dram_bandwidth', 'Parameters'),
    (output_df, 'output_dram_bandwidth', 'OFMAP'),
)

for ax, (frame, column, title) in zip(axes, PANELS):
    # The right-hand twin shares this panel's x-axis and draws only the attention class.
    # twinx hides the twin's own x-axis, so the shared tick labels stay on the left axes.
    right_ax = ax.twinx()
    class_axes = {'left': ax, 'right': right_ax}

    models = sorted(frame['model'].unique())
    for index, model in enumerate(models):
        style = model_styles.get(model, {'label': model, 'color': None, 'marker': 'o'})
        for gemm_class, class_style in class_styles.items():
            sub = frame[(frame['model'] == model) & (frame['gemm_class'] == gemm_class)]
            if sub.empty:
                warn(f'{model} {gemm_class}: no rows for the {title} panel')
                continue
            # ONE UNIFORM MARKER ON EVERY SAMPLED CAPACITY. Same circle, same size, for
            # every model and both classes, so the glyph carries no meaning of its own --
            # colour is the model, dash is the GEMM class, and a marker means "a run was
            # simulated here". Every one of the sweep's points gets one.
            class_axes[class_style['axis']].plot(
                sub['sram_size_mib'],
                sub[column],
                linewidth=0.5,
                markersize=MARKER_SIZE,
                marker='o',
                markevery=1,
                color=style['color'],
                markerfacecolor=style['color'] if class_style['filled'] else 'none',
                markeredgecolor=style['color'],
                markeredgewidth=MARKER_EDGE_WIDTH,
                linestyle=class_style['linestyle'],
                label='_nolegend_',
            )

    ax.set_title(title, loc='center', fontsize=8, pad=3)
    # Grid on the left axis only: two grids at different scales would draw two sets of
    # horizontal lines that mean different things.
    ax.grid(True, which='major', linewidth=0.3, alpha=0.4)
    ax.margins(x=0.07)
    right_ax.margins(x=0.07)

    # Linear y on both spines, each anchored at zero so every curve is read against
    # nothing rather than against a floating baseline, and each scaled to its OWN class's
    # maximum -- that separation is the whole point of the twin axis.
    for side, side_ax in class_axes.items():
        classes = [name for name, style in class_styles.items() if style['axis'] == side]
        usable = frame.loc[frame['gemm_class'].isin(classes), column].dropna()
        top = usable.max() if not usable.empty else 0
        if top > 0:
            side_ax.set_ylim(0, top * 1.08)
        else:
            # An empty side is the honest mark; say so rather than drawing a degenerate axis.
            warn(f'{title} ({side}): no usable (non-NaN, positive) bandwidth; axis left empty')

    # The dashes already say which family a line belongs to; tick colour says which spine
    # to read it against.
    right_ax.tick_params(axis='y', colors='#444444', labelsize=8)
    right_ax.spines['right'].set_color('#444444')
    right_axes.append(right_ax)

axes[-1].set_xscale('linear')
axes[-1].set_xlabel('SRAM capacity (MB)')
# EACH AXIS NAMES ITS OWN CLASS, and the line style that goes with it. The two axes are
# on wildly different scales -- attention's output lane is a fraction of a GB/s while
# proj/ffn's runs to four figures -- so a dashed line worth 0.2 GB/s draws at the TOP of a
# panel whose left axis reaches 1750. Read against the wrong axis it looks like the
# largest number in the figure. A single shared 'Required DRAM Bandwidth' label invited
# exactly that, so each side now states its class and its dash.
fig.supylabel('Proj/FFN required DRAM bandwidth (GB/s)', fontsize=8, x=0.005)

# ONE legend for both encodings: colour is the model, dash is the GEMM class. Proxy
# handles keep it independent of how many lines were actually drawn.
handles = [Line2D([], [], color=style['color'], marker='o',
                  markersize=MARKER_SIZE, linewidth=0.5, label=style['label'])
           for model, style in model_styles.items() if model in set(df['model'])]
# The class handles show BOTH cues together -- solid line + filled circle for Proj/FFN,
# dashed line + open circle for Attention -- so the key matches what is on the axes.
handles += [Line2D([], [], color='#444444', linewidth=0.5,
                   linestyle=style['linestyle'], marker='o', markersize=MARKER_SIZE,
                   markerfacecolor='#444444' if style['filled'] else 'none',
                   markeredgecolor='#444444', markeredgewidth=MARKER_EDGE_WIDTH,
                   label=style['label'])
            for style in class_styles.values()]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, .95),
           ncol=len(handles), fontsize=6, columnspacing=1.0, handlelength=1.8)

# The right edge stops short of 1.0 to leave a lane for the attention axis label below;
# without it the label would sit hard against the right-hand tick numbers.
fig.tight_layout(rect=(0, 0, 0.985, 1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))

# The right-hand counterpart, added AFTER tight_layout so it can be centred on the axes
# stack's real extent rather than on the whole figure (which the legend strip skews).
# Coloured to the right spine so the label and the ticks it belongs to read as one thing.
stack_top = max(ax.get_position().y1 for ax in axes)
stack_bottom = min(ax.get_position().y0 for ax in axes)
fig.text(1.0, (stack_top + stack_bottom) / 2,
         'Attention required DRAM bandwidth (GB/s)',
         fontsize=8, color='#444444', rotation=270, va='center', ha='right')

Path(FIGURE_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=1200)
print(f'wrote {FIGURE_PATH}')
