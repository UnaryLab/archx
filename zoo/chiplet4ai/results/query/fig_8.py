from pathlib import Path
import math
import sys

import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


def warn(message):
    print(f'WARNING [fig_8]: {message}', file=sys.stderr)


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

RESULTS_PATH = 'zoo/chiplet4ai/results/csv/operator_latency_metrics.csv'
FIGURE_PATH = 'zoo/chiplet4ai/results/figs/fig_8.pdf'

# Full LaTeX textwidth (~480pt), two stacked panels sharing the model axis.
#
# Each panel scales its own log y-axis, for the reason fig_1 states: prefill and decode sit
# at completely different heights -- decode runs 129k steps against prefill's single pass,
# so the two phases are five to seven decades apart -- while each phase internally spans
# only three to four. One shared axis would squeeze every prefill bar into the bottom
# sliver and flatten the operator contrasts this figure is about.
FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 2.05
LEGEND_HEADROOM_IN = 0.78  # reserved at the top of the figure for the shared legend
NOTE_HEIGHT_IN = 0.20      # reserved at the bottom for the cost-model note
FIG_HEIGHT = 2 * PANEL_HEIGHT + LEGEND_HEADROOM_IN + NOTE_HEIGHT_IN

# The cost model these cycles are measured under, stated on the figure rather than left to
# the caption: without it a reader takes the bars for wall-clock latency.
COST_NOTE = ('Cost model: array-compute cycles only -- weight loading and operand '
             'streaming are free (perfect memory).')

# Labels match fig_1, fig_2 and fig_3 word for word, so the same model reads as the same
# model across the figure set.
model_styles = {
    'llama_3_1_8b': {'label': 'Llama 3.1 8B'},
    'llama_3_1_70b': {'label': 'Llama 3.1 70B'},
    'llama_3_1_405b': {'label': 'Llama 3.1 405B'},
    'deepseek_v4': {'label': 'DeepSeek V4 Pro 1.6T'},
}

# THE HIERARCHY, drawn twice over: hue family carries the operator CLASS, shade within the
# family carries the individual operator, and the classes are separated along x by a gap
# with the class total drawn behind them. So the top-level split (attention vs projection
# vs FFN) is readable at a glance and the per-operator drill-down is readable up close.
#
# The families are fig_1's model palettes reused -- blues, purples, greens -- so the figure
# set shares one set of inks. No red/green pairing, and the shades run dark to light in a
# fixed operator order, so position and lightness both identify a bar if hue is hard to
# read.
class_styles = {
    'projection': {
        'label': 'Projection',
        'operators': [('proj_q', 'Q proj', '#1045b8'),
                      ('proj_k', 'K proj', '#4d80dd'),
                      ('proj_v', 'V proj', '#41a9ee'),
                      ('a_proj', 'O proj', '#7bcaee')],
    },
    'attention': {
        'label': 'Attention',
        'operators': [('qkt', r'QK$^\mathsf{T}$', '#7d2fa0'),
                      ('av', 'AV', '#b57cd2')],
    },
    'ffn': {
        'label': 'FFN',
        'operators': [('gate_proj', 'Gate', '#1e7a34'),
                      ('up_proj', 'Up', '#3f9b53'),
                      ('down_proj', 'Down', '#66b878')],
    },
    'lm_head': {
        'label': 'LM head',
        'operators': [('lm_head', 'LM head', '#8c8c8c')],
    },
}

PANELS = [('prefill', 'Prefill'), ('decode', 'Decode')]

# Bar geometry, in units of one model slot. The inter-class gap is what turns ten adjacent
# bars into four legible groups.
GROUP_WIDTH = 0.86
CLASS_GAP_BARS = 0.35
# The class total sits behind its operators in a pale grey, so each group reads as a whole
# and its parts at the same time. It must stay clearly DARKER than the 'lightgrey' log
# minor gridlines behind it (#d3d3d3): at a closer grey the totals read as part of the
# grid's striping rather than as bars.
CLASS_TOTAL_COLOR = '#bfbfbf'

df = pd.read_csv(RESULTS_PATH)

# The query already scoped the CSV to one design point per model at 1000 MHz, so this
# figure applies no slice of its own. Report what survived, since a model whose design
# point stopped being generated would otherwise just go missing from the axis.
if df.empty:
    raise SystemExit(f'no rows in {RESULTS_PATH}')
for model_name, model_rows in sorted(df.groupby('model')):
    points = sorted(model_rows['array_dim'].unique())
    batches = sorted(model_rows['batch_size'].unique())
    contexts = sorted(model_rows['max_seq_len'].unique())
    print(f'  {model_name}: array {",".join(points)} batch {batches} seq {contexts}, '
          f'{len(model_rows)} operator rows')

models = [model for model in model_styles if model in set(df['model'])]
for model in sorted(set(df['model']) - set(model_styles)):
    warn(f'{model}: not in model_styles, dropped from the figure')

# Bar offsets within a model slot, laid out once: ten operators in class order, with a gap
# between classes. Also records each class's span so its total bar can be drawn behind it.
operator_order = [(class_name, operator, label, color)
                  for class_name, style in class_styles.items()
                  for operator, label, color in style['operators']]
bar_width = GROUP_WIDTH / (len(operator_order)
                           + CLASS_GAP_BARS * (len(class_styles) - 1))

offsets = {}
class_span = {}
cursor = -GROUP_WIDTH / 2
previous_class = None
for class_name, operator, _, _ in operator_order:
    if previous_class is not None and class_name != previous_class:
        cursor += CLASS_GAP_BARS * bar_width
    offsets[(class_name, operator)] = cursor + bar_width / 2
    span = class_span.setdefault(class_name, [cursor, cursor])
    span[1] = cursor + bar_width
    cursor += bar_width
    previous_class = class_name

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FIG_WIDTH, FIG_HEIGHT))

for ax, (phase, title) in zip(axes, PANELS):
    phase_df = df[df['phase'] == phase]
    if phase_df.empty:
        warn(f'{phase}: no rows; panel left empty')
        continue

    ax.grid(True, which='major', axis='y', color='lightgrey', linewidth=0.5, zorder=0)
    ax.grid(True, which='minor', axis='y', color='lightgrey', linewidth=0.3, zorder=0)

    for index, model in enumerate(models):
        model_df = phase_df[phase_df['model'] == model]
        if model_df.empty:
            warn(f'{model} {phase}: no rows')
            continue

        # The class total, behind its operators. Read from the query's own column rather
        # than re-summed here, so the figure cannot disagree with the CSV.
        for class_name, (left, right) in class_span.items():
            class_rows = model_df[model_df['operator_class'] == class_name]
            if class_rows.empty:
                continue
            total = class_rows['class_cycle_count'].iloc[0]
            if total <= 0:
                continue
            ax.bar(index + (left + right) / 2, total, width=right - left,
                   color=CLASS_TOTAL_COLOR, edgecolor='none', zorder=1)

        for class_name, operator, _, color in operator_order:
            row = model_df[(model_df['operator_class'] == class_name)
                           & (model_df['operator'] == operator)]
            if row.empty:
                warn(f'{model} {phase}: no {operator} row')
                continue
            value = row['cycle_count'].iloc[0]
            if value <= 0:
                continue
            ax.bar(index + offsets[(class_name, operator)], value, width=bar_width,
                   color=color, edgecolor='none', zorder=3)

    ax.set_yscale('log')
    # Panel-local limits with log-space padding, so neither the smallest operator nor the
    # class total is clipped by the frame.
    values = phase_df.loc[phase_df['cycle_count'] > 0, 'cycle_count']
    tops = phase_df.loc[phase_df['class_cycle_count'] > 0, 'class_cycle_count']
    if not values.empty:
        ax.set_ylim(values.min() / 4, max(values.max(), tops.max()) * 4)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, pos: f'$10^{{{int(round(math.log10(y)))}}}$' if y > 0 else ''))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis='y', pad=0.5)
    ax.tick_params(axis='y', which='minor', length=2)
    ax.set_title(title, loc='center', fontsize=8, pad=3)
    ax.set_axisbelow(True)

# The design point's array shape and batch belong ON the axis, not only in stdout: each
# model is reported on its own array at its own batch, so a cross-model reading of absolute
# latency is only honest if the reader can see which machine, and how much work per step,
# each set of bars came from. Read from the CSV's own columns, never hardcoded, so the
# label follows the design point if that moves.
def design_label(model):
    model_rows = df[df['model'] == model]
    values = []
    for column in ('array_dim', 'batch_size'):
        unique = sorted(model_rows[column].unique())
        if len(unique) != 1:
            warn(f'{model}: {len(unique)} {column} values in the CSV; labelling with the first')
        values.append(unique[0])
    return f'{values[0]}, batch {values[1]}'


axes[-1].set_xticks(range(len(models)))
axes[-1].set_xticklabels([f"{model_styles[model]['label']}\n{design_label(model)}"
                          for model in models])
axes[-1].set_xlim(-0.5, len(models) - 0.5)
fig.supylabel('Operator latency (array compute cycles)', fontsize=8, x=0.005)
fig.text(0.5, 0.012, COST_NOTE, ha='center', va='bottom', fontsize=6)

# ONE legend carrying both levels: a pale handle for the class total, then the ten operator
# handles in class order, so the key reads in the same order the bars are drawn.
handles = [Patch(facecolor=CLASS_TOTAL_COLOR, edgecolor='none', label='Class total')]
handles += [Patch(facecolor=color, edgecolor='none',
                  # a single-operator class would otherwise read 'LM head: LM head'
                  label=label if label == class_styles[class_name]['label']
                  else f"{class_styles[class_name]['label']}: {label}")
            for class_name, _, label, color in operator_order]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995),
           ncol=6, fontsize=6, columnspacing=1.0, handlelength=1.6,
           title='Operator', title_fontsize=6)

fig.tight_layout(rect=(0, NOTE_HEIGHT_IN / FIG_HEIGHT, 1,
                       1 - LEGEND_HEADROOM_IN / FIG_HEIGHT))

Path(FIGURE_PATH).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=1200)
print(f'wrote {FIGURE_PATH}')
