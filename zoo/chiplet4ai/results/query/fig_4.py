from pathlib import Path
import sys

import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from chiplet4ai.results.query.utils import FIG_4_CRITERIA, fig_4_paths


def warn(message):
    print(f'WARNING [fig_4]: {message}', file=sys.stderr)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

FIG_WIDTH = 480 / 72.27
PANEL_HEIGHT = 2.3
CAPTION_HEADROOM_IN = 0.28  # the one-line criterion caption; the legends live inside the axes
FIG_HEIGHT = 2 * PANEL_HEIGHT + CAPTION_HEADROOM_IN


# The node axis is frequency and nothing else: the design space has no technology axis,
# so Sub-20nm and Sub-10nm are the SAME design point drawn twice. They are kept as
# separate bars to match the reference figure's layout, and they will be identical --
# that is a property of the sweep, not of the plot. Sub-5nm doubles the clock, and since
# cycle counts here are frequency-invariant it lands at exactly 2x: same bytes, half the
# time.
NODES = [
    ('Sub-20nm', 1000, '#8e44ad'),
    ('Sub-10nm', 1000, '#2e86c1'),
    ('Sub-5nm', 2000, '#17a589'),
]

# ONE SEQUENCE LENGTH PER MODEL -- each at its OWN maximum -- so each (model, node) is a
# single bar. This is the same 'mixed' slice fig_1 panel (c) and fig_6 report, so the three
# figures describe the same operating point.
#
# It is also the only context fig_4's design points exist at: three of the four are
# rectangular arrays, which description.py generates at max context alone.
MAX_SEQ_LEN = {
    'llama_3_1_8b': 131072,
    'llama_3_1_70b': 131072,
    'llama_3_1_405b': 131072,
    'deepseek_v4': 1048576,
}

MODEL_LABELS = {
    'llama_3_1_8b': 'Llama 3.1 8B',
    'llama_3_1_70b': 'Llama 3.1 70B',
    'llama_3_1_405b': 'Llama 3.1 405B',
    'deepseek_v4': 'DeepSeek V4 Pro 1.6T',
}

if not os.path.exists('zoo/chiplet4ai/results/figs'):
    os.makedirs('zoo/chiplet4ai/results/figs')

def render(criterion):
    """Draw one fig_4 variant: the design point each model reaches under `criterion`."""
    results_path, _, figure_path = fig_4_paths(criterion['key'])
    if not os.path.isfile(results_path):
        warn(f"[{criterion['key']}] {results_path} not found; figure skipped")
        return

    df = pd.read_csv(results_path)
    # Per model, not one shared length: a single `==` would silently plot DeepSeek at 131072
    # rather than at the 1048576 it is being reported for.
    df = df[df.apply(lambda row: row['max_seq_len'] == MAX_SEQ_LEN.get(row['model']), axis=1)]
    if df.empty:
        # warn rather than raise: one empty criterion must not stop the other two
        warn(f"[{criterion['key']}] no rows at the per-model maximum context in "
             f'{results_path}; figure skipped')
        return

    models = [model for model in MODEL_LABELS if model in set(df['model'])]
    for missing in set(df['model']) - set(MODEL_LABELS):
        warn(f'{missing}: no label configured, model dropped from the figure')

    # Array shape and batch are both fixed per model in this sweep (fig_4_query pins them to
    # that model's DESIGN_POINT, and neither varies across node or frequency), so one label per
    # model group is unambiguous. The warning fires if that ever stops being true, because a
    # single label over two design points would be a quietly wrong figure.
    def per_model_value(column):
        values = {}
        for model in models:
            distinct = df.loc[df['model'] == model, column].unique()
            if len(distinct) != 1:
                warn(f'{model}: {len(distinct)} distinct {column} values '
                     f'({sorted(distinct)}); labeling with the first')
            values[model] = distinct[0]
        return values

    array_shapes = per_model_value('array_dim')
    batch_sizes = per_model_value('batch_size')
    contexts = per_model_value('max_seq_len')

    def context_label(tokens):
        """'128K' / '1M' for a token count, the form fig_1 and fig_6 label their panels with."""
        if tokens >= 1 << 20 and tokens % (1 << 20) == 0:
            return f'{tokens >> 20}M'
        if tokens >= 1 << 10 and tokens % (1 << 10) == 0:
            return f'{tokens >> 10}K'
        return str(tokens)

    # The declared channel, drawn as the reference line. It never enters the bars: those are
    # a demand, computed with no knowledge of what the architecture provides.
    reference_gbs = df['dram_bandwidth'].unique()
    if len(reference_gbs) != 1:
        warn(f'configurations declare {len(reference_gbs)} different dram bandwidths '
             f'({sorted(reference_gbs)}); the reference line uses the first')
    reference_gbs = 600

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FIG_WIDTH, FIG_HEIGHT))
    bar_width = 0.8 / len(NODES)

    for ax, column, panel in ((axes[0], 'peak_bandwidth', '(a) Peak DRAM Bandwidth (GB/s)'),
                              (axes[1], 'average_bandwidth', '(b) Average DRAM Bandwidth (GB/s)')):
        for node_index, (node, frequency, color) in enumerate(NODES):
            heights, positions = [], []
            for model_index, model in enumerate(models):
                row = df[(df['model'] == model) & (df['frequency'] == frequency)]
                if row.empty:
                    warn(f'{model} at {frequency} MHz ({node}): no row, bar omitted')
                    continue
                if len(row) > 1:
                    warn(f'{model} at {frequency} MHz ({node}): {len(row)} rows, using the first')
                heights.append(float(row[column].iloc[0]))
                positions.append(model_index + (node_index - (len(NODES) - 1) / 2) * bar_width)

            bars = ax.bar(positions, heights, width=bar_width, color=color,
                          label=node if ax is axes[0] else '_nolegend_', zorder=3)
            ax.bar_label(bars, fmt='%.0f', fontsize=5, padding=1.5)

        # Keyed in panel (a) only: both panels draw the same line, and one legend serves
        # the whole figure.
        ax.axhline(reference_gbs, linestyle='--', linewidth=0.8, color='#444444', zorder=2,
                   label=f'{reference_gbs:g} GB/s reference' if ax is axes[0] else '_nolegend_')
        ax.set_ylabel(panel)
        ax.grid(True, axis='y', color='lightgrey', linewidth=0.5, zorder=0)
        # (a) stacks a legend row, the three-line design-point tag and the bar labels into
        # its headroom; (b) only needs room for its bar labels and a one-entry legend.
        # (a) needs enough of it that the tallest bar's value label clears the tag above
        # it: the tag sits at a fixed axes fraction, so more headroom lifts it away from
        # the bars rather than just padding the top.
        ax.margins(y=0.58 if ax is axes[0] else 0.24)
        ax.set_axisbelow(True)

        if ax is axes[0]:
            # THE DESIGN POINT per model group, in the headroom above the tallest bar:
            # array shape, batch, context. All three are part of the point rather than
            # properties of the model -- the bars would otherwise be read as a property of
            # 'Llama 8B' when they are specific to a shape, a batch AND a context -- and
            # all three are what change between the three fig_4 variants.
            for model_index, model in enumerate(models):
                label = (f'{array_shapes[model]}\n'
                         f'batch {batch_sizes[model]}\n'
                         f'{context_label(contexts[model])} context')
                ax.text(model_index, 0.86, label, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=6, style='italic', linespacing=1.3)

    axes[-1].set_xticks(range(len(models)))
    axes[-1].set_xticklabels([MODEL_LABELS[model] for model in models])

    # WHICH SELECTION THIS IS. The three variants are the same models on the same axes and
    # differ only in which design point was chosen, so without this they are near
    # indistinguishable -- and the design-point tags above each group are exactly what
    # changes between them.

    # TWO ROWS, BOTH CENTRED: the nodes across the top, the reference line centred
    # beneath them, frameless -- the layout of the reference figure.
    #
    # This is TWO legend artists, not one two-row legend. A single legend fills
    # column-major, so a fourth entry lands under the first column and left-aligns
    # against it; there is no way to centre a short trailing row within one legend.
    # Stacking two `upper center` legends centres each row independently. The first must
    # be re-added with add_artist, because ax.legend() replaces whatever legend the axes
    # already holds.
    #
    # Entries are picked out BY LABEL, not by position: get_legend_handles_labels walks
    # ax.lines before ax.containers, so the axhline comes back ahead of the bars even
    # though it was drawn after them, and the node bars are containers whose relative
    # order is not guaranteed either. Both are re-sorted here into NODES order.
    handles, labels = axes[0].get_legend_handles_labels()
    reference_label = f'{reference_gbs:g} GB/s reference'
    by_label = dict(zip(labels, handles))
    missing = [name for name, _, _ in NODES if name not in by_label]
    if reference_label not in by_label:
        missing.append(reference_label)
    if missing:
        warn(f'legend is missing entries for {missing}')

    node_entries = [(by_label[name], name) for name, _, _ in NODES if name in by_label]
    if node_entries:
        node_legend = axes[0].legend(
            [handle for handle, _ in node_entries], [label for _, label in node_entries],
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(node_entries),
            fontsize=6, columnspacing=1.8, handlelength=1.6, handletextpad=0.5,
            borderpad=0.2, frameon=False)
        axes[0].add_artist(node_legend)

    if reference_label in by_label:
        axes[0].legend([by_label[reference_label]], [reference_label],
                       loc='upper center', bbox_to_anchor=(0.5, 0.955), fontsize=6,
                       handlelength=2.4, handletextpad=0.5, borderpad=0.2, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 1 - CAPTION_HEADROOM_IN / FIG_HEIGHT))

    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    # release the figure: render() is called once per criterion and matplotlib keeps every
    # unclosed figure alive
    plt.close(fig)
    print(f'    wrote {figure_path}')


for criterion in FIG_4_CRITERIA:
    render(criterion)
