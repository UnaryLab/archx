import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

base_path = 'zoo/agraph/designs/sc_cnn_dse/'
paper_arch = '64x64x32'
peak_tmacs_per_pe = 50e9 / 16 / 1e12

workload_names = {'vgg16': 'VGG16', 'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50', 'googlenet': 'GoogLeNet', 'alexnet': 'AlexNet'}
# categorical slots in fixed order; a marker per model so identity isn't carried by color alone
colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
markers = ['o', 's', '^', 'D', 'v']
ink = '#0b0b0b'
muted = '#52514e'
grid = '#e1e0d9'

# column -> (short name, axis label, higher is better, tick format)
axes = {
    'area jj': ('Area', 'Area (JJ)', False, lambda v, _: f'{v / 1e9:g}G' if v >= 1e9 else f'{v / 1e6:g}M'),
    'throughput TMAC/s': ('Throughput', 'Throughput (TMAC/s)', True, lambda v, _: f'{v:g}'),
    'edp J*s': ('EDP', 'EDP (J·s)', False, None),
}

dse_df = pd.read_csv(f'{base_path}results/dse.csv')
dse_df['pe'] = dse_df['dim_x'] * dse_df['dim_y'] * dse_df['dim_z']
dse_df['edp J*s'] = dse_df['energy J'] * dse_df['latency s']
jj_per_pe = (dse_df['area jj'] / dse_df['pe']).median()
paper_area = dse_df.loc[dse_df['arch'] == paper_arch, 'area jj'].iloc[0]


def pareto(df, x, y, every_x=False):
    x_high, y_high = axes[x][2], axes[y][2]
    if x == 'area jj':
        # shapes with the same PE count differ by < 0.1% in area, so keep only the best one per PE count
        df = df.sort_values(y, ascending=not y_high).drop_duplicates('pe')
    if every_x:
        # the best shape at every x, past the optimum too, so the line spans the whole range
        return df.sort_values(x)
    # walk x from best to worst; keep a design only if it beats every design before it on y
    df = df.sort_values([x, y], ascending=[not x_high, not y_high])
    best = (df[y].cummax() if y_high else df[y].cummin()).shift()
    keep = best.isna() | (df[y] > best if y_high else df[y] < best)
    return df[keep].sort_values(x)


def plot_frontier(x, y, show_all=False, show_paper=True, every_x=False):
    name = f'pareto_{x.split()[0]}_{y.split()[0]}'
    fig, ax = plt.subplots(figsize=(8, 5.5))
    reference_handles = []
    if (x, y) == ('area jj', 'throughput TMAC/s'):
        area = np.geomspace(1e5, 1e10, 50)
        ax.plot(area, area / jj_per_pe * peak_tmacs_per_pe, color=muted, linestyle='--', linewidth=1, zorder=1)
        reference_handles.append(Line2D([], [], color=muted, linestyle='--', linewidth=1, label='peak (100% utilization)'))
        # clip the peak line to the data
        ax.set_xlim(3e5, 3e9)
        ax.set_ylim(4, 3000)
    if x == 'area jj' and show_paper:
        ax.axvline(paper_area, color=muted, linestyle=':', linewidth=1, zorder=1)

    fronts = []
    print(name)
    for i, (workload, model) in enumerate(workload_names.items()):
        df = dse_df[dse_df['workload'] == workload]
        front = pareto(df, x, y, every_x)
        fronts.append(front)
        if show_all:
            ax.scatter(df[x], df[y], color=colors[i], marker=markers[i], s=14, alpha=0.3, linewidths=0, zorder=2)
        ax.plot(front[x], front[y], color=colors[i], linewidth=2, marker=markers[i], markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=3)
        if show_paper:
            paper = df[df['arch'] == paper_arch].iloc[0]
            ax.plot(paper[x], paper[y], color=colors[i], marker=markers[i], markersize=8, markeredgecolor=ink, markeredgewidth=1.2, linestyle='none', zorder=4)
        print(f'  {workload:10s} {len(front):2d} frontier designs: {" ".join(front["arch"])}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    for column, axis in [(x, ax.xaxis), (y, ax.yaxis)]:
        _, label, _, fmt = axes[column]
        axis.set_label_text(label)
        axis.set_minor_formatter(NullFormatter())
        if column == 'throughput TMAC/s':
            axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        if fmt:
            axis.set_major_formatter(FuncFormatter(fmt))
    ax.grid(which='major', color=grid, linewidth=0.8)
    ax.set_axisbelow(True)

    model_handles = [Line2D([], [], color=c, linewidth=2, marker=m, markersize=6, markeredgecolor='white', label=n) for c, m, n in zip(colors, markers, workload_names.values())]
    if show_paper:
        reference_handles.insert(0, Line2D([], [], color=muted, marker='o', markersize=8, markerfacecolor='white', markeredgecolor=ink, markeredgewidth=1.2, linestyle=':' if x == 'area jj' else 'none', linewidth=1, label=f"{paper_arch.replace('x', '×')}, fixed (paper's mode A)"))
    # above the axes: models in one row, the reference marks centered in a row below it
    legend_y = 1.0
    legends = []
    if reference_handles:
        legends.append(ax.legend(handles=reference_handles, loc='lower center', bbox_to_anchor=(0.5, legend_y), ncol=len(reference_handles), frameon=False))
        ax.add_artist(legends[-1])
        legend_y += 0.06
    legends.append(ax.legend(handles=model_handles, loc='lower center', bbox_to_anchor=(0.5, legend_y), ncol=len(model_handles), frameon=False))
    fig.tight_layout()
    # one box around both legend rows
    fig.canvas.draw()
    box = Bbox.union([legend.get_window_extent() for legend in legends]).transformed(fig.transFigure.inverted())
    fig.patches.append(FancyBboxPatch(box.p0, box.width, box.height, boxstyle='round,pad=0,rounding_size=0.01', transform=fig.transFigure, facecolor='none', edgecolor='0.8', linewidth=1))
    fig.savefig(f'{base_path}figures/{name}{"_all" if show_all else ""}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # table view of the plotted frontiers
    columns = ['workload', 'arch', 'dim_x', 'dim_y', 'dim_z', 'area jj', 'leakage_power W', 'throughput TMAC/s', 'latency s', 'energy J', 'edp J*s', 'utilization']
    pd.concat(fronts)[columns].to_csv(f'{base_path}results/{name}.csv', index=False)


plot_frontier('area jj', 'throughput TMAC/s', every_x=True)
plot_frontier('area jj', 'throughput TMAC/s', show_all=True, show_paper=False, every_x=True)
plot_frontier('area jj', 'edp J*s', every_x=True)
plot_frontier('area jj', 'edp J*s', show_all=True, show_paper=False, every_x=True)
plot_frontier('throughput TMAC/s', 'edp J*s')
