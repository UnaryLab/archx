import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

base_path = 'zoo/agraph/designs/sc_cnn_dse/'
paper_arch = '64x64x32'
peak_tmacs_per_pe = 50e9 / 16 / 1e12

workload_names = {'vgg16': 'VGG16', 'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50', 'googlenet': 'GoogLeNet', 'alexnet': 'AlexNet'}
# metric -> (column, scale, y label, unit)
metrics = {
    'power': ('power W', 1, 'Power', 'W'),
    'throughput': ('throughput TMAC/s', 1, 'Throughput', 'TMAC/s'),
    'latency': ('latency s', 1e6, 'Latency', 'µs'),
}
z_values = [16, 32, 64, 128, 256]
# Okabe-Ito colors, one marker per z
z_colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
z_markers = ['o', 's', '^', 'D', 'v']
nudge = 0.17

dse_df = pd.read_csv(f'{base_path}results/dse.csv')
dse_df['pe'] = dse_df['dim_x'] * dse_df['dim_y'] * dse_df['dim_z']
dse_df['log2_pe'] = np.log2(dse_df['pe'])

pe_exps = np.arange(12, 25)
pe_labels = [f'{2**e // 2**20}M' if e >= 20 else f'{2**e // 2**10}K' for e in pe_exps]

for workload, name in workload_names.items():
    df = dse_df[dse_df['workload'] == workload]
    # one folder per model; fig9_epochs.png covers all models and stays in figures/
    out_path = f'{base_path}figures/{workload}/'
    os.makedirs(out_path, exist_ok=True)
    for metric, (column, scale, ylabel, unit) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if metric == 'throughput':
            ax.plot(pe_exps, 2.0**pe_exps * peak_tmacs_per_pe, color='black', linestyle='--', linewidth=1, label='Peak (100% util.)', zorder=1)
        for i, z in enumerate(z_values):
            sub = df[df['dim_z'] == z]
            ax.scatter(sub['log2_pe'] + (i - 2) * nudge, sub[column] * scale, color=z_colors[i], marker=z_markers[i], s=22, alpha=0.85, label=f'TM z = {z}', zorder=2)
        paper = df[df['arch'] == paper_arch].iloc[0]
        px = paper['log2_pe'] + (z_values.index(paper['dim_z']) - 2) * nudge
        py = paper[column] * scale
        ax.scatter(px, py, s=160, facecolors='none', edgecolors='black', linewidths=1.5, label=f'paper {paper_arch}: {py:.3g} {unit}', zorder=3)
        ax.set_xticks(pe_exps)
        ax.set_xticklabels(pe_labels, rotation=45)
        ax.set_xlim(pe_exps[0] - 0.6, pe_exps[-1] + 0.6)
        ax.set_yscale('log')
        ax.set_xlabel('PE count x × y × z (log2 scale)')
        ax.set_ylabel(f'{ylabel} ({unit}), log scale')
        ax.set_title(f'{name}: {metric} vs. array size')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(f'{out_path}{metric}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{out_path}{metric}.pdf', bbox_inches='tight')
        plt.close(fig)
