import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_path = 'zoo/agraph/designs/sc_cnn_dse/'

workloads = ['vgg16', 'resnet18', 'resnet50', 'googlenet', 'alexnet']
labels = ['VGG16', 'Resnet18', 'Resnet50', 'Googlenet', 'Alexnet']
arrays = [(8, 8, 512), (16, 16, 512), (64, 64, 32)]
# Okabe-Ito without black; cycles over layers
colors = ['#56B4E9', '#E69F00', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']


sys.path.insert(0, base_path)
from performance import layer_epochs
from loguru import logger

logger.remove()

keys = ['R', 'C', 'M', 'K_h', 'K_w', 'Z', 'H_in', 'W_in', 'stride']

epochs = {}
for w in workloads:
    with open(base_path + 'scale_sim_conv/' + w + '.json') as f:
        layers = json.load(f)['layers']
    for a in arrays:
        epochs[(w, a)] = [layer_epochs([l[k] for k in keys], *a) for l in layers]

dse = pd.read_csv(base_path + 'results/dse.csv')
dse = dse[dse['arch'] == '64x64x32'].set_index('workload')
print('64x64x32 total epochs: script vs dse.csv (latency s * 50e9 / 16)')
for w in workloads:
    ours = sum(epochs[(w, (64, 64, 32))])
    archx = dse.loc[w, 'latency s'] * 50e9 / 16
    print(f'{w:10s} {ours:>10d} {archx:>12.1f} {"match" if round(archx) == ours else "MISMATCH"}')
print('vgg16 hand values (8x8x512, 16x16x512, 64x64x32):', [sum(epochs[('vgg16', a)]) for a in arrays], 'expected [1504179, 398331, 498528]')
print('total epochs (millions):')
for w in workloads:
    print(f'{w:10s}', ' '.join(f'{sum(epochs[(w, a)]) / 1e6:.4f}' for a in arrays))

bar_w = 0.8
group_gap = 1.0
fig, ax = plt.subplots(figsize=(10, 5.5))
group_centers = []
for g, w in enumerate(workloads):
    xs = []
    for b, a in enumerate(arrays):
        x = g * (len(arrays) + group_gap) + b
        xs.append(x)
        bottom = 0
        for i, e in enumerate(epochs[(w, a)]):
            ax.bar(x, e / 1e6, bar_w, bottom=bottom, color=colors[i % len(colors)], edgecolor='black', linewidth=0.15)
            bottom += e / 1e6
        ax.text(x, -0.02, f'{a[0]}×{a[1]}×{a[2]}', rotation=45, ha='right', va='top', rotation_mode='anchor', transform=ax.get_xaxis_transform())
    group_centers.append(np.mean(xs))

ax.set_xticks(group_centers)
ax.set_xticklabels(labels)
ax.tick_params(axis='x', length=0, pad=70)
ax.set_xlim(-0.8, group_centers[-1] + 1.8)
ax.set_ylabel('Computing epochs (×10$^6$)\neach color is one layer, colors repeat')
ax.set_xlabel('Array size (TR×TC×TM) and benchmark', labelpad=8)
ax.grid(axis='y', linestyle=':', linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(base_path + 'figures/fig9_epochs.png', dpi=300, bbox_inches='tight')
