import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

df_8b = pd.read_csv('chiplet4ai/llama/results/array_performance_metrics_8b.csv')
df_70b = pd.read_csv('chiplet4ai/llama/results/array_performance_metrics_70b.csv')

# Filter for technology 16nm
df_8b = df_8b[df_8b['technology_nm'] == 16]
df_70b = df_70b[df_70b['technology_nm'] == 16]

# Filter for chiplet_size 32x32 and square array_dim
df_8b = df_8b[df_8b['chiplet_size'] == '32x32']
df_8b = df_8b[df_8b['array_dim'].apply(lambda x: x.split('x')[0] == x.split('x')[1])]
df_8b['array_dim_int'] = df_8b['array_dim'].apply(lambda x: int(x.split('x')[0]))

df_70b = df_70b[df_70b['chiplet_size'] == '32x32']
df_70b = df_70b[df_70b['array_dim'].apply(lambda x: x.split('x')[0] == x.split('x')[1])]
df_70b['array_dim_int'] = df_70b['array_dim'].apply(lambda x: int(x.split('x')[0]))

# Only keep array_dim in 32, 64, 128, 256, 512
valid_dims = [32, 64, 128, 256, 512]
df_8b = df_8b[df_8b['array_dim_int'].isin(valid_dims)]
df_70b = df_70b[df_70b['array_dim_int'].isin(valid_dims)]

# Sort for plotting
df_8b = df_8b.sort_values('array_dim_int')
df_70b = df_70b.sort_values('array_dim_int')

# Create evenly spaced x positions for plotting
x_positions = list(range(len(valid_dims)))

# Define color shades
blue_shades = ['#1f77b4', '#6baed6', '#c6dbef']  # dark to light blue
orange_shades = ['#d62728', '#ff7f0e', '#ffbb78']  # dark to light orange

# Plot for each batch_size in 8b (blue shades)
for i, batch_size in enumerate(sorted(df_8b['batch_size'].unique())):
    sub = df_8b[df_8b['batch_size'] == batch_size]
    sub_x = [valid_dims.index(d) for d in sub['array_dim_int']]
    color = blue_shades[i % len(blue_shades)]
    plt.plot(sub_x, sub['cycle_count'], marker='o', color=color, label=f'8B Batch {batch_size}')

# Plot for each batch_size in 70b (orange shades)
for i, batch_size in enumerate(sorted(df_70b['batch_size'].unique())):
    sub = df_70b[df_70b['batch_size'] == batch_size]
    sub_x = [valid_dims.index(d) for d in sub['array_dim_int']]
    color = orange_shades[i % len(orange_shades)]
    plt.plot(sub_x, sub['cycle_count'], marker='s', color=color, label=f'70B Batch {batch_size}')

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Array Dimension')
plt.ylabel('Clock Cycles')
plt.title('Clock Cycles vs Array Dimension')
plt.xticks(x_positions, [f'{d}x{d}' for d in valid_dims])
plt.yticks([10**i for i in range(11, 15)], [f'$10^{{{i}}}$' for i in range(11, 15)])
plt.ylim(1e11, 1e14)
plt.legend()
plt.tight_layout()
plt.savefig('chiplet4ai/llama/figures/fig_1.png')