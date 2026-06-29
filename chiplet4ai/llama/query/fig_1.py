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

df = pd.read_csv('chiplet4ai/llama/results/array_performance_metrics.csv')

df_8b = df[df['model'] == 'llama_3_8b'].copy()
df_70b = df[df['model'] == 'llama_3_70b'].copy()

# Filter for chiplet_size 32x32 and square array_dim
df_8b = df_8b[df_8b['array_dim'].apply(lambda x: x.split('x')[0] == x.split('x')[1])]
df_8b['array_dim_int'] = df_8b['array_dim'].apply(lambda x: int(x.split('x')[0]))

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
blue_shades = ["#1045b8", "#4d80dd", "#41a9ee", "#7bcaee", "#a7dcec"]  # dark to light blue
orange_shades = ["#d67627", "#e08e46", "#eea862", "#e0a975", "#e9c5a5"]  # dark to light orange

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Plot for each batch_size in 8b (blue shades)
for i, batch_size in enumerate(sorted(df_8b['batch_size'].unique())):
    sub = df_8b[df_8b['batch_size'] == batch_size]
    sub_x = [valid_dims.index(d) for d in sub['array_dim_int']]
    color = blue_shades[i % len(blue_shades)]
    label = f'8B Batch {batch_size}' if batch_size in (32, 512) else '_nolegend_'
    plt.plot(sub_x, sub['cycle_count'], marker='o', color=color, linewidth=0.9, label=label, markersize=5)

# Plot for each batch_size in 70b (orange shades)
for i, batch_size in enumerate(sorted(df_70b['batch_size'].unique())):
    sub = df_70b[df_70b['batch_size'] == batch_size]
    sub_x = [valid_dims.index(d) for d in sub['array_dim_int']]
    color = orange_shades[i % len(orange_shades)]
    label = f'70B Batch {batch_size}' if batch_size in (32, 512) else '_nolegend_'
    plt.plot(sub_x, sub['cycle_count'], marker='s', color=color, linewidth=0.9, label=label, markersize=5)

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Array Dimension')
plt.ylabel('Clock Cycles (log scale)')
plt.xticks(x_positions, [f'{d}x{d}' for d in valid_dims])
plt.tick_params(axis='y', pad=0.5)
plt.margins(x=0.07)
plt.yticks([10**11, 10**12, 10**13, 10**14, 10**15], ['11', '12', '13', '14', '15'])
plt.ylim(1e11, 10**15)
plt.legend()
plt.tight_layout()
plt.savefig('chiplet4ai/llama/figures/fig_1.png')
