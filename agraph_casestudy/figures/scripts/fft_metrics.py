from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd

fft_fg_path = 'agraph_casestudy/fft_fine_grain/results/'
fft_cg_path = 'agraph_casestudy/fft_course_grain/results/'

fft_fg_area_df = pd.read_csv(fft_fg_path + 'area.csv')
fft_cg_area_df = pd.read_csv(fft_cg_path + 'area.csv', index_col=0)
fft_fg_energy_df = pd.read_csv(fft_fg_path + 'energy.csv', index_col=0)
fft_cg_energy_df = pd.read_csv(fft_cg_path + 'energy.csv', index_col=0)
fft_fg_power_df = pd.read_csv(fft_fg_path + 'power.csv', index_col=0)
fft_cg_power_df = pd.read_csv(fft_cg_path + 'power.csv', index_col=0)

arch = fft_fg_area_df['arch'].tolist()
# Extract just the numbers from arch names (e.g., "8x8" -> "8")
arch_numbers = [arch_name.split('x')[0] for arch_name in arch]
fft_fg_area_error = fft_fg_area_df['percent_error'].tolist()
fft_fg_area_archx_norm = fft_fg_area_df['archx_norm'].tolist()
fft_fg_area_syn_norm = fft_fg_area_df['syn_norm'].tolist()
fft_cg_area_error = fft_cg_area_df['percent_error'].tolist()
fft_cg_area_archx_norm = fft_cg_area_df['archx_norm'].tolist()
fft_fg_energy_error = fft_fg_energy_df['percent_error'].to_list()
fft_fg_energy_archx_norm = fft_fg_energy_df['archx_norm'].to_list()
fft_fg_energy_syn_norm = fft_fg_energy_df['syn_norm'].to_list()
fft_cg_energy_error = fft_cg_energy_df['percent_error'].tolist()
fft_cg_energy_archx_norm = fft_cg_energy_df['archx_norm'].tolist()
fft_fg_power_error = fft_fg_power_df['percent_error'].to_list()
fft_fg_power_archx_norm = fft_fg_power_df['archx_norm'].to_list()
fft_fg_power_syn_norm = fft_fg_power_df['syn_norm'].to_list()
fft_cg_power_error = fft_cg_power_df['percent_error'].tolist()
fft_cg_power_archx_norm = fft_cg_power_df['archx_norm'].tolist()

# Normalized values for plotting (swapped Energy and Power order)
fg_norm_values = [fft_fg_area_archx_norm, fft_fg_power_archx_norm, fft_fg_energy_archx_norm]
syn_norm_values = [fft_fg_area_syn_norm, fft_fg_power_syn_norm, fft_fg_energy_syn_norm]
cg_norm_values = [fft_cg_area_archx_norm, fft_cg_power_archx_norm, fft_cg_energy_archx_norm]

# Percent error values for annotations (swapped Energy and Power order)
fg_error_values = [fft_fg_area_error, fft_fg_power_error, fft_fg_energy_error]
cg_error_values = [fft_cg_area_error, fft_cg_power_error, fft_cg_energy_error]

width = 252 / 72
height = width / 2.2# Adjusted for 1 row layout
rows = 1
cols = 3

# Color settings (hex values)
sm_color = "#4697bd"  # Blue (SM)
pe_color = "#e38a3c"     # Orange (PE) 
syn_color = "#3bac3b"    # Green (EDA baseline)

# Metric names for titles (swapped Energy and Power order)
metric_names = ['Area', 'L-Power', 'D-Energy']
grain_types = ['SM', 'PE', 'EDA']

# Create a single figure with one axis
fig, ax_main = plt.subplots(1, 1, figsize=(width, height))

# Set up basic formatting
ax_main.tick_params(axis='x', labelsize=8, pad=0.5)
ax_main.tick_params(axis='y', labelsize=8, pad=0.5)
ax_main.grid(axis='y', linestyle=(0, (2, 7)), linewidth=0.4)

# Format y-axis ticks
ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
ax_main.set_ylabel('Norm metric', fontsize=8, labelpad=2)
ax_main.set_ylim(0, 22.5)
ax_main.set_yticks([0, 6, 12, 18])

# Calculate x positions for all data points across three metrics
# Each metric gets 4 architecture sizes, with spacing between metrics
n_arch = len(arch)
spacing_between_metrics = 0  # Space between metric groups
x_offset = 0

# Plot data for each metric
for j in range(cols):
    # Calculate x positions for this metric
    x_positions = [x_offset + i for i in range(n_arch)]
    
    # Plot fine grain normalized data
    fg_norm_current = fg_norm_values[j]
    fg_error_current = fg_error_values[j]
    fg_label = 'SM' if j == 0 else None
    ax_main.plot(x_positions, fg_norm_current, color=sm_color, linewidth=0.8, alpha=0.7, label=fg_label)
    ax_main.scatter(x_positions, fg_norm_current, color=sm_color, s=15, alpha=1.0, zorder=5)
    
    # Plot coarse grain normalized data
    cg_norm_current = cg_norm_values[j]
    cg_error_current = cg_error_values[j]
    cg_label = 'PE' if j == 0 else None
    ax_main.plot(x_positions, cg_norm_current, color=pe_color, linewidth=0.8, alpha=0.7, label=cg_label, linestyle='--')
    ax_main.scatter(x_positions, cg_norm_current, color=pe_color, s=15, alpha=1.0, zorder=5, marker='^')
    
    # Plot pnr baseline
    syn_norm_current = syn_norm_values[j]
    syn_label = 'EDA' if j == 0 else None
    ax_main.plot(x_positions, syn_norm_current, color=syn_color, linewidth=0.8, alpha=0.7, label=syn_label, linestyle=':')
    ax_main.scatter(x_positions, syn_norm_current, color=syn_color, s=15, alpha=1.0, zorder=5, marker='s')
    
    # Add metric title at the center of this group
    title_x = x_offset + (n_arch - 1) / 2
    ax_main.text(title_x, ax_main.get_ylim()[1] * 1.01, metric_names[j], 
                 ha='center', va='bottom', fontsize=8)
    
    # Update x_offset for next metric
    x_offset += n_arch + spacing_between_metrics

# Define offsets for each point [metric][array_size][grain_type]
# Structure: offsets[j][k] = {'sm': (x_offset, y_offset), 'pe': (x_offset, y_offset)}
offsets = [
    # Area (j=0)
    [
        {'sm': (1, 6), 'pe': (2.5, 11)},    # 4x4
        {'sm': (0, 10), 'pe': (1.5, 15)},    # 8x8
        {'sm': (-5, 11), 'pe': (-3.5, 14)},    # 16x16
        {'sm': (-15, -0.5), 'pe': (-13.5, -2)},    # 32x32
    ],
    # Power (j=1) - swapped with Energy
    [
        {'sm': (-1, 6), 'pe': (0.5, 11)},    # 4x4
        {'sm': (-2, 10), 'pe': (-1.7, 14)},    # 8x8
        {'sm': (-8, 11.5), 'pe': (-6.5, 13)},    # 16x16
        {'sm': (-17, 3), 'pe': (-15.4, -2)},    # 32x32
    ],
    # Energy (j=2) - swapped with Power
    [
        {'sm': (-1, 7), 'pe': (0, 12)},    # 4x4
        {'sm': (0, 10), 'pe': (1, 12)},    # 8x8
        {'sm': (-8, 16), 'pe': (-7, 14)},    # 16x16
        {'sm': (-18, 14.5), 'pe': (-17, 2)},    # 32x32
    ]
]

# Add percent error annotations for all metrics
x_offset = 0
for j in range(cols):
    x_positions = [x_offset + i for i in range(n_arch)]
    fg_norm_current = fg_norm_values[j]
    fg_error_current = fg_error_values[j]
    cg_norm_current = cg_norm_values[j]
    cg_error_current = cg_error_values[j]
    
    # Add percent error annotations for fg (SM) data
    for k, (x, fg_norm, fg_err) in enumerate(zip(x_positions, fg_norm_current, fg_error_current)):
        xp, yp = offsets[j][k]['sm']
        ax_main.annotate(f'{fg_err:.1f}', (x, fg_norm), 
                        xytext=(xp, yp), textcoords='offset points',
                        fontsize=6, ha='center', color=sm_color)
    
    # Add percent error annotations for cg (PE) data
    for k, (x, cg_norm, cg_err) in enumerate(zip(x_positions, cg_norm_current, cg_error_current)):
        xp, yp = offsets[j][k]['pe']
        ax_main.annotate(f'{cg_err:.1f}', (x, cg_norm), 
                    xytext=(xp, yp), textcoords='offset points',
                    fontsize=6, ha='center', color=pe_color)
    
    x_offset += n_arch + spacing_between_metrics
    
# Set x-axis ticks and labels
all_x_positions = []
all_x_labels = []
x_offset = 0
for j in range(cols):
    for i in range(n_arch):
        all_x_positions.append(x_offset + i)
        all_x_labels.append(arch_numbers[i])
    x_offset += n_arch + spacing_between_metrics

ax_main.set_xticks(all_x_positions)
ax_main.set_xticklabels(all_x_labels, fontsize=8)

# Add a single legend for the entire figure with proper markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=sm_color, markersize=8, linewidth=0, label='PE-Sub'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=pe_color, markersize=8, linewidth=0, label='PE'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=syn_color, markersize=8, linewidth=0, label='EDA')
]
ax_main.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=8, 
               handletextpad=0.1, columnspacing=0.4)

# Adjust layout to prevent overlapping
plt.tight_layout(pad=1.0)

# Save the figure
plt.savefig('agraph_casestudy/figures/figures/fft_metrics_comparison.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('agraph_casestudy/figures/figures/fft_metrics_comparison.pdf', 
           dpi=300, bbox_inches='tight', facecolor='white')