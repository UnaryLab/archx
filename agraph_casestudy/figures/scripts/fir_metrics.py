from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd

fir_path_32 = 'agraph_casestudy/fir_sc_32/results/'
fir_path_256 = 'agraph_casestudy/fir_sc_256/results/'

# Split dataframes by arch column (256 and 32)
fir_area_256_df = pd.read_csv(fir_path_256 + 'area.csv')
fir_area_32_df = pd.read_csv(fir_path_32 + 'area.csv')
fir_throughput_256_df = pd.read_csv(fir_path_256 + 'throughput.csv')
fir_throughput_32_df = pd.read_csv(fir_path_32 + 'throughput.csv')

# Extract bitwidth, area, and throughput into lists for arch 256
bitwidth_256 = fir_area_256_df['bitwidth'].tolist()
area_256 = [a / 1000 for a in fir_area_256_df['area jj'].tolist()]
throughput_256 = fir_throughput_256_df['throughput GOPs'].tolist()

# Extract bitwidth, area, and throughput into lists for arch 32
bitwidth_32 = fir_area_32_df['bitwidth'].tolist()
area_32 = [a / 1000 for a in fir_area_32_df['area jj'].tolist()]
throughput_32 = fir_throughput_32_df['throughput GOPs'].tolist()

u256_color = "#4697bd"  # Blue (SM)
u32_color = "#e38a3c"     # Orange (PE) 

# Create figure with two subplots
width = 252 / 72
height = width / 2.7 # Adjusted for 1 row layout
wspace = 0.3  # Adjust spacing between subplots
fig, (ax_throughput, ax_area) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'wspace': wspace})

# Plot throughput on left axis
ax_throughput.plot(bitwidth_256, throughput_256, linewidth=0.5, label='U 256', color=u256_color)
ax_throughput.plot(bitwidth_32, throughput_32, linewidth=0.5, label='U 32', color=u32_color)
ax_throughput.set_xlabel('Bits', fontsize=8)
ax_throughput.set_title('Thr. [GOPs]', fontsize=8, pad=5)
ax_throughput.tick_params(axis='both', labelsize=8, pad=0.5, width=0.25, length=3, direction='in', which='major')
ax_throughput.tick_params(axis='both', which='minor', width=0.25, length=1.5, direction='in')
ax_throughput.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False, direction='in')
ax_throughput.set_xticks(bitwidth_256)
ax_throughput.set_xlim(bitwidth_256[0], bitwidth_256[-1])
ax_throughput.set_yscale('log')
ax_throughput.set_ylim(0.001, 1000)
ax_throughput.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax_throughput.set_yticklabels(['$10^{-3}$', '', '$10^{-1}$', '', '$10^{1}$', '', '$10^{3}$'])
ax_throughput.grid(axis='both', linestyle='--', linewidth=0.3, which='major')
ax_throughput.legend(fontsize=8, frameon=False)
for spine in ax_throughput.spines.values():
    spine.set_linewidth(0.5)

# Plot area on right axis
ax_area.plot(bitwidth_256, area_256, linewidth=0.5, label='U 256', color=u256_color)
ax_area.plot(bitwidth_32, area_32, linewidth=0.5, label='U 32', color=u32_color)
ax_area.set_xlabel('Bits', fontsize=8)
ax_area.set_title('JJs (× 10³)', fontsize=8, pad=5)
ax_area.tick_params(axis='both', labelsize=8, pad=0.5, width=0.25, length=3, direction='in', which='major')
ax_area.tick_params(axis='both', which='minor', width=0.25, length=1.5, direction='in')
ax_area.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False, direction='in')
ax_area.set_xticks(bitwidth_256)
ax_area.set_xlim(bitwidth_256[0], bitwidth_256[-1])
ax_area.set_yscale('log')
ax_area.set_ylim(1, 1000)
ax_area.set_yticks([1, 10, 100, 1000])
ax_area.grid(axis='both', linestyle='--', linewidth=0.3, which='major')
ax_area.legend(fontsize=8, frameon=False, loc='lower center')
for spine in ax_area.spines.values():
    spine.set_linewidth(0.5)

# Adjust layout
plt.tight_layout(pad=1.0)

# Save figure
plt.savefig('agraph_casestudy/figures/figures/fir_metrics_comparison.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('agraph_casestudy/figures/figures/fir_metrics_comparison.pdf', 
           dpi=300, bbox_inches='tight', facecolor='white')
