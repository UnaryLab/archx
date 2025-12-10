import os
import re
import pandas as pd

runtime_path = 'agraph_casestudy/runtime/'

# Dictionary to store runtime data
runtime_data = {}

# Loop through each text file in the runtime directory
for filename in os.listdir(runtime_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(runtime_path, filename)
        
        # Extract case study name (filename without .txt extension)
        cs_name = filename.replace('.txt', '')
        
        # Read the file and extract runtime value
        with open(filepath, 'r') as f:
            content = f.read().strip()
            # Extract the numeric value from "Total runtime: X seconds"
            match = re.search(r'Total runtime: ([\d.]+) seconds', content)
            if match:
                runtime_seconds = float(match.group(1))
                runtime_data[cs_name] = runtime_seconds

runtime_df = pd.read_csv(runtime_path + 'eda_runtime_csv.csv')

tnn_runtime = runtime_data.get('tnn', None) / 3 if runtime_data.get('tnn', None) else None
fft_course_runtime = runtime_data.get('fft_course_grain', None) / 4 if runtime_data.get('fft_course_grain', None) else None
fft_fine_runtime = runtime_data.get('fft_fine_grain', None) / 4 if runtime_data.get('fft_fine_grain', None) else None
systolic_course_runtime = runtime_data.get('systolic_course_grain', None) / 4 if runtime_data.get('systolic_course_grain', None) else None
systolic_fine_runtime = runtime_data.get('systolic_fine_grain', None) / 4 if runtime_data.get('systolic_fine_grain', None) else None


# Helper function to convert time string (H:MM:SS) to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == '':
        return 0
    parts = time_str.split(':')
    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    return hours * 3600 + minutes * 60 + seconds

# Extract FFT EDA data
fft_eda = {}
for i in range(len(runtime_df)):
    design = runtime_df.iloc[i]['Design']
    if pd.notna(design) and design.startswith('fft_'):
        genus_time = time_to_seconds(runtime_df.iloc[i]['Genus Runtime'])
        innovus_time = time_to_seconds(runtime_df.iloc[i]['Innovus Runtime'])
        fft_eda[design] = genus_time + innovus_time

# Extract Systolic EDA data
systolic_eda = {}
for i in range(len(runtime_df)):
    design = runtime_df.iloc[i]['Design']
    if pd.notna(design) and design.startswith('sys_'):
        genus_time = time_to_seconds(runtime_df.iloc[i]['Genus Runtime'])
        innovus_time = time_to_seconds(runtime_df.iloc[i]['Innovus Runtime'])
        systolic_eda[design] = genus_time + innovus_time

# Extract TNN EDA data
tnn_eda = {}
for i in range(len(runtime_df)):
    design = runtime_df.iloc[i]['Design']
    if pd.notna(design) and design.startswith('tnn_'):
        genus_time = time_to_seconds(runtime_df.iloc[i]['Genus Runtime'])
        innovus_time = time_to_seconds(runtime_df.iloc[i]['Innovus Runtime'])
        tnn_eda[design] = genus_time + innovus_time

# Extract and sum TNN MACROS
tnn_eda_modules = 0
in_tnn_macros = False
for i in range(len(runtime_df)):
    design = runtime_df.iloc[i]['Design']
    if pd.notna(design) and design == 'TNN MACROS':
        in_tnn_macros = True
        continue
    if in_tnn_macros and pd.notna(design) and design != '':
        genus_time = time_to_seconds(runtime_df.iloc[i]['Genus Runtime'])
        innovus_time = time_to_seconds(runtime_df.iloc[i]['Innovus Runtime'])
        tnn_eda_modules += genus_time + innovus_time

tnn_module_runtime = tnn_eda_modules / 4

modules_dict = {
    'adder': {
        'syn': 51,
        'pnr': 67
    },
    'multiplier': {
        'syn': 24,
        'pnr': 61
    },
    'register': {
        'syn': 13,
        'pnr': 57
    },
    'not': {
        'syn': 14,
        'pnr': 55
    },
    'fifo': {
        'syn': 26,
        'pnr': 77
    },
    'sys_pe': {
        'syn': 76,
        'pnr': 99
    },
    'fft_pe': {
        'syn': 84,
        'pnr': 141
    }
}

fft_fine_module_runtime = (modules_dict['multiplier']['pnr'] + modules_dict['multiplier']['syn'] + \
                    modules_dict['register']['pnr'] + modules_dict['register']['syn'] + \
                    modules_dict['not']['pnr'] + modules_dict['not']['syn'] + \
                    modules_dict['adder']['pnr'] + modules_dict['adder']['syn']) / 4

fft_course_module_runtime = (modules_dict['fft_pe']['pnr'] + modules_dict['fft_pe']['syn']) / 4

systolic_fine_module_runtime = (modules_dict['multiplier']['pnr'] + modules_dict['multiplier']['syn'] + \
                         modules_dict['register']['pnr'] + modules_dict['register']['syn'] + \
                         modules_dict['adder']['pnr'] + modules_dict['adder']['syn'] + \
                         modules_dict['fifo']['pnr'] + modules_dict['fifo']['syn']) / 4

systolic_course_module_runtime = (modules_dict['sys_pe']['pnr'] + modules_dict['sys_pe']['syn']) / 4

# Prepare data for plotting
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# FFT data - 4 configurations (8, 16, 32, 64)
fft_configs = ['8', '16', '32', '64']
fft_eda_values = [fft_eda.get(f'fft_{c}', 0) for c in fft_configs]
fft_fg_values = [fft_fine_runtime for _ in fft_configs]  # Same for each config
fft_cg_values = [fft_course_runtime for _ in fft_configs]  # Same for each config

# Systolic data - 4 configurations (4, 8, 16, 32)
sys_configs = ['4', '8', '16', '32']
sys_eda_values = [systolic_eda.get(f'sys_{c}', 0) for c in sys_configs]
sys_fg_values = [systolic_fine_runtime for _ in sys_configs]  # Same for each config
sys_cg_values = [systolic_course_runtime for _ in sys_configs]  # Same for each config

# TNN data - 3 configurations (96x2, 152x2, 343x2)
tnn_configs_full = ['96x2', '152x2', '343x2']
tnn_configs = [c.split('x')[0] for c in tnn_configs_full]
tnn_eda_values = [tnn_eda.get(f'tnn_{c}', 0) for c in tnn_configs_full]
tnn_runtime_values = [tnn_runtime for _ in tnn_configs]  # Same for each config

# Module runtime data (constant across configs)
fft_fg_module_values = [fft_fine_module_runtime for _ in fft_configs]
fft_cg_module_values = [fft_course_module_runtime for _ in fft_configs]
sys_fg_module_values = [systolic_fine_module_runtime for _ in sys_configs]
sys_cg_module_values = [systolic_course_module_runtime for _ in sys_configs]
tnn_module_values = [tnn_module_runtime for _ in tnn_configs]

# Plot settings
width = 252 / 72
height = width / 1.95
rows = 1
cols = 3

# Color settings
fg_color = "#4697bd"  # Blue (A-Sub)
cg_color = "#e38a3c"   # Orange (A-PE)
eda_color = "#3bac3b"   # Green (EDA-F)
module_fg_color = "#3ca0a3"  # Purple (EDA-Sub)
module_cg_color = "#c29d3e"  # Red (EDA-PE)

# Metric names
metric_names = ['FFT', 'Systolic', 'TNN']

# Create figure
fig, ax_main = plt.subplots(1, 1, figsize=(width, height))

# Set up basic formatting
ax_main.tick_params(axis='x', labelsize=8, pad=0.5, width=0.5)
ax_main.grid(axis='y', linestyle=(0, (2, 7)), linewidth=0.4, which='major')

# Make axis spines thinner
for spine in ax_main.spines.values():
    spine.set_linewidth(0.5)

# Format y-axis ticks with log scale
ax_main.set_yscale('log')
ax_main.yaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)))
ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else '0'))
ax_main.set_ylabel('Runtime (s)', fontsize=8, labelpad=-3)
ax_main.set_ylim(10, 1e7)
ax_main.set_yticks([0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 1e7])

# Set tick parameters after setting scale
ax_main.tick_params(axis='y', labelsize=8, pad=0.5, direction='in', which='major', width=0.5, length=4)
ax_main.tick_params(axis='y', which='minor', direction='in', width=0.3, length=2.5)

# Add a solid gridline at 10^1 (10)
ax_main.axhline(y=10, color='gray', linestyle='-', linewidth=0.4, alpha=0.7, zorder=1)

# Time format labels - adjust positions and colors as needed
time_format_hms_pos = (0.75, 0.35)  # (x, y) in axes coordinates (0-1 range)
time_format_ms_pos = (0.75, 0.23)   # (x, y) in axes coordinates (0-1 range)
time_format_hms_color = 'grey'     # Color for t=hh:mm:ss label
time_format_ms_color = 'grey'      # Color for t=ms label

ax_main.text(time_format_hms_pos[0], time_format_hms_pos[1], 't=hh:mm:ss', 
             transform=ax_main.transAxes, fontsize=7, va='top', ha='left', color=time_format_hms_color, fontweight='normal')
ax_main.text(time_format_ms_pos[0], time_format_ms_pos[1], 't=ms', 
             transform=ax_main.transAxes, fontsize=7, va='top', ha='left', color=time_format_ms_color, fontweight='normal')

# Define annotation offsets for each PNR point [design][config_index] = (x_offset_points, y_offset_points)
# Uses xytext with textcoords='offset points' - positive y is up, negative is down
annotation_offsets = {
    'fft': [
        (1, 3),  # fft_8
        (-3, 3),  # fft_16
        (-3, 4),  # fft_32
        (-2, 3),  # fft_64
    ],
    'fft_fg': (-32.5, 2),  # FFT fine grain line
    'fft_cg': (-32.5, -7),  # FFT coarse grain line
    'fft_fg_module': (-35, 2),  # FFT fine grain module line
    'fft_cg_module': (-35, -7),  # FFT coarse grain module line
    'systolic': [
        (-3, 3),  # sys_4
        (-4, 3),  # sys_8
        (-5, 3),  # sys_16
        (7, -8),  # sys_32
    ],
    'systolic_fg': (-32.5, 2),  # Systolic fine grain line
    'systolic_cg': (-32.5, -7),  # Systolic coarse grain line
    'systolic_fg_module': (-35, 2),  # Systolic fine grain module line
    'systolic_cg_module': (-35, -7),  # Systolic coarse grain module line
    'tnn': [
        (-3, 2.5),  # tnn_96x2
        (-3, 4),  # tnn_152x2
        (-3, 3),  # tnn_343x2
    ],
    'tnn_module': (-32.5, -7),  # TNN ArchX runtime line
    'tnn_module_only': (-35, -7),  # TNN module line
}

# Spacing between metrics
spacing_between_metrics = 0
x_offset = 0

# Plot FFT data
x_positions = [x_offset + i for i in range(len(fft_configs))]
ax_main.plot(x_positions, fft_fg_values, color=fg_color, linewidth=0.8, alpha=0.7, label='PE-Sub')
ax_main.scatter(x_positions, fft_fg_values, color=fg_color, s=15, alpha=1.0, zorder=5, clip_on=False)
ax_main.plot(x_positions, fft_cg_values, color=cg_color, linewidth=0.8, alpha=0.7, label='PE', linestyle='--')
ax_main.scatter(x_positions, fft_cg_values, color=cg_color, s=15, alpha=1.0, zorder=5, marker='^', clip_on=False)
ax_main.plot(x_positions, fft_eda_values, color=eda_color, linewidth=0.8, alpha=0.7, label='EDA-F', linestyle=':')
ax_main.scatter(x_positions, fft_eda_values, color=eda_color, s=15, alpha=1.0, zorder=5, marker='s', clip_on=False)
ax_main.plot(x_positions, fft_fg_module_values, color=module_fg_color, linewidth=0.8, alpha=0.7, label='EDA-Sub', linestyle='-.')
ax_main.scatter(x_positions, fft_fg_module_values, color=module_fg_color, s=15, alpha=1.0, zorder=5, marker='D', clip_on=False)
ax_main.plot(x_positions, fft_cg_module_values, color=module_cg_color, linewidth=0.8, alpha=0.7, label='EDA-PE', linestyle='-.')
ax_main.scatter(x_positions, fft_cg_module_values, color=module_cg_color, s=15, alpha=1.0, zorder=5, marker='p', clip_on=False)

# Add FFT title
title_x = x_offset + (len(fft_configs) - 1) / 2
ax_main.text(title_x, ax_main.get_ylim()[1] * 1.01, metric_names[0], 
             ha='center', va='bottom', fontsize=8)

# Add runtime annotations for FFT PNR values
for i, (x_pos, eda_val) in enumerate(zip(x_positions, fft_eda_values)):
    x_off, y_off = annotation_offsets['fft'][i]
    hours = int(eda_val // 3600)
    minutes = int((eda_val % 3600) // 60)
    seconds = int(eda_val % 60)
    if hours > 0:
        time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
    else:
        time_str = f'{minutes:02d}:{seconds:02d}'
    ax_main.annotate(time_str, (x_pos, eda_val), xytext=(x_off, y_off), textcoords='offset points',
                     fontsize=5, ha='center', va='bottom', color=eda_color)

# Add annotation for FFT fine grain (one label for the line)
fg_val = fft_fg_values[0]
x_off, y_off = annotation_offsets['fft_fg']
milliseconds = int(fg_val * 1e3)
time_str = f'{milliseconds}'
ax_main.annotate(time_str, (x_positions[-1], fg_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=fg_color)

# Add annotation for FFT coarse grain (one label for the line)
cg_val = fft_cg_values[0]
x_off, y_off = annotation_offsets['fft_cg']
milliseconds = int(cg_val * 1e3)
time_str = f'{milliseconds}'
ax_main.annotate(time_str, (x_positions[-1], cg_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=cg_color)

# Add annotation for FFT fine grain modules
fg_module_val = fft_fg_module_values[0]
x_off, y_off = annotation_offsets['fft_fg_module']
hours = int(fg_module_val // 3600)
minutes = int((fg_module_val % 3600) // 60)
seconds = int(fg_module_val % 60)
if hours > 0:
    time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
else:
    time_str = f'{minutes}:{seconds:02d}'
ax_main.annotate(time_str, (x_positions[-1], fg_module_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=module_fg_color)

# Add annotation for FFT coarse grain modules
cg_module_val = fft_cg_module_values[0]
x_off, y_off = annotation_offsets['fft_cg_module']
hours = int(cg_module_val // 3600)
minutes = int((cg_module_val % 3600) // 60)
seconds = int(cg_module_val % 60)
if hours > 0:
    time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
else:
    time_str = f'{minutes}:{seconds:02d}'
ax_main.annotate(time_str, (x_positions[-1], cg_module_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=module_cg_color)

x_offset += len(fft_configs) + spacing_between_metrics

# Plot Systolic data
x_positions = [x_offset + i for i in range(len(sys_configs))]
ax_main.plot(x_positions, sys_fg_values, color=fg_color, linewidth=0.8, alpha=0.7)
ax_main.scatter(x_positions, sys_fg_values, color=fg_color, s=15, alpha=1.0, zorder=5, clip_on=False)
ax_main.plot(x_positions, sys_cg_values, color=cg_color, linewidth=0.8, alpha=0.7, linestyle='--')
ax_main.scatter(x_positions, sys_cg_values, color=cg_color, s=15, alpha=1.0, zorder=5, marker='^', clip_on=False)
ax_main.plot(x_positions, sys_eda_values, color=eda_color, linewidth=0.8, alpha=0.7, linestyle=':')
ax_main.scatter(x_positions, sys_eda_values, color=eda_color, s=15, alpha=1.0, zorder=5, marker='s', clip_on=False)
ax_main.plot(x_positions, sys_fg_module_values, color=module_fg_color, linewidth=0.8, alpha=0.7, linestyle='-.')
ax_main.scatter(x_positions, sys_fg_module_values, color=module_fg_color, s=15, alpha=1.0, zorder=5, marker='D', clip_on=False)
ax_main.plot(x_positions, sys_cg_module_values, color=module_cg_color, linewidth=0.8, alpha=0.7, linestyle='-.')
ax_main.scatter(x_positions, sys_cg_module_values, color=module_cg_color, s=15, alpha=1.0, zorder=5, marker='p', clip_on=False)

# Add Systolic title
title_x = x_offset + (len(sys_configs) - 1) / 2
ax_main.text(title_x, ax_main.get_ylim()[1] * 1.01, metric_names[1], 
             ha='center', va='bottom', fontsize=8)

# Add runtime annotations for Systolic PNR values
for i, (x_pos, eda_val) in enumerate(zip(x_positions, sys_eda_values)):
    x_off, y_off = annotation_offsets['systolic'][i]
    hours = int(eda_val // 3600)
    minutes = int((eda_val % 3600) // 60)
    seconds = int(eda_val % 60)
    if hours > 0:
        time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
    else:
        time_str = f'{minutes:02d}:{seconds:02d}'
    ax_main.annotate(time_str, (x_pos, eda_val), xytext=(x_off, y_off), textcoords='offset points',
                     fontsize=5, ha='center', va='bottom', color=eda_color)

# Add annotation for Systolic fine grain (one label for the line)
fg_val = sys_fg_values[0]
x_off, y_off = annotation_offsets['systolic_fg']
milliseconds = int(fg_val * 1e3)
time_str = f'{milliseconds}'
ax_main.annotate(time_str, (x_positions[-1], fg_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=fg_color)

# Add annotation for Systolic coarse grain (one label for the line)
cg_val = sys_cg_values[0]
x_off, y_off = annotation_offsets['systolic_cg']
milliseconds = int(cg_val * 1e3)
time_str = f'{milliseconds}'
ax_main.annotate(time_str, (x_positions[-1], cg_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=cg_color)

# Add annotation for Systolic fine grain modules
fg_module_val = sys_fg_module_values[0]
x_off, y_off = annotation_offsets['systolic_fg_module']
hours = int(fg_module_val // 3600)
minutes = int((fg_module_val % 3600) // 60)
seconds = int(fg_module_val % 60)
if hours > 0:
    time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
else:
    time_str = f'{minutes}:{seconds:02d}'
ax_main.annotate(time_str, (x_positions[-1], fg_module_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=module_fg_color)

# Add annotation for Systolic coarse grain modules
cg_module_val = sys_cg_module_values[0]
x_off, y_off = annotation_offsets['systolic_cg_module']
hours = int(cg_module_val // 3600)
minutes = int((cg_module_val % 3600) // 60)
seconds = int(cg_module_val % 60)
if hours > 0:
    time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
else:
    time_str = f'{minutes}:{seconds:02d}'
ax_main.annotate(time_str, (x_positions[-1], cg_module_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=module_cg_color)

x_offset += len(sys_configs) + spacing_between_metrics

# Plot TNN data
x_positions = [x_offset + i for i in range(len(tnn_configs))]
ax_main.plot(x_positions, tnn_runtime_values, color=fg_color, linewidth=0.8, alpha=0.7)
ax_main.scatter(x_positions, tnn_runtime_values, color=fg_color, s=15, alpha=1.0, zorder=5, clip_on=False)
ax_main.plot(x_positions, tnn_eda_values, color=eda_color, linewidth=0.8, alpha=0.7, linestyle=':')
ax_main.scatter(x_positions, tnn_eda_values, color=eda_color, s=15, alpha=1.0, zorder=5, marker='s', clip_on=False)
ax_main.plot(x_positions, tnn_module_values, color=module_fg_color, linewidth=0.8, alpha=0.7, linestyle='-.')
ax_main.scatter(x_positions, tnn_module_values, color=module_fg_color, s=15, alpha=1.0, zorder=5, marker='D', clip_on=False)

# Add TNN title
title_x = x_offset + (len(tnn_configs) - 1) / 2
ax_main.text(title_x, ax_main.get_ylim()[1] * 1.01, metric_names[2], 
             ha='center', va='bottom', fontsize=8)

# Add runtime annotations for TNN PNR values
for i, (x_pos, eda_val) in enumerate(zip(x_positions, tnn_eda_values)):
    x_off, y_off = annotation_offsets['tnn'][i]
    hours = int(eda_val // 3600)
    minutes = int((eda_val % 3600) // 60)
    seconds = int(eda_val % 60)
    if hours > 0:
        time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
    else:
        time_str = f'{minutes:02d}:{seconds:02d}'
    ax_main.annotate(time_str, (x_pos, eda_val), xytext=(x_off, y_off), textcoords='offset points',
                     fontsize=5, ha='center', va='bottom', color=eda_color)

# Add annotation for TNN module (one label for the line)
module_val = tnn_runtime_values[0]
x_off, y_off = annotation_offsets['tnn_module']
milliseconds = int(module_val * 1e3)
time_str = f'{milliseconds}'
ax_main.annotate(time_str, (x_positions[-1], module_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=fg_color)

# Add annotation for TNN modules only
tnn_module_val = tnn_module_values[0]
x_off, y_off = annotation_offsets['tnn_module_only']
hours = int(tnn_module_val // 3600)
minutes = int((tnn_module_val % 3600) // 60)
seconds = int(tnn_module_val % 60)
if hours > 0:
    time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
else:
    time_str = f'{minutes}:{seconds:02d}'
ax_main.annotate(time_str, (x_positions[-1], tnn_module_val), xytext=(x_off, y_off), textcoords='offset points',
                 fontsize=5, ha='left', va='bottom', color=module_fg_color)

# Set x-axis ticks and labels
all_x_positions = []
all_x_labels = []
x_offset = 0

for i in range(len(fft_configs)):
    all_x_positions.append(x_offset + i)
    all_x_labels.append(fft_configs[i])
x_offset += len(fft_configs) + spacing_between_metrics

for i in range(len(sys_configs)):
    all_x_positions.append(x_offset + i)
    all_x_labels.append(sys_configs[i])
x_offset += len(sys_configs) + spacing_between_metrics

for i in range(len(tnn_configs)):
    all_x_positions.append(x_offset + i)
    all_x_labels.append(tnn_configs[i])

ax_main.set_xticks(all_x_positions)
ax_main.set_xticklabels(all_x_labels, fontsize=8)

# Add legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=fg_color, markersize=6, linewidth=0, label='A-Sub'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=cg_color, markersize=6, linewidth=0, label='A-PE'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=eda_color, markersize=6, linewidth=0, label='EDA-F'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=module_fg_color, markersize=6, linewidth=0, label='EDA-Sub'),
    Line2D([0], [0], marker='p', color='w', markerfacecolor=module_cg_color, markersize=6, linewidth=0, label='EDA-PE')
]
ax_main.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=7.5, 
               handletextpad=-0.4, columnspacing=-0.1)

# Adjust layout
plt.tight_layout(pad=1.0)

# Save figure
plt.savefig('agraph_casestudy/figures/figures/runtime_comparison.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('agraph_casestudy/figures/figures/runtime_comparison.pdf', 
           dpi=300, bbox_inches='tight', facecolor='white')

