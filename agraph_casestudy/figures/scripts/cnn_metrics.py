import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load the data
power_df = pd.read_csv('agraph_casestudy/cnn_sc/results/power_breakdown.csv', index_col=0)
area_df = pd.read_csv('agraph_casestudy/cnn_sc/results/area_breakdown.csv', index_col=0)

# Combine all splitters into one category
def combine_splitters(df):
    df_copy = df.copy()
    splitter_rows = df_copy[df_copy.index.str.contains('splitter')]
    if not splitter_rows.empty:
        splitter_sum = splitter_rows.sum()
        df_copy = df_copy[~df_copy.index.str.contains('splitter')]
        df_copy.loc['splitter'] = splitter_sum
    return df_copy

power_df = combine_splitters(power_df)
area_df = combine_splitters(area_df)

# Rename nrdo to reg
power_df.rename(index={'nrdo': 'reg'}, inplace=True)
area_df.rename(index={'nrdo': 'reg'}, inplace=True)

# Color dictionaries for each slice
power_colors = {
    'mac': "#4697bd",
    'splitter': "#e38a3c",
    'reg': "#3bac3b"
}

area_colors = {
    'mac': "#4697bd",
    'splitter': "#e38a3c",
    'reg': "#3bac3b"
}

# Label mapping
label_mapping = {'mac': 'PE'}

# Create figure with two subplots
width = 252 / 72
height = width / 2.2
fig, ax = plt.subplots(1, 2, figsize=(width * 2, height))

# Power breakdown pie chart
power_values = power_df['0'].values
power_labels = [label_mapping.get(label, label) for label in power_df.index.values]
power_color_list = [power_colors.get(label, '#CCCCCC') for label in power_df.index.values]

# Function to show percentage only if >= 2%
def autopct_format(pct):
    return f'{pct:.1f}%' if pct >= 2 else ''

wedges0, texts0, autotexts0 = ax[0].pie(power_values, labels=power_labels, autopct=autopct_format, 
          colors=power_color_list, startangle=90, textprops={'fontsize': 8})
ax[0].set_title('Power Breakdown', fontsize=8, fontweight='bold')

# Rotate percentage text to match slice angle only if < 10%
power_percentages = 100 * power_values / power_values.sum()
for i, (autotext, wedge, pct) in enumerate(zip(autotexts0, wedges0, power_percentages)):
    if pct < 10:
        angle = (wedge.theta2 + wedge.theta1) / 2
        autotext.set_rotation(angle)

# Area breakdown pie chart
area_values = area_df['0'].values
area_labels = [label_mapping.get(label, label) for label in area_df.index.values]
area_color_list = [area_colors.get(label, '#CCCCCC') for label in area_df.index.values]

wedges1, texts1, autotexts1 = ax[1].pie(area_values, labels=area_labels, autopct=autopct_format, 
          colors=area_color_list, startangle=90, textprops={'fontsize': 8})
ax[1].set_title('Area Breakdown', fontsize=8, fontweight='bold')

# Rotate percentage text to match slice angle only if < 10%
area_percentages = 100 * area_values / area_values.sum()
for i, (autotext, wedge, pct) in enumerate(zip(autotexts1, wedges1, area_percentages)):
    if pct < 10:
        angle = (wedge.theta2 + wedge.theta1) / 2
        autotext.set_rotation(angle)

# Adjust label position for 'splitter' in area breakdown
for i, (text, label) in enumerate(zip(texts1, area_labels)):
    if label == 'splitter':
        x, y = text.get_position()
        text.set_position((x - 0.17, y + 0.12))

plt.subplots_adjust(wspace=-0.5)
plt.savefig('agraph_casestudy/figures/figures/cnn_metrics.png', dpi=1200, bbox_inches='tight')

plt.savefig('agraph_casestudy/figures/figures/cnn_metrics.pdf', dpi=1200, bbox_inches='tight')