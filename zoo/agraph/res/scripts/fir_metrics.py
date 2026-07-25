from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd
import os
from PIL import Image
import fitz  # PyMuPDF

fir_path = 'zoo/agraph/designs/sc_fir/results/'

# Read combined area.csv (and throughput.csv if available)
area_csv_path = 'zoo/agraph/designs/sc_fir/results/area.csv'
throughput_csv_path = 'zoo/agraph/designs/sc_fir/results/throughput.csv'  # Update if needed

fir_area_df = pd.read_csv(area_csv_path)
try:
    fir_throughput_df = pd.read_csv(throughput_csv_path)
except FileNotFoundError:
    fir_throughput_df = None


# Filter area and throughput dataframes by arch
fir_area_256_df = fir_area_df[fir_area_df['arch'] == 256]
fir_area_32_df = fir_area_df[fir_area_df['arch'] == 32]

if fir_throughput_df is not None and 'arch' in fir_throughput_df.columns:
    fir_throughput_256_df = fir_throughput_df[fir_throughput_df['arch'] == 256]
    fir_throughput_32_df = fir_throughput_df[fir_throughput_df['arch'] == 32]
    throughput_256 = fir_throughput_256_df['throughput GOPs'].tolist()
    throughput_32 = fir_throughput_32_df['throughput GOPs'].tolist()
else:
    throughput_256 = [None] * len(fir_area_256_df)
    throughput_32 = [None] * len(fir_area_32_df)

bitwidth_256 = fir_area_256_df['bitwidth'].tolist()
area_256 = [a / 1000 for a in fir_area_256_df['area jj'].tolist()]
bitwidth_32 = fir_area_32_df['bitwidth'].tolist()
area_32 = [a / 1000 for a in fir_area_32_df['area jj'].tolist()]

u256_color = "#4697bd"  # Blue (SM)
u32_color = "#e38a3c"     # Orange (PE) 

# Create figure with two subplots
width = 252 / 72
height = width / 2.7 # Adjusted for 1 row layout
wspace = 0.3  # Adjust spacing between subplots
fig, (ax_throughput, ax_area) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'wspace': wspace})

# Plot throughput on left axis
if all(x is not None for x in throughput_256) and all(x is not None for x in throughput_32):
    ax_throughput.plot(bitwidth_256, throughput_256, linewidth=0.5, label='U 256', color=u256_color)
    ax_throughput.plot(bitwidth_32, throughput_32, linewidth=0.5, label='U 32', color=u32_color)
    ax_throughput.legend(fontsize=8, frameon=False)
else:
    ax_throughput.text(0.5, 0.5, 'No throughput data', ha='center', va='center', fontsize=8, color='gray', transform=ax_throughput.transAxes)
ax_throughput.set_xlabel('Bits', fontsize=8)
ax_throughput.set_title('Thr. [GOPs]', fontsize=8, pad=5)
ax_throughput.tick_params(axis='both', labelsize=8, pad=0.5, width=0.25, length=3, direction='in', which='major')
ax_throughput.tick_params(axis='both', which='minor', width=0.25, length=1.5, direction='in')
ax_throughput.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False, direction='in')
if bitwidth_256:
    ax_throughput.set_xticks(bitwidth_256)
    ax_throughput.set_xlim(bitwidth_256[0], bitwidth_256[-1])
ax_throughput.set_yscale('log')
ax_throughput.set_ylim(0.001, 1000)
ax_throughput.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax_throughput.set_yticklabels(['$10^{-3}$', '', '$10^{-1}$', '', '$10^{1}$', '', '$10^{3}$'])
ax_throughput.grid(axis='both', linestyle='--', linewidth=0.3, which='major')
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

if not os.path.exists('zoo/agraph/res/figures'):
    os.makedirs('zoo/agraph/res/figures')

# Save figure
png_path = 'zoo/agraph/res/figures/fir_metrics_comparison.png'
pdf_path = 'zoo/agraph/res/figures/fir_metrics_comparison.pdf'

plt.savefig(png_path, dpi=1200, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, dpi=1200, bbox_inches='tight', facecolor='white')

# Post-process: crop top and bottom of PNG
crop_top = 28   # pixels to crop from top
crop_bottom = 28  # pixels to crop from bottom
try:
    with Image.open(png_path) as im:
        width, height = im.size
        cropped = im.crop((0, crop_top, width, height - crop_bottom))
        cropped.save(png_path)
except Exception as e:
    print(f"Warning: Could not crop image: {e}")

# Post-process: crop top and bottom of PDF (in points)
crop_top_pt = 7  # points to crop from top
crop_bottom_pt = 8  # points to crop from bottom
try:
    doc = fitz.open(pdf_path)
    for page in doc:
        rect = page.rect
        # CropBox: (x0, y0, x1, y1)
        page.set_cropbox(fitz.Rect(
            rect.x0,
            rect.y0 + crop_top_pt,
            rect.x1,
            rect.y1 - crop_bottom_pt
        ))
    doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
except Exception as e:
    print(f"Warning: Could not crop PDF: {e}")