"""Generate the LaTeX CNN array-breakdown table for the SC-CNN study.

Reads designs/sc_cnn/results/{area_breakdown,power_breakdown}.csv (per-module
area in JJ and power in W) and emits a booktabs table to
res/tables/sc_cnn_breakdown.txt.
"""
import os
import pandas as pd

results = 'agraph/designs/sc_cnn/results/'
out_dir = 'agraph/res/tables/'
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir + 'cnn_breakdown.txt'

caption = ('Validation of CNN array breakdown.\n'
           '    S denotes splitters.\n'
           '    NDRO is for Non-Destructive Readout memory.')
label = 'tab:sc_cnn_breakdown'

# module key in the *_breakdown.csv -> column header in the table
MODULES = [
    ('mac',             'MAC'),
    ('mac_splitter',    'MAC S'),
    ('weight_splitter', 'Weight S'),
    ('input_splitter',  'Input S'),
    ('nrdo',            'NDRO'),
]

area = pd.read_csv(results + 'area_breakdown.csv', index_col=0).iloc[:, 0]
power = pd.read_csv(results + 'power_breakdown.csv', index_col=0).iloc[:, 0]


def area_fmt(v):
    if v >= 1e6:
        return f'{v / 1e6:.1f}M'
    if v >= 1e3:
        return f'{v / 1e3:.1f}K'
    return f'{int(round(v))}'


def power_fmt(v):  # stored in W, displayed in mW
    return f'{v * 1e3:.2f}'


headers = ' & '.join(rf'\textbf{{{h}}}' for _, h in MODULES)
area_cells = ' & '.join(area_fmt(area[k]) for k, _ in MODULES)
power_cells = ' & '.join(power_fmt(power[k]) for k, _ in MODULES)

table = rf'''\begin{{table}}[!t]
    \centering
    \caption{{{caption}}}
    \begin{{adjustbox}}{{max width=\linewidth}}
    \begin{{tabular}}{{cccccc}}
        \toprule
        \textbf{{Module}} & {headers} \\
        \cmidrule{{1-6}}
        \textbf{{Area (JJ)}} & {area_cells} \\
        \cmidrule{{1-6}}
        \textbf{{Power (mW)}} & {power_cells} \\
        \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \label{{{label}}}
\end{{table}}
'''

with open(out_path, 'w') as f:
    f.write(table)

# Companion CSV with the same results (raw values, no LaTeX) for easy reading:
# one row per module, area in JJ and power in mW.
csv_dir = 'agraph/res/csv/'
os.makedirs(csv_dir, exist_ok=True)
csv_records = [{'module': h, 'area_jj': float(area[k]), 'power_mw': float(power[k]) * 1e3}
               for k, h in MODULES]
pd.DataFrame(csv_records).to_csv(csv_dir + 'cnn_breakdown.csv', index=False)
