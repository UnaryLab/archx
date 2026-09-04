"""Generate the LaTeX validation table for the RISC-V GEMM study.

Reads the per-metric result CSVs written by designs/riscv/query.py (each has
columns archx / pnr / percent_error) and emits a booktabs table comparing archx
against the baseline to res/tables/txt/riscv_validation.txt.
"""
import math
import os
import pandas as pd

results = 'zoo/agraph/designs/riscv/results/'
out_dir = 'zoo/agraph/res/tables/txt/'
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir + 'riscv_validation.txt'

caption = 'Validation of RISC-V GEMM Results.'
label = 'tab:riscv_gemm'

# (row label, csv, scale applied to archx & baseline, value format)
# scale converts the stored unit to the displayed unit (e.g. mW -> uW).
ROWS = [
    (r'Area ($mm^2$)',       'area.csv',   1.0,    'sci'),
    (r'Dynamic energy (nJ)', 'energy.csv', 1.0,    'f3'),
    (r'Leakage power (uW)',  'power.csv',  1000.0, 'f2'),
]


def fmt_value(v, style):
    if style == 'sci':
        if v == 0:
            return r'$0$'
        exp = math.floor(math.log10(abs(v)))
        return rf'${v / 10**exp:.3f}\times10^{{{exp}}}$'
    if style == 'f3':
        return f'{v:.3f}'
    if style == 'f2':
        return f'{v:.2f}'
    raise ValueError(f'unknown value format <{style}>')


def fmt_err(v):
    s = f'{v:.2f}'
    if s == '-0.00':
        s = '0.00'
    return rf'{s}\%'


body = []
for lbl, csv, scale, style in ROWS:
    row = pd.read_csv(results + csv).iloc[0]
    archx = fmt_value(row['archx'] * scale, style)
    base = fmt_value(row['pnr'] * scale, style)
    body.append(rf'        \textbf{{{lbl}}} & {archx} & {base} & {fmt_err(row["percent_error"])} \\')

rows_tex = '\n        \\cmidrule{1-4}\n'.join(body)

table = rf'''\begin{{table}}[!t]
    \centering
    \caption{{{caption}}}
    \begin{{adjustbox}}{{max width=\linewidth}}
    \begin{{tabular}}{{cccc}}
        \toprule
        \textbf{{Metric}} & \textbf{{Archx}} & \textbf{{Baseline}} & \textbf{{Relative error (\%)}}\\
        \cmidrule{{1-4}}
{rows_tex}
        \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \label{{{label}}}
\end{{table}}
'''

with open(out_path, 'w') as f:
    f.write(table)

# Companion CSV with the same results (raw values, no LaTeX) for easy reading.
import re

csv_dir = 'zoo/agraph/res/tables/csv/'
os.makedirs(csv_dir, exist_ok=True)


def _plain(s):
    s = s.replace(r'\mu', 'u').replace(r'\times', 'x')
    s = re.sub(r'\\[a-zA-Z]+|[${}^]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


csv_records = []
for lbl, csv, scale, style in ROWS:
    row = pd.read_csv(results + csv).iloc[0]
    csv_records.append({'metric': _plain(lbl), 'archx': row['archx'] * scale,
                        'baseline': row['pnr'] * scale, 'relative_error_percent': row['percent_error']})
pd.DataFrame(csv_records).to_csv(csv_dir + 'riscv_validation.csv', index=False)
