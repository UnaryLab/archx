"""Generate the LaTeX power-validation table for the SC-FIR study.

Reads designs/sc_fir/results/split_power.csv (dynamic/leakage power vs baseline,
one row per bitwidth) and emits a booktabs table for a single bitwidth operating
point to res/tables/fir_power.txt.
"""
import math
import os
import pandas as pd

# The FIR study sweeps bitwidth; the validation table reports one operating
# point. Change this to retarget the table at a different bitwidth.
BITWIDTH = 8

results = 'agraph/designs/sc_fir/results/'
out_dir = 'agraph/res/tables/'
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir + 'fir_power.txt'

caption = 'Validation of FIR power results.'
label = 'tab:fir_power'

df = pd.read_csv(results + 'split_power.csv')
sel = df[df['bitwidth'] == BITWIDTH]
assert len(sel) == 1, f'expected one bitwidth-{BITWIDTH} row in split_power.csv, got {len(sel)}'
row = sel.iloc[0]


def fmt_val(v):
    return f'{v:.3f}'


def fmt_err(v):
    # Near-zero errors (e.g. leakage) are shown in scientific notation so they
    # do not collapse to 0.00; everything else uses two decimals.
    if v != 0 and abs(v) < 1e-2:
        exp = math.floor(math.log10(abs(v)))
        return rf'{v / 10**exp:.2f}$\times10^{{{exp}}}$'
    s = f'{v:.2f}'
    return '0.00' if s == '-0.00' else s


# (row label, archx column, baseline column, error column, W -> display scale)
ROWS = [
    (r'Dynamic power ($\mu$W)', 'dynamic_power W', 'baseline dynamic power W', 'dynamic power percent_dif', 1e6),
    (r'Leakage power (mW)',     'leakage_power W', 'baseline leakage power W', 'leakage power percent_dif', 1e3),
]

body = []
for lbl, a_col, b_col, e_col, scale in ROWS:
    archx = fmt_val(row[a_col] * scale)
    base = fmt_val(row[b_col] * scale)
    body.append(rf'        \textbf{{{lbl}}} & {archx} & {base} & {fmt_err(row[e_col])} \\')

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
