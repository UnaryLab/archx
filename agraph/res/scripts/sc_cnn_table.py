"""Generate the LaTeX validation table for the SC-CNN study.

Reads designs/sc_cnn/results/{area,power,throughput}.csv and emits a booktabs
table (archx vs baseline + relative error) to res/tables/sc_cnn.txt.
"""
import os
import pandas as pd

results = 'agraph/designs/sc_cnn/results/'
out_dir = 'agraph/res/tables/'
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir + 'sc_cnn.txt'

caption = ('Validation of CNN Results.\n'
           r'    3D PE array has a dimension of $64\times64\times32$.')
label = 'tab:sc_cnn'

area = pd.read_csv(results + 'area.csv')
power = pd.read_csv(results + 'power.csv')
thr = pd.read_csv(results + 'throughput.csv')


def millions(v):
    return f'{v / 1e6:.1f}M'


def f1(v):
    return f'{v:.1f}'


def f2(v):
    return f'{v:.2f}'


base_row = area[area['config'] == 'base'].iloc[0]
ovh_row = area[area['config'] == 'with_overhead'].iloc[0]

# power.csv / throughput.csv have quirky column names (a trailing "W W", and
# throughput headers that embed the value), so read those by position.
p = power.iloc[0]
power_archx, power_base, power_pct = p.iloc[3], p.iloc[4], p.iloc[5]

t = thr.iloc[0]
# The stored archx throughput is 1000x the baseline's unit; convert to TMACs
# and recompute the error (the stored percent_dif compares mismatched units).
thr_archx, thr_base = t.iloc[1] / 1000.0, t.iloc[2]
thr_pct = (thr_archx - thr_base) / thr_base * 100

rows = [
    (r'Area w/o overhead (JJ)', millions(base_row['area jj']), millions(base_row['baseline area jj']), f2(base_row['area_percent_dif'])),
    (r'Area w/ overhead (JJ)',  millions(ovh_row['area jj']),  millions(ovh_row['baseline area jj']),  f2(ovh_row['area_percent_dif'])),
    (r'Power (W)',              f1(power_archx),               f1(power_base),                          f2(power_pct)),
    (r'Throughput (TMACs)',     f1(thr_archx),                 f1(thr_base),                            f2(thr_pct)),
]

body = [rf'        \textbf{{{lbl}}} & {a} & {b} & {e} \\' for lbl, a, b, e in rows]
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
csv_dir = 'agraph/res/csv/'
os.makedirs(csv_dir, exist_ok=True)
csv_records = [
    {'metric': 'Area w/o overhead (JJ)', 'archx': base_row['area jj'], 'baseline': base_row['baseline area jj'], 'relative_error_percent': base_row['area_percent_dif']},
    {'metric': 'Area w/ overhead (JJ)',  'archx': ovh_row['area jj'],  'baseline': ovh_row['baseline area jj'],  'relative_error_percent': ovh_row['area_percent_dif']},
    {'metric': 'Power (W)',              'archx': power_archx,          'baseline': power_base,                   'relative_error_percent': power_pct},
    {'metric': 'Throughput (TMACs)',    'archx': thr_archx,            'baseline': thr_base,                     'relative_error_percent': thr_pct},
]
pd.DataFrame(csv_records).to_csv(csv_dir + 'sc_cnn.csv', index=False)
