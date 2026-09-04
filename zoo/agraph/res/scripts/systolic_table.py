"""Generate the systolic-array configuration + full-results table (tab:sys_thr).

Reads designs/systolic_cg/results/{area,energy,power,memory,runtime}.csv (one
row per array config, keyed by array dimension) plus each config's workload.yaml
for the GEMM matrix dimension, and emits a booktabs table to
res/tables/txt/systolic_breakdown.txt. Metrics are one column per array config.
"""
import glob
import os
import pandas as pd
import yaml

base = 'zoo/agraph/designs/systolic_cg/'
results = base + 'results/'
out_dir = 'zoo/agraph/res/tables/txt/'
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir + 'systolic_breakdown.txt'

caption = ('Systolic array configuration and full \\thiswork results.\n'
           '    Array dimension is the shape of the systolic array.\n'
           r'    Matrix dimension represents m$\times$k$\times$n of GEMM.')
label = 'tab:sys_thr'

area = pd.read_csv(results + 'area.csv')
energy = pd.read_csv(results + 'energy.csv')
power = pd.read_csv(results + 'power.csv')
memory = pd.read_csv(results + 'memory.csv')
runtime = pd.read_csv(results + 'runtime.csv')

n = len(area)
array_dims = area['arch'].tolist()  # e.g. '4x4', '8x8', ...


def _run_dir(i):
    return sorted(glob.glob(f'{base}description/runs/gemm/arch_{i}/config_*'))[0]


def matrix_dim(i):
    """GEMM matrix dimension for config i, from its workload.yaml (authoritative)."""
    wl = yaml.safe_load(open(f'{_run_dir(i)}/workload.yaml'))
    return int(wl['workload']['configuration']['matrix_dim'])


def sram_capacity_kb(i):
    """Total on-chip SRAM capacity (KB) for config i, summed over every SRAM
    module in its architecture.yaml (isram + osram + wsram), each sized as
    depth * width * bank / 1024."""
    arch = yaml.safe_load(open(f'{_run_dir(i)}/architecture.yaml'))
    mods = arch.get('architecture', arch).get('module', arch)
    total = 0.0
    for m in mods.values():
        q = m.get('query', {}) if isinstance(m, dict) else {}
        if q.get('class') == 'sram':
            total += q['depth'] * q['width'] * q['bank'] / 1024
    assert total > 0, f'no SRAM module found for config {i}'
    return total


mds = [matrix_dim(i) for i in range(n)]

# Throughput (GFLOPs): MACs (matrix_dim^3) / archx runtime. runtime.csv is in ms,
# so / runtime_ms / 1e6 converts to GFLOPs.
throughput = [(md ** 3) / runtime['archx'][i] / 1e6 for i, md in enumerate(mds)]
sram_kb = [sram_capacity_kb(i) for i in range(n)]


def f1(v):
    return f'{v:.1f}'


def f2(v):
    return f'{v:.2f}'


def gi(v):
    return f'{int(round(v))}'


# (row label, list of 4 formatted cells)
ROWS = [
    (r'Array dim.',                array_dims),
    (r'Matrix dim.',              [f'{m}x{m}x{m}' for m in mds]),
    (r'Throughput ($GFLOPs$)',    [f1(t) for t in throughput]),
    (r'Compute Area ($\mu m^2$)', [f2(area['archx'][i] * 1000) for i in range(n)]),
    (r'Compute D-Energy ($nJ$)',  [f2(energy['archx'][i]) for i in range(n)]),
    (r'Compute L-Power ($mW$)',   [f2(power['archx'][i]) for i in range(n)]),
    (r'Total SRAM ($KB$)',        [gi(s) for s in sram_kb]),
    (r'SRAM Area ($\mu m^2$)',    [f2(memory['memory_area'][i] * 1000) for i in range(n)]),
    (r'SRAM D-energy ($nJ$)',     [f2(memory['memory_dynamic_energy'][i]) for i in range(n)]),
    (r'SRAM L-Power ($mW$)',      [f2(memory['memory_leakage_power'][i]) for i in range(n)]),
]

col_spec = 'c' * (n + 1)
body = [rf'        \textbf{{{lbl}}} & ' + ' & '.join(cells) + r' \\' for lbl, cells in ROWS]
rows_tex = f'\n        \\cmidrule{{1-{n + 1}}}\n'.join(body)

table = rf'''\begin{{table}}[!t]
    \centering
    \caption{{{caption}}}
    \begin{{adjustbox}}{{max width=\linewidth}}
    \begin{{tabular}}{{{col_spec}}}
        \toprule
{rows_tex}
        \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \label{{{label}}}
\end{{table}}
'''

with open(out_path, 'w') as f:
    f.write(table)

# Companion CSV with the same results (raw values, no LaTeX) for easy reading:
# one row per array config.
csv_dir = 'zoo/agraph/res/tables/csv/'
os.makedirs(csv_dir, exist_ok=True)
csv_records = [{
    'array_dim': array_dims[i],
    'matrix_dim': f'{mds[i]}x{mds[i]}x{mds[i]}',
    'throughput_gflops': throughput[i],
    'compute_area_um2': area['archx'][i] * 1000,
    'compute_denergy_nj': energy['archx'][i],
    'compute_lpower_mw': power['archx'][i],
    'total_sram_kb': sram_kb[i],
    'sram_area_um2': memory['memory_area'][i] * 1000,
    'sram_denergy_nj': memory['memory_dynamic_energy'][i],
    'sram_lpower_mw': memory['memory_leakage_power'][i],
} for i in range(n)]
pd.DataFrame(csv_records).to_csv(csv_dir + 'systolic_breakdown.csv', index=False)
