from loguru import logger
import os, runpy, warnings

logger.remove()

# The query/figure files are standalone scripts (run from the repository root,
# like zoo/llm/results/figure_generation.py): the *_query scripts read
# zoo/chiplet4ai/designs/<design>/description runs and write results/csv, and
# the fig_* scripts read results/csv and write results/figs.
csv_path = './zoo/chiplet4ai/results/csv/'
fig_path = './zoo/chiplet4ai/results/figs/'

if not os.path.exists(csv_path):
    os.makedirs(csv_path)

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# Suppress matplotlib warnings that are not necessary
warnings.filterwarnings(
    "ignore",
    message=".*tight_layout.*",
    category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message=".*FigureCanvasAgg is non-interactive.*",
    category=UserWarning
)

_QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query')

# Delete last run's outputs BEFORE regenerating them. If an assertion fires mid-run (the
# fig_2_query root-branch check, say), the outputs that never got rewritten are then ABSENT
# and self-announcing, instead of silently stale - stale files are byte-for-byte
# indistinguishable from fresh ones, so a partial run would otherwise be read as a full one.
#
# EXPLICIT LIST, never a glob. Every path below is the literal destination of a `to_csv` /
# `savefig` in the four scripts run by the loop, so re-running the loop recreates exactly
# this set. A glob over results/csv or results/figs would be unsafe: it would also match any
# hand-made or externally supplied file dropped in those directories. Nothing outside this
# list is touched, and in particular nothing under designs/*/description (the per-run
# architecture.yaml / workload.yaml / checkpoint.json / metric.yaml source data and the
# configurations.csv index) is reachable from here - those are INPUTS the queries read.
_GENERATED_OUTPUTS = [
    csv_path + 'array_performance_metrics.csv',             # fig_1_query
    csv_path + 'array_performance_metrics_scientific.csv',  # fig_1_query
    fig_path + 'fig_1.png',                                 # fig_1
    csv_path + 'bandwidth_performance_metrics.csv',             # fig_2_query
    csv_path + 'bandwidth_performance_metrics_scientific.csv',  # fig_2_query
    fig_path + 'fig_2.png',                                 # fig_2
]

for path in _GENERATED_OUTPUTS:
    if os.path.exists(path):
        os.remove(path)
        print(f'removed stale output {path}')

for name in ['fig_1_query', 'fig_1', 'fig_2_query', 'fig_2']:
    print(f'\nStarting {name}...')
    runpy.run_path(os.path.join(_QUERY_DIR, name + '.py'), run_name='__main__')
    print(f'{name} complete.')
