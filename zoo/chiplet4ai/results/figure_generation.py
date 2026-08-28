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
# `savefig` in the scripts run by the loop, so re-running the loop recreates exactly
# this set. A glob over results/csv or results/figs would be unsafe: it would also match any
# hand-made or externally supplied file dropped in those directories. Nothing outside this
# list is touched, and in particular nothing under designs/*/description (the per-run
# architecture.yaml / workload.yaml / checkpoint.json / metric.yaml source data and the
# configurations.csv index) is reachable from here - those are INPUTS the queries read.
#
# The three sequence-length slices share one naming scheme, so they are built rather than
# spelled out; the stems below are the literal `out_name` values in the query scripts.
_SEQ_SLICES = ['4096', '131072', 'mixed']

# Kept in step with FIG_4_CRITERIA in query/utils.py, which is where the criteria are
# defined; spelled out here rather than imported so this module stays free of the query
# package's own imports (it runs the queries as scripts, it does not import them).
_FIG_4_CRITERIA = ['runtime', 'avg_band', 'throughput']

def _slice_csvs(stem):
    return [csv_path + f'{stem}_seqlen_{slice_name}{suffix}.csv'
            for slice_name in _SEQ_SLICES
            for suffix in ('', '_scientific')]

_GENERATED_OUTPUTS = [
    *_slice_csvs('array_performance_metrics'),              # fig_1_query
    fig_path + 'fig_1.pdf',                                 # fig_1
    csv_path + 'bandwidth_performance_metrics.csv',             # fig_2_query
    csv_path + 'bandwidth_performance_metrics_scientific.csv',  # fig_2_query
    fig_path + 'fig_2.pdf',                                 # fig_2
    *_slice_csvs('array_utilization_metrics'),              # fig_3_query
    fig_path + 'fig_3.pdf',                                 # fig_3
    # fig_4 is drawn once per selection criterion (see FIG_4_CRITERIA in query/utils.py):
    # the same models and axes, differing only in which design point each model was given.
    *[path
      for key in _FIG_4_CRITERIA
      for path in (csv_path + f'dram_bandwidth_metrics_{key}.csv',
                   csv_path + f'dram_bandwidth_metrics_{key}_scientific.csv',
                   fig_path + f'fig_4_{key}.pdf')],
    *_slice_csvs('array_utilization_nostall_metrics'),      # fig_5_query
    fig_path + 'fig_5.pdf',                                 # fig_5
    csv_path + 'array_shape_performance_metrics.csv',            # fig_6_query
    csv_path + 'array_shape_performance_metrics_scientific.csv', # fig_6_query
    fig_path + 'fig_6.pdf',                                 # fig_6
]

for path in _GENERATED_OUTPUTS:
    if os.path.exists(path):
        os.remove(path)
        print(f'removed stale output {path}')

# ORDER MATTERS: THE LAST THREE PAIRS ARE A CHAIN, which is why they run out of numeric
# order. fig_6_query sweeps the array-shape grid; fig_4_query picks a design point per
# model out of that grid (one per criterion in FIG_4_CRITERIA); fig_2_query then reads
# fig_4's avg_band point so its capacity sweep describes the same machine. Each needs the
# previous one's CSV to already exist.
#
#     fig_6_query -> fig_4_query -> fig_2_query
#
# fig_1, fig_3 and fig_5 are independent and stay next to their own queries.
for name in ['fig_1_query', 'fig_1', 'fig_3_query', 'fig_3', 'fig_5_query', 'fig_5',
             'fig_6_query', 'fig_6', 'fig_4_query', 'fig_4', 'fig_2_query', 'fig_2']:
    print(f'\nStarting {name}...')
    runpy.run_path(os.path.join(_QUERY_DIR, name + '.py'), run_name='__main__')
    print(f'{name} complete.')
