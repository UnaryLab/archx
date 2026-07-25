from zoo.llm.results.query import area_power_breakdown, batch_size_breakdown, comprehensive_table, end_to_end_latency_breakdown, \
    gemm_breakdown, noc_breakdown, nonlinear_breakdown, op_carbon

from loguru import logger
import os, sys, warnings

logger.remove()

input_path = './zoo/llm/runs/'
csv_path = './zoo/llm/results/csv/'
fig_path = './zoo/llm/results/figs/'
table_path = './zoo/llm/results/tables/'

if not os.path.exists(csv_path):
    os.makedirs(csv_path)

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

if not os.path.exists(table_path):
    os.makedirs(table_path)

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

print('\nStarting Area and Power Breakdown Query...')
area_power_breakdown.query(input_path=input_path, output_path=csv_path)
print('Area and power breakdown query complete.')

print('\nStarting Batch Size Breakdown Query...')
batch_size_breakdown.query(input_path=input_path, output_path=csv_path)
print('Batch size breakdown query complete.')

print('\nStarting Comprehensive Table Query...')
comprehensive_table.query(input_path=input_path, output_path=csv_path)
print('Comprehensive table query complete.')

print('\nStarting End-to-End Latency Breakdown Query...')
end_to_end_latency_breakdown.query(input_path=input_path, output_path=csv_path)
print('End-to-end latency breakdown query complete.')

print('\nStarting GEMM Breakdown Query...')
gemm_breakdown.query(input_path=input_path, output_path=csv_path)
print('GEMM breakdown query complete.')

print('\nStarting NoC Breakdown Query...')
noc_breakdown.query(input_path=input_path, output_path=csv_path)
print('NoC breakdown query complete.')

print('\nStarting Nonlinear Breakdown Query...')
nonlinear_breakdown.query(input_path=input_path, output_path=csv_path)
print('Nonlinear breakdown query complete.')

print('\nStarting Operational Carbon Query...')
op_carbon.query(input_path=input_path, output_path=csv_path)
print('Carbon query complete.')

print('\nStarting Figure Generation...')
area_power_breakdown.figure(input_path=csv_path, output_path=fig_path)
batch_size_breakdown.figure(input_path=csv_path, output_path=fig_path)
comprehensive_table.table(input_path=csv_path, output_path=table_path)
end_to_end_latency_breakdown.figure(input_path=csv_path, output_path=fig_path)
gemm_breakdown.figure(input_path=csv_path, output_path=fig_path)
noc_breakdown.figure(input_path=csv_path, output_path=fig_path)
nonlinear_breakdown.figure(input_path=csv_path, output_path=fig_path)
op_carbon.figure(input_path=csv_path, output_path=fig_path)
print('Figure generation complete.')