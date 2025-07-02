from zoo.llm.results.query import area_power_breakdown, batch_size_breakdown, comprehensive_table, gemm_breakdown, noc_breakdown, nonlinear_breakdown

from loguru import logger
import os, sys

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

area_power_breakdown.query(input_path=input_path, output_path=csv_path)
print('Area and power breakdown query complete.')
batch_size_breakdown.query(input_path=input_path, output_path=csv_path)
print('Batch size breakdown query complete.')
comprehensive_table.query(input_path=input_path, output_path=csv_path)
print('Comprehensive table query complete.')
gemm_breakdown.query(input_path=input_path, output_path=csv_path)
print('GEMM breakdown query complete.')
noc_breakdown.query(input_path=input_path, output_path=csv_path)
print('NoC breakdown query complete.')
nonlinear_breakdown.query(input_path=input_path, output_path=csv_path)
print('Model query complete.')

area_power_breakdown.figure(input_path=csv_path, output_path=fig_path)
batch_size_breakdown.figure(input_path=csv_path, output_path=fig_path)
comprehensive_table.table(input_path=csv_path, output_path=table_path)
gemm_breakdown.figure(input_path=csv_path, output_path=fig_path)
noc_breakdown.figure(input_path=csv_path, output_path=fig_path)
nonlinear_breakdown.figure(input_path=csv_path, output_path=fig_path)
print('Figure generation complete.')