import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

df = pd.read_csv('zoo/chiplet4ai/results/csv/array_performance_metrics.csv')
df_8b = df[df['model'] == 'llama_3_1_8b']
df_70b = df[df['model'] == 'llama_3_1_70b']
df_405b = df[df['model'] == 'llama_3_1_405b']
df_dsv4 = df[df['model'] == 'deepseek_v4']
