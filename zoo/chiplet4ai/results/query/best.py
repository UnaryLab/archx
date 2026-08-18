import pandas as pd

df = pd.read_csv('zoo/chiplet4ai/results/csv/array_performance_metrics.csv')

for model in ['llama_3_1_8b', 'llama_3_1_70b', 'llama_3_1_405b', 'deepseek_v4']:
    df_model = df[df['model'] == model]
    min_cycles = df_model['cycle_count'].min()
    best_rows = df_model[df_model['cycle_count'] == min_cycles]
    print(f'Best performing row(s) for {model}:')
    print(best_rows)
    print()
