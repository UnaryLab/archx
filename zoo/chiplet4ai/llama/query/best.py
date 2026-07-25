import pandas as pd

# File paths
file_8b = 'zoo/chiplet4ai/llama/results/array_performance_metrics_8b_scientific.csv'
file_70b = 'zoo/chiplet4ai/llama/results/array_performance_metrics_70b.csv'

def get_best_rows(file_path):
    df = pd.read_csv(file_path)
    min_cycles = df['cycle_count'].min()
    best_rows = df[df['cycle_count'] == min_cycles]
    return best_rows

best_8b = get_best_rows(file_8b)
best_70b = get_best_rows(file_70b)

print('Best performing row(s) for 8b:')
print(best_8b)
print('\nBest performing row(s) for 70b:')
print(best_70b)
