from zoo.llm.results.query.utils import query_execution_time, compute_latency, load_yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

def query(input_path, output_path):
    vlp_list = ['mugi', 'carat']
    vlp_arch_dim_list = ['128x8', '256x8']
    gemm_vlp_latency_module = 'and_gate'

    mugi_subarch_list = ['vlp', 'lut']

    baseline_list = ['systolic', 'simd']
    baseline_arch_dim_list = ['16x16']
    simd_subarch_list = ['mac', 'figna']
    systolic_subarch_list = ['mac', 'figna', 'taylor', 'pwl']
    gemm_baseline_latency_module = 'multiplier'

    mugi_nonlinear_module = 'magnitude_register'
    baseline_nonlinear_module = 'accumulator_vector'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network = 'single_node'

    end_to_end_breakdown = pd.DataFrame()

    for arch in vlp_list + baseline_list:
       for arch_dim in (vlp_arch_dim_list if arch in vlp_list else baseline_arch_dim_list if arch in baseline_list else ['8x16x16'] if arch in ['tensor'] else ['']):
            for subarch in (systolic_subarch_list if arch in ['systolic'] else simd_subarch_list if arch in ['simd'] else mugi_subarch_list if arch in ['mugi'] else ['']):
                for model in model_list:

                    module = gemm_vlp_latency_module if arch in vlp_list else gemm_baseline_latency_module
                    nonlinear_module = mugi_nonlinear_module if arch in ['mugi'] else baseline_nonlinear_module
                    termination_path = 'full_termination' if arch == 'mugi' else ''
                    run_path = os.path.normpath(f'{input_path}{arch}/{network}/{subarch}/{arch_dim}/{model}/{max_seq_len}/{batch_size}/{termination_path}/')
                    yaml_dict = load_yaml(run_path)

                    event_graph = yaml_dict['event_graph']
                    metric_dict = yaml_dict['metric_dict']

                    proj_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='projection')
                    proj_latency_eff_dict = {
                        'latency': proj_execution_time
                    }

                    attn_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='attention')
                    attn_latency_eff_dict = {
                        'latency': attn_execution_time
                    }

                    ffn_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='ffn')
                    ffn_latency_eff_dict = {
                        'latency': ffn_execution_time
                    }

                    nonlinear_execution_time = query_execution_time(event_graph=event_graph, metric_dict=metric_dict, workload=model, event='nonlinear')
                    nonlinear_latency_eff_dict = {
                        'latency': nonlinear_execution_time
                    }

                    proj_metric_df = pd.DataFrame(proj_latency_eff_dict, index=[0])
                    proj_metric_df['layer'] = 'projection'
                    attn_metric_df = pd.DataFrame(attn_latency_eff_dict, index=[0])
                    attn_metric_df['layer'] = 'attention'
                    ffn_metric_df = pd.DataFrame(ffn_latency_eff_dict, index=[0])
                    ffn_metric_df['layer'] = 'ffn'
                    nonlinear_metric_df = pd.DataFrame(nonlinear_latency_eff_dict, index=[0])
                    nonlinear_metric_df['layer'] = 'nonlinear'


                    end_to_end_metric_df = pd.concat([proj_metric_df, attn_metric_df, ffn_metric_df, nonlinear_metric_df])
                    end_to_end_metric_df['arch'] = arch
                    end_to_end_metric_df['subarch'] = subarch
                    end_to_end_metric_df['arch_dim'] = arch_dim
                    end_to_end_metric_df['model'] = model

                    end_to_end_metric_df = end_to_end_metric_df.drop(columns=['flops', 'execution_time', 'power', 'energy'], errors='ignore')
                    end_to_end_breakdown = pd.concat([end_to_end_breakdown, end_to_end_metric_df], axis=0)

    end_to_end_breakdown.to_csv(output_path + 'end_to_end_latency_breakdown.csv', index=False)

    group_cols = [col for col in end_to_end_breakdown.columns if col not in ['latency', 'layer']]
    total_latency_df = end_to_end_breakdown.groupby(group_cols)['latency'].sum().reset_index()

    baseline_df = total_latency_df[
        (total_latency_df['arch'] == 'systolic') &
        (total_latency_df['subarch'] == 'mac') &
        (total_latency_df['arch_dim'] == '16x16') &
        (total_latency_df['model'] == 'llama_2_7b')
    ]

    
    end_to_end_breakdown['normalized_latency'] = end_to_end_breakdown['latency'] / baseline_df['latency'].values[0]

    end_to_end_breakdown.to_csv(output_path + 'end_to_end_latency_breakdown_norm.csv', index=False)