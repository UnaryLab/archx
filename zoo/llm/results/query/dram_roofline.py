from zoo.llm.results.query.utils import query_performance_gemm_metrics, query_performance_nonlinear_metrics, compute_throughput, load_yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

def total_memory(batch_size, d_model, d_vocab, d_hidden, seq_len, layers, heads, kv_heads):

    seq_len_sum = (seq_len + seq_len + 1) / 2

    q_proj_memory =  ((batch_size * d_model * 2) + (d_model * d_model / 2) + (batch_size * d_model * 2)) * seq_len * layers
    kv_proj_memory = ((batch_size * d_model * 2) + (d_model * d_model * (kv_heads / heads) / 2) + (batch_size * d_model * (kv_heads / heads) * 2)) * seq_len * layers
    qkt_memory = ((batch_size * heads * (d_model / heads) * 2 * seq_len) + (batch_size * kv_heads * seq_len_sum * (d_model / heads) / 2) + (batch_size * heads * seq_len_sum * 2)) * layers
    softmax_memory = (2 * batch_size * heads * seq_len_sum) * 2 * layers
    av_memory = ((batch_size * heads * seq_len_sum * 2) + (batch_size * seq_len_sum * kv_heads * (d_model/heads) / 2) + (batch_size * d_model * 2 * seq_len)) * layers
    up_proj_memory = ((batch_size * d_model * 2) + (d_model * d_hidden / 2) + (batch_size * d_hidden * 2)) * seq_len * layers
    gate_proj_memory = ((batch_size * d_model * 2) + (d_hidden * d_model / 2) + (batch_size * d_hidden * 2)) * seq_len * layers
    down_proj_memory = ((batch_size * d_hidden * 2) + (d_hidden * d_model / 2) + (batch_size * d_model * 2)) * seq_len * layers
    silu_memory = (batch_size * d_hidden * 2) * 2 * seq_len * layers
    output_memory = ((batch_size * d_model * 2) + (d_model * d_vocab / 2) + (batch_size * d_vocab * 2)) * seq_len

    gemm =  q_proj_memory + kv_proj_memory + qkt_memory + av_memory + up_proj_memory + gate_proj_memory + down_proj_memory + output_memory
    nonlinear = softmax_memory +  silu_memory
    return gemm, nonlinear

def query(input_path, output_path):
    vlp_list = ['mugi']
    vlp_arch_dim_list = ['256x8', '128x8', '64x8']
    gemm_vlp_latency_module = 'and_gate'

    mugi_subarch_list = ['vlp']

    baseline_list = ['systolic']
    baseline_arch_dim_list = ['8x8', '16x16']
    simd_subarch_list = ['mac', 'figna']
    systolic_subarch_list = ['mac']
    gemm_baseline_latency_module = 'multiplier'

    mugi_nonlinear_module = 'magnitude_register'
    baseline_nonlinear_module = 'accumulator_vector'

    model_list = ['llama_2_7b', 'llama_2_13b', 'llama_2_70b', 'llama_2_70b_GQA']
    max_seq_len = 'max_seq_len_4096'
    batch_size = 'batch_size_8'
    network = 'single_node'

    total_throughput_global = pd.DataFrame()

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
                    gemm_throughput = query_performance_gemm_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=model, module=module, event='gemm')

                    nonlinear_throughput = query_performance_nonlinear_metrics(event_graph=event_graph, metric_dict=metric_dict, workload=model, module=nonlinear_module, event='nonlinear')


                    total_throughput = {
                        'gemm_flops': gemm_throughput['flops'],
                        'nonlinear_flops': nonlinear_throughput['flops'],
                        'gemm_execution_time': gemm_throughput['execution_time'],
                        'nonlinear_execution_time': nonlinear_throughput['execution_time'],
                    }
                    total_throughput['gemm_throughput'] = total_throughput['gemm_flops'] / total_throughput['gemm_execution_time']
                    total_throughput['nonlinear_throughput'] = total_throughput['nonlinear_flops'] / total_throughput['nonlinear_execution_time']
                    total_throughput['total_throughput'] = (total_throughput['gemm_flops'] + total_throughput['nonlinear_flops']) / (total_throughput['gemm_execution_time'] + total_throughput['nonlinear_execution_time'])

                    if(model == 'llama_2_7b'):
                        memory_bytes_gemm, memory_bytes_nonlinear = total_memory(
                            batch_size=8,
                            d_model=4096,
                            d_vocab=32000,
                            d_hidden=11008,
                            seq_len=4096,
                            layers=32,
                            heads=32,
                            kv_heads=32
                        )
                    elif(model == 'llama_2_13b'):
                        memory_bytes_gemm, memory_bytes_nonlinear = total_memory(
                            batch_size=8,
                            d_model=5120,
                            d_vocab=32000,
                            d_hidden=13824,
                            seq_len=4096,
                            layers=40,
                            heads=40,
                            kv_heads=40
                        )
                    elif(model == 'llama_2_70b'):
                        memory_bytes_gemm, memory_bytes_nonlinear = total_memory(
                            batch_size=8,
                            d_model=8192,
                            d_vocab=32000,
                            d_hidden=28672,
                            seq_len=4096,
                            layers=80,
                            heads=64,
                            kv_heads=64
                        )
                    elif(model == 'llama_2_70b_GQA'):
                        memory_bytes_gemm, memory_bytes_nonlinear = total_memory(
                            batch_size=8,
                            d_model=8192,
                            d_vocab=32000,
                            d_hidden=28672,
                            seq_len=4096,
                            layers=80,
                            heads=64,
                            kv_heads=8
                        )

                    total_throughtput_df = pd.DataFrame(total_throughput, index=[0])
                    total_throughtput_df['model'] = model
                    total_throughtput_df['arch'] = arch
                    total_throughtput_df['arch_dim'] = arch_dim
                    total_throughtput_df['subarch'] = subarch
                    total_throughtput_df['memory_bytes_gemm'] = memory_bytes_gemm
                    total_throughtput_df['memory_bytes_nonlinear'] = memory_bytes_nonlinear
                    total_throughtput_df['oi_gemm'] = 10**9 * total_throughput['gemm_flops'] / memory_bytes_gemm
                    total_throughtput_df['oi_nonlinear'] = 10**9 * total_throughput['nonlinear_flops'] / memory_bytes_nonlinear
                    total_throughtput_df['oi_total'] = 10**9 * (total_throughput['gemm_flops'] + total_throughput['nonlinear_flops']) / (memory_bytes_gemm + memory_bytes_nonlinear)

                    total_throughput_global = pd.concat([total_throughput_global, total_throughtput_df], axis=0)

    total_throughput_global.to_csv(output_path + 'dram_roofline.csv', index=False)

def figure(input_path, output_path):
    arch_df = pd.read_csv(input_path + 'dram_roofline.csv')

    fig_width_pt = 240
    fig_width = fig_width_pt/72
    fig_height = fig_width/2.7

    fontsize = 6.5

    # Setup figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Roofline parameters
    bandwidth_gb_s = 256  # GB/s
    peak_gemm_throughput = {
        'Mugi 256': 2 * 256 * 4 / 10,     # GFLOPs
        'Mugi 128 / SA 16': 2 * 128 * 4 / 10, # GFLOPs  
        'Mugi 64 / SA 8': 2 * 64 * 4 / 10,     # GFLOPs
    }
    
    # Additional configuration dictionary for each line
    line_config = {
        'Mugi 256': {
            'color': '#000000',
            'linestyle': '--',
            'linewidth': 0.6,
            'alpha': 0.8,
            'arch_type': 'mugi'
        },
        'Mugi 128 / SA 16': {
            'color': "#000000",
            'linestyle': '--',
            'linewidth': 0.6,
            'alpha': 0.8,
            'arch_type': 'systolic'
        },
        'Mugi 64 / SA 8': {
            'color': "#000000",
            'linestyle': '--',
            'linewidth': 0.6,
            'alpha': 0.8,
            'arch_type': 'systolic'
        }
    }
    
    # Create operational intensity range (powers of 2)
    oi_range = np.logspace(-3, 9, 1000, base=2)  # From 2^-6 to 2^6 FLOPS/Byte
    
    # Calculate memory-bound performance (bandwidth line)
    memory_bound_gflops = oi_range * bandwidth_gb_s
    
    # Plot bandwidth line
    ax.loglog(oi_range, memory_bound_gflops, 'r-', linewidth=0.75, base=2)
    
    # Add bandwidth label to the line
    ax.text(1/3.5, bandwidth_gb_s * 0.34, f'{bandwidth_gb_s} GB/s', 
            color='red', fontsize=fontsize, fontweight='bold',
            ha='left', va='bottom', rotation=48)
    
    # Plot peak throughput lines for each architecture using line_config
    for arch_dim, peak_throughput in peak_gemm_throughput.items():
        config = line_config[arch_dim]
        
        # Calculate compute-bound performance (peak throughput line)
        compute_bound_gflops = np.full_like(oi_range, peak_throughput)
        
        # Calculate actual roofline (minimum of memory and compute bounds)
        roofline_gflops = np.minimum(memory_bound_gflops, compute_bound_gflops)
        
        # Plot peak throughput line
        ax.loglog(oi_range, compute_bound_gflops, 
                 linestyle=config['linestyle'], 
                 color=config['color'], 
                 linewidth=config['linewidth'], 
                 alpha=config['alpha'], base=2)
        
        # Plot actual roofline
        ax.loglog(oi_range, roofline_gflops, '-', 
                 color=config['color'], 
                 linewidth=config['linewidth'], 
                 alpha=0.7, base=2)
        
        # Add text annotation for peak throughput on the line
        ax.text(490, peak_throughput * 0.77, f'{arch_dim}', 
                color=config['color'], fontsize=fontsize-0.75, fontweight='bold',
                ha='right', va='bottom')
    
    # Create color mapping for arch + arch_dim combinations
    arch_dim_colors = {
        'Mugi 64': "#6dbfe6",
        'Mugi 128': "#FD6868",
        'Mugi 256': "#67C069",
        'SA 8': "#e4aa5f", 
        'SA 16': "#b880e5"
    }
    
    # Create marker shape mapping based on model
    model_markers = {
        '7B': 'o',      # circle
        '13B': 's',     # square
        '70B': '^',     # triangle up
        '70B GQA': 'D'  # diamond
    }
    
    # Plot scatter points with arch + arch_dim combinations and model-based markers
    for arch in arch_df['arch'].unique():
        for arch_dim in arch_df[arch_df['arch'] == arch]['arch_dim'].unique():
            for model in arch_df['model'].unique():
                subset = arch_df[(arch_df['arch'] == arch) & 
                                (arch_df['arch_dim'] == arch_dim) & 
                                (arch_df['model'] == model)]
                
                if not subset.empty:
                    arch_key = 'Mugi' if arch == 'mugi' else 'SA'
                    arch_dim_key = arch_dim.split('x')[0]
                    color_key = f"{arch_key} {arch_dim_key}"
                    color = arch_dim_colors.get(color_key)
                    model_key = model.split('_')[2].upper() + ' ' +  model.split('_')[3].upper() if 'GQA' in model else model.split('_')[2].upper()
                    marker = model_markers.get(model_key, 'o')
                    
                    ax.scatter(subset['oi_total'], subset['total_throughput'], 
                              color=color, 
                              marker=marker,
                              s=15, alpha=0.8, edgecolors='black', linewidth=0.25)
    
    # Create separate legend entries for architectures and models
    # Architecture legend entries
    arch_legend_elements = []
    for arch_dim, color in arch_dim_colors.items():
        arch_legend_elements.append(ax.scatter([], [], color=color, marker='o', s=10, alpha=0.8, 
                                             edgecolors='black', linewidth=.25, label=arch_dim))
    
    # Model legend entries  
    model_legend_elements = []
    for model, marker in model_markers.items():
        model_legend_elements.append(ax.scatter([], [], color='gray', marker=marker, s=10, alpha=0.8,
                                               edgecolors='black', linewidth=.25, label=model))
    
    # Set axis labels
    ax.set_xlabel('Operational Intensity', fontsize=fontsize-.25, labelpad=0.1)
    ax.set_ylabel('Throughput', fontsize=fontsize-.25, labelpad=2)
    
    # Set tick font sizes and make axes thinner
    ax.tick_params(axis='both', which='major', labelsize=fontsize-0.5, length=2, pad=1)
    
    # Make the box outline (spines) thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Set axis limits (powers of 2)
    ax.set_xlim(1/8, 512)  # 2^-6 to 2^6
    ax.set_ylim(28, 250)  # 2^0 to 2^14
    
    # Remove grid
    # ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add legend with only data points (architectures and models)
    ax.legend(fontsize=fontsize-1, ncol=2, loc='upper center', bbox_to_anchor=(0.4, .85), 
              columnspacing=0.1, handletextpad=0.1)
    
    # Reduce whitespace around the figure
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path + 'dram_roofline.png', bbox_inches='tight', dpi=1200, pad_inches=0.065)
    plt.savefig(output_path + 'dram_roofline.pdf', bbox_inches='tight', dpi=1200, pad_inches=0.065)
    plt.show()
