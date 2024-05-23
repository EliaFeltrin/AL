import pandas as pd
import matplotlib.pyplot as plt
import re

import pandas as pd
import matplotlib.pyplot as plt

# Funzione per parsare il file di output di nvprof
def parse_nvprof_output(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    metrics = {}
    kernel_name = None
    for line in lines:
        if line.startswith("==") or line.startswith("Device"):
            continue
        if "Kernel" in line:
            kernel_name = line.split(":")[1].strip()
            metrics[kernel_name] = {}
        else:
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[1]
                metric_value = parts[-1]
                metric_value = re.sub(r'\D,', '', metric_value)
                metrics[kernel_name][metric_name] = metric_value
    
    return metrics

# Funzione per calcolare le metriche necessarie per il grafico Roofline
def calculate_metrics(metrics):
    data = []
    for kernel, values in metrics.items():
        gld_throughput = values.get('gld_throughput', 0)
        gst_throughput = values.get('gst_throughput', 0)
        dram_read_throughput = values.get('dram_read_throughput', 0)
        dram_write_throughput = values.get('dram_write_throughput', 0)
        inst_per_warp = values.get('inst_per_warp', 0)
        
        # Calcolo del throughput computazionale (assumendo 32 threads per warp)
        compute_throughput = inst_per_warp * 32
        
        # Calcolo del throughput di memoria
        memory_throughput = max(gld_throughput, gst_throughput, dram_read_throughput, dram_write_throughput)
        
        data.append({
            'Kernel': kernel,
            'Compute Throughput': compute_throughput,
            'Memory Throughput': memory_throughput
        })
    
    return pd.DataFrame(data)

# Function to generate the Roofline plot
def plot_roofline(data):
    plt.figure(figsize=(10, 6))
    
    # Peak performance and memory bandwidth (values for GTX 1050 Ti)
    peak_fp32_performance = 2.1e12  # 2.1 TFLOPS in FLOP/s
    peak_memory_bandwidth = 112e9  # 112 GB/s in B/s
    
    # Generate the Roofline plot
    for i, row in data.iterrows():
        print(row)
        plt.scatter(row['Memory Throughput'], row['Compute Throughput'], label=row['Kernel'])
    
    # Adding the Roofline
    x = [0, peak_memory_bandwidth]  # Extend the x range
    y = [xi * (peak_fp32_performance / peak_memory_bandwidth) for xi in x]
    plt.plot(x, y, label='Roofline', linestyle='--')
    
    # Plot peak performance line
    plt.axhline(y=peak_fp32_performance, color='r', linestyle='-', label='Peak FP32 Performance')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Memory Throughput (Bytes/s)')
    plt.ylabel('Compute Throughput (Operations/s)')
    plt.title('Roofline Model')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


# Main
file_path = './metrics.txt'  # Cambia con il percorso corretto del file
metrics = parse_nvprof_output(file_path)
data = calculate_metrics(metrics)
plot_roofline(data)



