import pandas as pd
import matplotlib.pyplot as plt
import sys

# Funzione per parsare il file di output di nvprof
def parse_nvprof_output(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    metrics = {}
    kernel_name = None
    for i,line in enumerate(lines):
        if i == 3:
            continue
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
                if metric_name == 'gld_throughput' or metric_name == 'gst_throughput':
                    metric_value = metric_value[0:-4]
                    metric_value = float(metric_value) * 1024 * 1024
                elif metric_name == 'dram_read_throughput' or metric_name == 'dram_write_throughput':
                    metric_value = metric_value[0:-4]
                    metric_value = float(metric_value) * 1024 * 1024 * 1024
                elif metric_name == 'inst_per_warp':
                    metric_value = float(metric_value)
                else:
                    continue
                metrics[kernel_name][metric_name] = float(metric_value)
    
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
    peak_fp32_performance = 2.488e12  # 2.1 TFLOPS in FLOP/s
    peak_memory_bandwidth = 112.1e9  # 112 GB/s in B/s
    
    # Generate the Roofline plot
    for i, row in data.iterrows():
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

    # Set x and y axis limits
    plt.xlim(left=1)  # Set left limit of x-axis
    plt.ylim(bottom=1)  # Set bottom limit of y-axis

    plt.show()


# Main
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python roofline.py <file_path>")
        sys.exit(1)
    file_path = './metrics.txt'  # Cambia con il percorso corretto del file
    metrics = parse_nvprof_output(file_path)
    data = calculate_metrics(metrics)
    plot_roofline(data)



