import os
import pandas as pd

# Directory contenente i file
directory = '/path/to/directory'

# Configurazioni in ordine
configurations = [
    'without_feasible', 'shared_mem', 'const_mem', 'no_mult', 'coarsening', 'argmin_rec', 'main'
]

# Kernels da analizzare
kernels = ['brute_force', 'brute_force_AL', 'reduce_argmin']

# Funzione per estrarre i dati dai file di eventi
def extract_event_data(file_path):
    data = {}
    current_kernel = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            for kernel in kernels:
                if kernel in line:
                    current_kernel = kernel
                    data[current_kernel] = {}
            if current_kernel:
                if 'branch' in line:
                    data[current_kernel]['Branch Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'warp_execution_efficiency' in line:
                    data[current_kernel]['Warp Execution Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'gld_efficiency' in line:
                    data[current_kernel]['Global Load Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'shared_efficiency' in line:
                    data[current_kernel]['Shared Memory Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'stall_exec_dependency' in line:
                    data[current_kernel]['Stalls Execution Dependency'] = float(line.split()[-1].strip('%'))
                elif 'stall_memory_dependency' in line:
                    data[current_kernel]['Stalls Memory Dependency'] = float(line.split()[-1].strip('%'))
                elif 'stall_other' in line:
                    data[current_kernel]['Stalls Other'] = float(line.split()[-1].strip('%'))
                elif 'stall_inst_fetch' in line:
                    data[current_kernel]['Stalls Fetch'] = float(line.split()[-1].strip('%'))
    return data

# Funzione per estrarre i dati dai file di metriche
def extract_metric_data(file_path):
    data = {}
    current_kernel = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            for kernel in kernels:
                if kernel in line:
                    current_kernel = kernel
                    data[current_kernel] = {}
            if current_kernel:
                if 'branch_efficiency' in line:
                    data[current_kernel]['Branch Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'warp_execution_efficiency' in line:
                    data[current_kernel]['Warp Execution Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'gld_efficiency' in line:
                    data[current_kernel]['Global Load Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'shared_efficiency' in line:
                    data[current_kernel]['Shared Memory Efficiency'] = float(line.split()[-1].strip('%'))
                elif 'stall_exec_dependency' in line:
                    data[current_kernel]['Stalls Execution Dependency'] = float(line.split()[-1].strip('%'))
                elif 'stall_memory_dependency' in line:
                    data[current_kernel]['Stalls Memory Dependency'] = float(line.split()[-1].strip('%'))
                elif 'stall_other' in line:
                    data[current_kernel]['Stalls Other'] = float(line.split()[-1].strip('%'))
                elif 'stall_inst_fetch' in line:
                    data[current_kernel]['Stalls Fetch'] = float(line.split()[-1].strip('%'))
    return data

# Struttura della tabella
columns = [
    'Configuration', 'Kernel', 'Branch Efficiency', 'Warp Execution Efficiency', 'Global Load Efficiency',
    'Shared Memory Efficiency', 'Stalls Execution Dependency', 'Stalls Memory Dependency',
    'Stalls Other', 'Stalls Fetch'
]
data = []

# Processa i file e crea la tabella
for config in configurations:
    event_file = os.path.join(directory, f"{config}_events.txt")
    metric_file = os.path.join(directory, f"{config}_metrics.txt")
    
    event_data = extract_event_data(event_file) if os.path.exists(event_file) else {}
    metric_data = extract_metric_data(metric_file) if os.path.exists(metric_file) else {}
    
    for kernel in kernels:
        combined_data = {**event_data.get(kernel, {}), **metric_data.get(kernel, {})}
        combined_data['Configuration'] = config
        combined_data['Kernel'] = kernel
        data.append(combined_data)

# Crea il DataFrame
df = pd.DataFrame(data, columns=columns)

# Salva il DataFrame in un file CSV
df.to_csv('kernel_performance_summary.csv', index=False)

# Stampa il DataFrame
print(df)
