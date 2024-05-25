import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory containing the CSV files
folder_path = 'coarse_test'

# Lists to store the extracted values
N_values = []
K_values = []
avg_brute_force_values = []
avg_brute_force_AL_values = []

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Extract N and K from the file name
        N, K = map(int, file_name.replace('.csv', '').split('_'))
        
        # Import the CSV file, ignoring the first three lines
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path, skiprows=3)
        
        # Extract Avg values for brute_force_AL and brute_force
        brute_force_AL_avg = data[data['Name'].str.contains('brute_force_AL', na=False)]
        brute_force_avg = data[data['Name'].str.contains('brute_force\(', na=False)]
        
        # Calculate average values
        avg_brute_force_AL = brute_force_AL_avg.iloc[:, 1].astype(float).mean()
        avg_brute_force = brute_force_avg.iloc[:, 1].astype(float).mean()
        
        # Append the values to the lists
        N_values.append(N)
        K_values.append(K)
        avg_brute_force_values.append(avg_brute_force)
        avg_brute_force_AL_values.append(avg_brute_force_AL)

# Creating 3D plots
fig = plt.figure(figsize=(12, 6))

# Plot for brute_force
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(N_values, K_values, avg_brute_force_values, c='r', marker='o')
ax1.set_xlabel('N')
ax1.set_xlim(4, 23)
ax1.set_xticks(np.arange(4, 24, 2))
ax1.set_ylabel('K')
ax1.set_ylim(0, 10)
ax1.set_zlabel('Avg brute_force')
ax1.set_title('Avg brute_force vs N and K')

# Plot for brute_force_AL
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(N_values, K_values, avg_brute_force_AL_values, c='b', marker='^')
ax2.set_xlabel('N')
ax2.set_xlim(4, 23)
ax2.set_xticks(np.arange(4, 24, 2))
ax2.set_ylabel('K')
ax2.set_ylim(0, 10)
ax2.set_zlabel('Avg brute_force_AL')
ax2.set_title('Avg brute_force_AL vs N and K')

plt.show()
