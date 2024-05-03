import pandas as pd

import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('results.csv')


# Plotting initialization time and run time as a function of population size
plt.figure(figsize=(10, 5))

# Subplot for Initialization Time
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(data['population_size'], data['init_time'], color='b', label='Init Time (ms)')
plt.title('Initialization Time vs Population Size')
plt.xlabel('Population Size')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Subplot for Run Time
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(data['population_size'], data['run_time'], color='r', label='Run Time (ms)')
plt.title('Run Time vs Population Size')
plt.xlabel('Population Size')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Adjust layout to not overlap and show the plot
plt.tight_layout()
plt.show()
