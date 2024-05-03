# Generate plots for the data in results.csv
# The x-axis should be in logarithmic scale.
# Each run has 3 trials, so you should average the results for each configuration.

import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('results.csv')

# Plot 3 graphs with 2 subplots each for initialization time and run time
# Population size, offspring size, and generations are the x-axis
# The default configuration for testing is generations=100, population_size=50, offspring_size=20

# Plotting initialization time and run time as a function of population size
plt.figure(figsize=(20, 15))

# Only take the data for the default configuration
default_data = data[(data['population_size'] == 50) & (data['offspring_size'] == 20)]
# Since there are 3 trials for each configuration, we average the results
default_data = default_data.groupby(['generations'])

# calculate the mean of initialization time and run time
default_data = default_data.agg({'init_time': 'mean', 'run_time': 'mean'}).reset_index()

# Subplot for Initialization Time
plt.subplot(3, 2, 1)  # 3 rows, 2 columns, 1st subplot
plt.plot(default_data['generations'], default_data['init_time'], color='b', label='Init Time (ms)')
plt.title('Initialization Time vs Generations')
plt.xlabel('Generations')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Subplot for Run Time
plt.subplot(3, 2, 2)  # 3 rows, 2 columns, 2nd subplot
plt.plot(default_data['generations'], default_data['run_time'], color='r', label='Run Time (ms)')
plt.title('Run Time vs Generations')
plt.xlabel('Generations')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Only take the data for the default configuration
default_data = data[(data['generations'] == 100) & (data['offspring_size'] == 20)]
# Since there are 3 trials for each configuration, we average the results
default_data = default_data.groupby(['population_size'])

# calculate the mean of initialization time and run time
default_data = default_data.agg({'init_time': 'mean', 'run_time': 'mean'}).reset_index()

# Subplot for Initialization Time
plt.subplot(3, 2, 3)  # 3 rows, 2 columns, 3rd subplot
plt.plot(default_data['population_size'], default_data['init_time'], color='b', label='Init Time (ms)')
plt.title('Initialization Time vs Population Size')
plt.xlabel('Population Size')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Subplot for Run Time
plt.subplot(3, 2, 4)  # 3 rows, 2 columns, 4th subplot
plt.plot(default_data['population_size'], default_data['run_time'], color='r', label='Run Time (ms)')
plt.title('Run Time vs Population Size')
plt.xlabel('Population Size')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Only take the data for the default configuration
default_data = data[(data['generations'] == 100) & (data['population_size'] == 50)]
# Since there are 3 trials for each configuration, we average the results
default_data = default_data.groupby(['offspring_size'])

# calculate the mean of initialization time and run time
default_data = default_data.agg({'init_time': 'mean', 'run_time': 'mean'}).reset_index()

# Subplot for Initialization Time
plt.subplot(3, 2, 5)  # 3 rows, 2 columns, 5th subplot
plt.plot(default_data['offspring_size'], default_data['init_time'], color='b', label='Init Time (ms)')
plt.title('Initialization Time vs Offspring Size')
plt.xlabel('Offspring Size')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Subplot for Run Time
plt.subplot(3, 2, 6)  # 3 rows, 2 columns, 6th subplot
plt.plot(default_data['offspring_size'], default_data['run_time'], color='r', label='Run Time (ms)')
plt.title('Run Time vs Offspring Size')
plt.xlabel('Offspring Size')
plt.ylabel('Time (s)')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid(True)
plt.legend()

# Adjust layout to not overlap and show the plot
plt.tight_layout()

plt.show()