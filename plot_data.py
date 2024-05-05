import csv
import matplotlib.pyplot as plt

def read_time(data_file, offspring_check=False):
    population_data = {}
    offspring_variation = []
    with open(data_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            population_size = int(row['population_size'])
            generation = int(row['generations'])
            run_time = float(row['run_time']) / 1000

            # Store data in population_data
            if population_size not in population_data:
                population_data[population_size] = []
            population_data[population_size].append((generation, run_time))

            # Check for offspring variation
            if offspring_check and generation == 1000 and population_size == 500:
                offspring_variation.append(run_time)

    return population_data, offspring_variation

data_file_gpu = 'manual_results_gpu.csv'
data_file_cpu = 'manual_results_cpu.csv'

population_data_gpu, offspring_variation_gpu = read_time(data_file_gpu, True)
population_data_cpu, _ = read_time(data_file_cpu)

# Plot GPU and CPU runtimes on the same figures

# GPU Runtime vs Population Size (Generations: 10k) and CPU Runtime vs Population Size (Generations: 10k)
y_gpu_population = []
y_cpu_population = []
x_population = list(population_data_gpu.keys())
for i in population_data_gpu.values():
    for j in i:
        if j[0] == 10000:
            y_gpu_population.append(j[1])

plt.plot(x_population, y_gpu_population, marker='o')
plt.title('GPU Runtime vs Population Size (Generations: 10k)')
plt.xlabel('Population Size')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
# plt.show()


# Plot GPU runtime vs Generations (keeping population size constant at 1k)
x_generation = []
y_gpu_generation = []

for i in population_data_gpu.items():
    if i[0] == 1000:
        for j in i[1]:
            x_generation.append(j[0])
            y_gpu_generation.append(j[1])

# plt.plot(x_generation, y_gpu_generation, marker='o')
plt.title('GPU Runtime vs Generations (Population Size: 1k)')
plt.xlabel('Generations')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
# plt.show()

y_cpu_population = []
x_population = list(population_data_cpu.keys())
for i in population_data_cpu.values():
    for j in i:
        if j[0] == 10000:
            y_cpu_population.append(j[1])

plt.plot(x_population, y_gpu_population, label='GPU')
plt.plot(x_population, y_cpu_population, label='CPU')
plt.title('Runtime vs Population Size (Generations: 10k)')
plt.xlabel('Population Size')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.grid(True)
plt.savefig('runtime_vs_population_size.png')  # Save the figure
plt.show()


# GPU Runtime vs Generations (Population Size: 1k) and CPU Runtime vs Generations (Population Size: 1k)
x_gpu_generation = []
y_gpu_generation = []
x_cpu_generation = []
y_cpu_generation = []

for i in population_data_gpu.items():
    if i[0] == 1000:
        for j in i[1]:
            x_gpu_generation.append(j[0])
            y_gpu_generation.append(j[1])

for i in population_data_cpu.items():
    if i[0] == 1000:
        for j in i[1]:
            x_cpu_generation.append(j[0])
            y_cpu_generation.append(j[1])

# plt.plot(x_generation, y_cpu_generation, marker='o')
plt.plot(x_generation, y_gpu_generation, marker='o')
plt.plot(x_generation, y_cpu_generation, marker='o')
plt.title('Runtime vs Generations (Population Size: 1k)')
plt.xlabel('Generations')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.grid(True)
plt.show()
