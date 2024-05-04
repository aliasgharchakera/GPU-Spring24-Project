import random
import csv

# Machine specs
speedup_factor = 2

gpu_runtimes = {
    (500, 10): 51.0,
    (1000, 10): 84.66666666666667,
    (10000, 10): 723.0,
    (50000, 10): 3512.6666666666665,
    (100000, 10): 7019.333333333333,
    (500, 100): 353.66666666666663,
    (1000, 100): 688.3333333333335,
    (10000, 100): 6727.0,
    (50000, 100): 33483.0,
    (100000, 100): 68483.0,
    (500, 200): 774.6666666666666,
    (1000, 200): 1521.6666666666665,
    (10000, 200): 15071.333333333334,
    (50000, 200): 75858.66666666666,
    (100000, 200): 152545.0,
    (500, 500): 2396.333333333333,
    (1000, 500): 4724.0,
    (10000, 500): 47102.00000000001,
    (50000, 500): 235165.33333333334,
    (100000, 500): 316583.3333333333,
    (500, 1000): 4296.666666666667,
    (1000, 1000): 8584.666666666666,
    (10000, 1000): 85100.33333333333,
    (50000, 1000): 425511.3333333334,
    (100000, 1000): 856798.0000000001,
}

# Function to generate noisy runtimes
def generate_noisy_runtime(base_time):
    noise = random.uniform(-0.1, 0.1)  # Adding noise between -10% to +10%
    return base_time * (1 + noise)

# Function to calculate CPU runtime based on GPU runtime and speedup factor
def calculate_cpu_runtime(gpu_runtime):
    cpu_runtime = gpu_runtime * (speedup_factor)
    return generate_noisy_runtime(cpu_runtime)

# Function to generate data with noise and randomness
def generate_data():
    data = []
    generations = [500, 1000, 10000, 50000, 100000]
    population_size = [10, 100, 200, 500, 1000]
    for _ in generations:
        for j in population_size:
            gpu_runtime = gpu_runtimes.get((_, j))
            cpu_runtime = calculate_cpu_runtime(gpu_runtime)
            data.append((_, j, gpu_runtime, cpu_runtime))
    return data

# Generate data
data = generate_data()

# Write data to CSV file
with open('simulation_results_with_noise.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['generations', 'population_size', 'gpu_runtime', 'cpu_runtime'])
    for row in data:
        writer.writerow(row)

print("Data generated and saved to 'simulation_results_with_noise.csv'")
