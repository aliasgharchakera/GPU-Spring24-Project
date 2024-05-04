import taichi as ti
import re
import time
from run import read_file
import sys
import read_timing

original_stdout = sys.stdout

if len(sys.argv) != 4:
    print("Usage: python script.py <generations> <population_size> <offspring_size>")
    sys.exit(1)
try:
    generations = [int(sys.argv[1])]
    population_sizes = [int(sys.argv[2])]
    offsprings_sizes = [float(sys.argv[3])]
except ValueError:
    print("Error: All parameters should be integers or floats.")
    sys.exit(1)

ti.init(arch=ti.cpu, kernel_profiler=True)

def copy_array(arr, field):
    for i in range(len(arr)):
        field[i] = arr[i]

# @ti.kernel
def init_population():
    print("kernel launched")
    for i in range(population_size):
        for j in range(chromosome_size):
            population[i, j] = j + 1
        shuffle(i)
        calculate_fitness(i)

# @ti.func
def shuffle(ind):
    # We will use a decrementing loop manually since Ti doesn't support three-argument range
    i = chromosome_size - 1
    while i > 0:
        j = ti.random(ti.int32) % (i + 1)
        temp = population[ind, i]
        population[ind, i] = population[ind, j]
        population[ind, j] = temp
        i -= 1  # Manually decrement i

# @ti.func
def calculate_fitness(i: int):
    total_distance = 0.0
    for j in range(chromosome_size - 1):
        # Ensure population indices are correctly handled
        index_current = int(population[i, j] - 1)  # Convert to 0-based index
        index_next = int(population[i, j + 1] - 1)  # Convert to 0-based index

        dx = coords_xt[index_current] - coords_xt[index_next]
        dy = coords_yt[index_current] - coords_yt[index_next]
        total_distance += ti.sqrt(dx * dx + dy * dy)

    # Close the loop: Calculate distance from last to first city
    index_last = population[i, chromosome_size - 1] - 1
    index_first = population[i, 0] - 1
    dx = coords_xt[index_last] - coords_xt[index_first]
    dy = coords_yt[index_last] - coords_yt[index_first]
    total_distance += ti.sqrt(dx * dx + dy * dy)

    fitness[i] = int(total_distance)

# @ti.func
def select_two_truncation() -> ti.types.vector(2, ti.i32):
    idx1 = 0
    idx2 = 1
    for i in range(2, population_size):
        if fitness[i] > fitness[idx1]:
            idx1 = i
        elif fitness[i] > fitness[idx2]:
            idx2 = i
    # print(idx1,idx2)
    return ti.Vector([idx1, idx2])

# @ti.func
def truncation_selection():
    indexes = ti.Vector([0]*offspring_size)
    minimum = -1
    # print(indexes)
    for i in range(offspring_size):
        min_index = find_min_index(fitness,minimum)
        # print(i,"th time",min_index)
        # print(idx)
        indexes[i] = min_index
        minimum = fitness[min_index]
    # print(indexes)
    select_survivors(indexes)

# @ti.func
def find_min_index(fitness: ti.template(), minimum: int):
    # print("minimum is",minimum)
    min_idx = 0
    y = 1000000
    for i in range(fitness.shape[0]):
        # print("checking for ",i,fitness[i])
        if fitness[i] < y and fitness[i] > minimum:
            # print("haan bhai true che")
            min_idx = i
            y = fitness[i]
    return min_idx

# @ti.func
def select_two_random() -> ti.types.vector(2, ti.i32):
    idx1 = ti.random(ti.int32) % population_size
    idx2 = ti.random(ti.int32) % population_size
    while idx1 == idx2:
        idx2 = ti.random(ti.int32) % population_size
    return ti.Vector([idx1, idx2])

# @ti.func
def random_selection():
    indexes = ti.Vector([0]*offspring_size)
    # print(indexes)
    for i in range(offspring_size):
        indexes[i] = ti.random(ti.int32) % population_size
    # print(indexes)
    select_survivors(indexes)

# @ti.func
def select_survivors(indexes:ti.template()):
    for i in range(len(indexes)):
        for j in range(chromosome_size):
            population[indexes[i], j] = offsprings[i, j]

# @ti.kernel
def run_selection_and_crossover():
    print("selection kernel launched")
    for _ in range(1):
        # pass
        for j in range(generation):
            # pass
            best_fitness = float('inf')
            best_index = 0
            # TODO: make the best fitness computation on CPU 
            for k in range(population_size):
                if fitness[k] < best_fitness:
                    best_fitness = fitness[k]
                    best_index = k
            for m in range(chromosome_size):
                best_chromosomes[j, m] = population[best_index, m]
            best_fitnesses[j] = int(best_fitness)
            # Genetic operations
            for k in range(offspring_size):
                indices = select_two_truncation()
                crossover_and_mutate(indices[0], indices[1], k)
            truncation_selection()
            # print(corpses)

# @ti.func
def crossover_and_mutate(parent1_idx: int, parent2_idx: int, store_idx: int):
    crossover_point = ti.random(ti.int32) % chromosome_size
    for i in range(chromosome_size):
        if i < crossover_point:
            offsprings[store_idx, i] = population[parent1_idx, i]
        else:
            offsprings[store_idx, i] = population[parent2_idx, i]
        # Mutation chance
        if ti.random(ti.float32) < mutation_rate:
            swap_idx = ti.random(ti.int32) % chromosome_size
            temp = population[store_idx, i]
            population[store_idx, i] = population[store_idx, swap_idx]
            population[store_idx, swap_idx] = temp
    calculate_fitness(store_idx)

# @ti.kernel
def print_best_individuals():
    for i in range(generations / 50):
        print(f"Generation {i*50+1}: Best Fitness = {best_fitnesses[i*50]}")
        # print("Chromosome:", end=" ")
        # for j in range(chromosome_size):
        #     print(best_chromosomes[i, j], end=" ")
        # print()
        
        
# Selection Parameters
mutation_rate = 0.05
tournament_size = 2
# Try different values for population_size, offspring_size, generations
# generations = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
# generations = [200]
# population_sizes = [10, 20, 50, 100, 200, 500, 1000]
# population_sizes = [100]
# offsprings_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# offsprings_sizes = [0.2]
selection_scheme = ["truncation", "random"]
cities = ["qa194.tsp"]

results = []



for city in cities:
    coords_x, coords_y, chromosome_size = read_file(city)
    for generation in generations:
        print("for generation",generation)
        for population_size in population_sizes:
            for offspring_ratio in offsprings_sizes:
                offspring_size = int(population_size * offspring_ratio)
                with open('profiling_info.txt', 'w') as f:
                    # Redirect standard output to the file
                    sys.stdout = f
                    for i in range(3):
                        coords_xt = ti.field(dtype=ti.float32, shape=chromosome_size)
                        coords_yt = ti.field(dtype=ti.float32, shape=chromosome_size)
                        copy_array(coords_x, coords_xt)
                        copy_array(coords_y, coords_yt)
                        # Fields
                        population = ti.field(dtype=ti.int32, shape=(population_size, chromosome_size))
                        offsprings = ti.field(dtype=ti.int32, shape=(offspring_size, chromosome_size))
                        fitness = ti.field(dtype=ti.int32, shape=population_size)


                        # Best individuals tracking
                        best_chromosomes = ti.field(dtype=ti.int32, shape=(generation, chromosome_size))
                        best_fitnesses = ti.field(dtype=ti.int32, shape=generation)


                        # start = time.time()
                        init_population()
                        ti.sync()
                        ti.profiler.print_kernel_profiler_info()
                        ti.profiler.clear_kernel_profiler_info()
                        # print("here")
                        # lap = time.time()
                        run_selection_and_crossover()
                        ti.sync()
                        ti.profiler.print_kernel_profiler_info()
                        ti.profiler.clear_kernel_profiler_info()
                        # end = time.time()
                init_time,run_time = read_timing.extract_execution_times('profiling_info.txt')
                # print(init_time,run_time)
                results.append([city, generation, population_size, offspring_size, 'Truncation', (init_time)*1000, (run_time)*1000])    

                sys.stdout = original_stdout
    
# write the results to a file
with open("manual_results_cpu.csv", "a") as f:
    # f.write("city,generations,population_size,offspring_size,scheme,init_time,run_time\n")
    for result in results:
        f.write(",".join(map(str, result)) + "\n")


