import taichi as ti

ti.init(arch=ti.cpu)

# Constants
chromosome_size = 194
max_population = 100

# Fields
population = ti.field(dtype=ti.int32, shape=(max_population, chromosome_size))
fitness = ti.field(dtype=ti.float32, shape=max_population)

# Selection Parameters
population_size = 100
offspring_size = 50
generations = 50
mutation_rate = 0.05
iterations = 10
tournament_size = 2

best_chromosomes = ti.field(dtype=ti.int32, shape=(generations, chromosome_size))
best_fitnesses = ti.field(dtype=ti.float32, shape=generations)

@ti.kernel
def init_population():
    for i in range(population_size):
        for j in range(chromosome_size):
            population[i, j] = (j + 1) % chromosome_size + 1  # Ensure proper cycling through chromosome values
        shuffle(i)
        calculate_fitness(i)

@ti.func
def shuffle(ind):
    # We will use a decrementing loop manually since Ti doesn't support three-argument range
    i = chromosome_size - 1
    while i > 0:
        j = ti.random(ti.int32) % (i + 1)
        temp = population[ind, i]
        population[ind, i] = population[ind, j]
        population[ind, j] = temp
        i -= 1  # Manually decrement i

@ti.func
def calculate_fitness(i: int):
    total_distance = 0.0
    for j in range(chromosome_size - 1):
        # Assume data points are somehow defined or fetched
        total_distance += abs(population[i, j] - population[i, j + 1])
    fitness[i] = total_distance

@ti.func
def select_two_random() -> ti.types.vector(2, ti.i32):
    idx1 = ti.random(ti.int32) % population_size
    idx2 = ti.random(ti.int32) % population_size
    while idx1 == idx2:
        idx2 = ti.random(ti.int32) % population_size
    return ti.Vector([idx1, idx2])

@ti.kernel
def run_selection_and_crossover():
    for i in range(iterations):
        for j in range(generations):
            best_fitness = float('inf')
            best_index = 0
            for k in range(population_size):
                if fitness[k] < best_fitness:
                    best_fitness = fitness[k]
                    best_index = k
            for m in range(chromosome_size):
                best_chromosomes[j, m] = population[best_index, m]
            best_fitnesses[j] = best_fitness
            # Genetic operations
            for k in range(offspring_size):
                indices = select_two_random()
                crossover_and_mutate(indices[0], indices[1], k)

@ti.func
def crossover_and_mutate(parent1_idx: int, parent2_idx: int, store_idx: int):
    crossover_point = ti.random(ti.int32) % chromosome_size
    for i in range(chromosome_size):
        if i < crossover_point:
            population[store_idx, i] = population[parent1_idx, i]
        else:
            population[store_idx, i] = population[parent2_idx, i]
        # Mutation chance
        if ti.random(ti.float32) < mutation_rate:
            swap_idx = ti.random(ti.int32) % chromosome_size
            temp = population[store_idx, i]
            population[store_idx, i] = population[store_idx, swap_idx]
            population[store_idx, swap_idx] = temp
    calculate_fitness(store_idx)

@ti.kernel
def print_best_individuals():
    for i in range(generations):
        print(f"Generation {i+1}: Best Fitness = {best_fitnesses[i]}")
        print("Chromosome:", end=" ")
        for j in range(chromosome_size):
            print(best_chromosomes[i, j], end=" ")
        print()

init_population()
run_selection_and_crossover()
print_best_individuals()
