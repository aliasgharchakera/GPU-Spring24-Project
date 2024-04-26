import taichi as ti

ti.init(arch=ti.gpu)  # Initialize Taichi to use the GPU

# Assuming the chromosome is represented as a series of integers
chromosome_size = 194  # Placeholder for chromosome length
max_population = 100   # Maximum population size

# Taichi fields
population = ti.field(dtype=ti.int32, shape=(max_population, chromosome_size))
fitness = ti.field(dtype=ti.float32, shape=max_population)
data = ti.Vector.field(2, dtype=ti.float32, shape=chromosome_size)  # Assuming data points are 2D vectors

# Constants for selection
tournament_size = ti.field(dtype=ti.int32, shape=())
population_size = ti.field(dtype=ti.int32, shape=())
offspring_size = ti.field(dtype=ti.int32, shape=())
generations = ti.field(dtype=ti.int32, shape=())
mutation_rate = ti.field(dtype=ti.float32, shape=())
iterations = ti.field(dtype=ti.int32, shape=())

@ti.func
def calculate_fitness(i: ti.template()):
    # Dummy fitness calculation based on sum of chromosome values
    total_distance = 0.0
    for j in range(chromosome_size - 1):
        total_distance += ti.sqrt((data[population[i, j]][0] - data[population[i, j + 1]][0])**2 +
                                  (data[population[i, j]][1] - data[population[i, j + 1]][1])**2)
    total_distance += ti.sqrt((data[population[i, -1]][0] - data[population[i, 0]][0])**2 +
                                (data[population[i, -1]][1] - data[population[i, 0]][1])**2)
    fitness[i] = total_distance
    return total_distance

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

@ti.kernel
def init_population():
    for i in range(population_size[None]):
        for j in range(chromosome_size):
            population[i, j] = j + 1  # Initialize chromosomes in order
        shuffle(i)  # Shuffle the chromosomes
        calculate_fitness(i)

@ti.func
def select_two_random():
    # Randomly select two individuals, for demonstration
    i = ti.random(ti.int32) % population_size[None]
    j = ti.random(ti.int32) % population_size[None]
    while i == j:
        j = ti.random(ti.int32) % population_size[None]
    return i, j

@ti.func
def crossover(parent1: ti.template(), parent2: ti.template()):
    # Perform crossover to create a new chromosome from two parents
    crossover_point = ti.random(ti.int32) % (chromosome_size // 2)
    crossover_point_2 = ti.random(ti.int32) % (chromosome_size // 2) + chromosome_size // 2

    # Create a new chromosome by combining parts of both parents
    new_chromosome1 = ti.Vector([0] * chromosome_size, dt=ti.int32)
    new_chromosome2 = ti.Vector([0] * chromosome_size, dt=ti.int32)

    for i in range(crossover_point):
        new_chromosome1[i] = parent1[i]
        new_chromosome2[i] = parent2[i]

    for i in range(crossover_point, crossover_point_2):
        new_chromosome1[i] = parent2[i]
        new_chromosome2[i] = parent1[i]

    for i in range(crossover_point_2, chromosome_size):
        new_chromosome1[i] = parent1[i]
        new_chromosome2[i] = parent2[i]

    return new_chromosome1, new_chromosome2

@ti.func
def mutate(chromosome: ti.template()):
    # Perform mutation by swapping two cities in the chromosome based on mutation rate
    new_chromosome = ti.Vector([0] * chromosome_size, dt=ti.int32)
    for i in range(chromosome_size):
        new_chromosome[i] = chromosome[i]

    r = ti.random(ti.float32)
    if r < mutation_rate[None]:
        mutation_point = ti.random(ti.int32) % chromosome_size
        mutation_point_2 = ti.random(ti.int32) % chromosome_size
        new_chromosome[mutation_point], new_chromosome[mutation_point_2] = new_chromosome[mutation_point_2], new_chromosome[mutation_point]

    return new_chromosome

@ti.func
def random_chromosome():
    # Generate random chromosome from TSP set
    solution = ti.Vector([0] * chromosome_size, dt=ti.int32)
    for i in range(chromosome_size):
        solution[i] = i + 1
    ti.random.shuffle(solution)
    return solution

@ti.func
def insert_missing(chromosome: ti.template(), new_chromosome: ti.template()):
    missing = ti.Vector([0] * chromosome_size, dt=ti.int32)
    for i in range(chromosome_size):
        missing[i] = i + 1
    for i in range(chromosome_size):
        if missing[i] not in new_chromosome:
            new_chromosome.append(missing[i])
            
@ti.kernel
def run_evolution():
    for i in range(iterations[None]):
        for j in range(generations[None]):
            for k in range(offspring_size[None]/2):
                idx1, idx2 = select_two_random()
                parents = population[idx1], population[idx2]
                offsprings = crossover(parents[0], parents[1])
                population[k] = mutate(offsprings[0])
                population[k + 1] = mutate(offsprings[1])
                population[k + 2] = offsprings[0]
                population[k + 3] = offsprings[1]
            top_solution_generation = min(population, key=lambda x: fitness[x])
            print("Generation: ", j + 1)
            print("Top solution for this iteration: ", fitness[top_solution_generation])


# Use global variables for configuration
population_size[None] = 100
offspring_size[None] = 50
generations[None] = 50
mutation_rate[None] = 0.05
iterations[None] = 10
tournament_size[None] = 2

init_population()
run_evolution()
