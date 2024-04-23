import taichi as ti
import numpy as np
import random

ti.init(arch=ti.cpu)  # Use GPU for computation

# Constants
NUM_CITIES = 194
POP_SIZE = 100
TOP_K = 20  # Top K individuals for truncation selection
MUTATION_RATE = 0.15
NUM_GENERATIONS = 500

# TSP Data: distances between cities. Placeholder, replace with actual data
distances = ti.field(dtype=ti.f32, shape=(NUM_CITIES, NUM_CITIES))

# Population and fitness
population = ti.field(dtype=ti.i32, shape=(POP_SIZE, NUM_CITIES))

arr = ti.field(dtype=ti.i32, shape=(NUM_CITIES))
# print(arr)


fitness = ti.field(dtype=ti.f32, shape=POP_SIZE)

@ti.func
def compute_distance(tour):
    # print(tour)
    dist = 0.0
    for i in range(NUM_CITIES - 1):
        dist += distances[tour[i], tour[i + 1]]
    dist += distances[tour[NUM_CITIES - 1], tour[0]]  # Return to the starting city
    return dist

@ti.kernel
def evaluate():
    for i in range(POP_SIZE):
        local_arr = ti.Vector([0] * NUM_CITIES)
        for j in range(NUM_CITIES):
            local_arr[j] = population[i, j]
        fitness[i] = compute_distance(local_arr)

@ti.kernel
def initialize_population():
    for i in range(POP_SIZE):
        for j in range(NUM_CITIES):
            population[i, j] = j
        for j in range(NUM_CITIES):
            k = ti.random(int) % NUM_CITIES
            temp = population[i, j]
            population[i, j] = population[i, k]
            population[i, k] = temp

def crossover(parent1, parent2):
    child = [-1] * NUM_CITIES
    start, end = sorted([random.randint(0, NUM_CITIES - 1) for _ in range(2)])
    child[start:end] = parent1[start:end]
    filled = set(parent1[start:end])
    pos = end
    for city in parent2:
        if city not in filled:
            if pos >= NUM_CITIES:
                pos = 0
            child[pos] = city
            pos += 1
    return child

@ti.func
def mutate(tour):
    for i in range(NUM_CITIES):
        if ti.random() < MUTATION_RATE:
            j = ti.random(int) % NUM_CITIES
            temp = tour[i]
            tour[i] = tour[j]
            tour[j] = temp

@ti.kernel
def generate_new_population():
    parents_indices = fitness.argsort()[:TOP_K]
    for i in range(POP_SIZE):
        parent1, parent2 = [population[parents_indices[random.randint(0, TOP_K - 1)]] for _ in range(2)]
        child = crossover(parent1, parent2)
        for j in range(NUM_CITIES):
            population[i, j] = child[j]
        mutate(population[i])

def run():
    initialize_population()
    for gen in range(NUM_GENERATIONS):
        evaluate()
        generate_new_population()
        if gen % 10 == 0:
            print(f'Generation {gen}, Best fitness: {fitness.min()}')


if __name__ == "__main__":
    run()
