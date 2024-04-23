import taichi as ti
import numpy as np
import random

ti.init(arch=ti.cpu)  # Initialize Taichi to use the CPU

#constants
POPULATION_SIZE = 10
GENOME_LENGTH = 10
MUTATION_RATE = 0.01
GENERATIONS = 1
OFFSPRINGS = 2

population = ti.field(dtype=ti.i32, shape=(POPULATION_SIZE, GENOME_LENGTH))
fitness = ti.field(dtype=ti.f32, shape=POPULATION_SIZE)

parents = ti.field(dtype=ti.i32, shape=(2))
offsprings = ti.field(dtype=ti.i32, shape=(OFFSPRINGS))

# Example usage:
city_coords_array = np.array([
    [0, 0],
    [1, 2],
    [3, 1],
    [2, 4],
    [5, 2],
    [4, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14]
])

city_coords = ti.Vector.field(2, dtype=float, shape=len(city_coords_array))
for i in range(len(city_coords_array)):
    city_coords[i] = city_coords_array[i]

@ti.func
def euclidean_distance(x1, y1, x2, y2):
    return ti.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@ti.kernel
def calculate_fitness(city_coords: ti.template(), chromosome: ti.template()) -> ti.f32:
    fitness = 0.0
    num_cities = chromosome.shape[0]
    for i in range(num_cities - 1):
        current_city_index = chromosome[i]
        next_city_index = chromosome[i + 1]
        x1, y1 = city_coords[current_city_index]
        x2, y2 = city_coords[next_city_index]
        fitness += euclidean_distance(x1, y1, x2, y2)
    # Add distance from last city back to the starting city
    x1, y1 = city_coords[chromosome[num_cities - 1]]
    x2, y2 = city_coords[chromosome[0]]
    fitness += euclidean_distance(x1, y1, x2, y2)
    return fitness

@ti.kernel
def truncation(values: ti.template(), top: ti.template()):
    n = values.shape[0]
    dummy = [0] * top.shape[0]

    # Find top values
    for i in range(n):
        val = values[i]
        if val > dummy[0]:
            dummy[1] = dummy[0]
            top[1] = top[0]
            dummy[0] = val
            top[0] = i
        elif val > dummy[1]:
            dummy[1] = val
            top[1] = i

def initialize_population():
    for i in range(POPULATION_SIZE):
        arr = [i for i in range(GENOME_LENGTH)]
        random.shuffle(arr)
        for j in range(GENOME_LENGTH):
            population[i, j] = arr[j]
        # population[i, j] = int(random.random() * 10)

def run():
    initialize_population()
    for generation in range(GENERATIONS):
        # print(population)
        for chr in range(POPULATION_SIZE):
            local_arr = ti.field(dtype=ti.i32, shape=(GENOME_LENGTH))
            for j in range(GENOME_LENGTH):
                local_arr[j] = population[chr, j]
            fitness[chr] = calculate_fitness(city_coords, local_arr)
        truncation(fitness, parents)
        for 


run()
# print(parents)
# print(fitness.to_numpy())
# print(population)