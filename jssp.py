import taichi as ti
import time

ti.init(arch=ti.gpu)

# Constants
population_size = 100
offspring_size = 100
generations = 1000
mutation_rate = 0.01
chromosome_size = 100

# Data
fitness = ti.field(dtype=ti.f32, shape=population_size)
population = ti.field(dtype=ti.i32, shape=(population_size, chromosome_size))
best_chromosomes = ti.field(dtype=ti.i32, shape=(generations, chromosome_size))
best_fitnesses = ti.field(dtype=ti.f32, shape=generations)

# Randomly initialize the population
@ti.kernel
def init_population():
    for i in range(population_size):
        for j in range(chromosome_size):
            population[i, j] = ti.random(ti.i32)