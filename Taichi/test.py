import taichi as ti
import numpy as np
import random

ti.init(arch=ti.cpu)  # Initialize Taichi to use the CPU

# Constants
POPULATION_SIZE = 100
GENOME_LENGTH = 10
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
NUM_GENERATIONS = 50
SELECTION_PRESSURE = 5  # Higher means more biased towards better candidates

# Taichi fields
population = ti.field(dtype=ti.i32, shape=(POPULATION_SIZE, GENOME_LENGTH))
fitness = ti.field(dtype=ti.f32, shape=POPULATION_SIZE)

@ti.kernel
def evaluate():
    for i in range(POPULATION_SIZE):
        fit = 0.0
        for j in range(GENOME_LENGTH):
            fit += population[i, j]
        fitness[i] = fit

@ti.kernel
def mutate():
    for i in range(POPULATION_SIZE):
        for j in range(GENOME_LENGTH):
            if random.random() < MUTATION_RATE:
                population[i, j] = 1 - population[i, j]

def select_parent():
    # Tournament selection
    best = random.randint(0, POPULATION_SIZE - 1)
    for _ in range(SELECTION_PRESSURE):
        next = random.randint(0, POPULATION_SIZE - 1)
        if fitness[next] > fitness[best]:
            best = next
    return best

@ti.kernel
def crossover(parent1: int, parent2: int, child: int):
    for j in range(GENOME_LENGTH):
        if random.random() < CROSSOVER_RATE:
            population[child, j] = population[parent1, j]
        else:
            population[child, j] = population[parent2, j]

def initialize_population():
    for i in range(POPULATION_SIZE):
        for j in range(GENOME_LENGTH):
            population[i, j] = random.randint(0, 1)

def run_evolution():
    initialize_population()
    for generation in range(NUM_GENERATIONS):
        evaluate()
        new_population = np.zeros((POPULATION_SIZE, GENOME_LENGTH), dtype=int)
        for i in range(POPULATION_SIZE):
            parent1 = select_parent()
            parent2 = select_parent()
            crossover(parent1, parent2, i)
        population.from_numpy(new_population)
        mutate()
        print(f'Generation {generation + 1} complete')

if __name__ == "__main__":
    run_evolution()
