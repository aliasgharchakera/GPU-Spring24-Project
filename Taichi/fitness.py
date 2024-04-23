import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

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

# Example usage:
city_coords_array = np.array([
    [0, 0],
    [1, 2],
    [3, 1],
    [2, 4],
    [5, 2],
    [4, 6]
])

# Example chromosome
chromosome = ti.field(dtype=int, shape=len(city_coords_array))

for i in range(len(city_coords_array)):
    chromosome[i] = i

# Convert city coordinates to Taichi format
city_coords = ti.Vector.field(2, dtype=float, shape=len(city_coords_array))
for i in range(len(city_coords_array)):
    city_coords[i] = city_coords_array[i]

# Calculate fitness
fitness = calculate_fitness(city_coords, chromosome)
print("Fitness of the chromosome:", fitness)
