from taichi_utils import *
import taichi as ti
import time

ti.init(arch=ti.cpu)

@ti.kernel
def get_population_fitness(population: ti.template(), population_cost: ti.template(), population_fitness: ti.template(), citymap: ti.template()):
    for tid in range(ISLANDS):
        start_idx = tid * num_cities
        end_idx = start_idx + num_cities - 1
        last_idx = start_idx + num_cities
        evaluate_route_taichi(population,population_cost, population_fitness, citymap,start_idx,end_idx,last_idx)

@ti.kernel
def selection(population: ti.template(), population_cost: ti.template(), population_fitness: ti.template(), parent_cities: ti.template()):
    for tid in range(ISLANDS):
        parent1 = ti.Vector.zero(ti.i32,num_cities)

        parent1 = tournamentSelection(population, population_cost, population_fitness, tid)

        for c in range(num_cities):
            parent_cities[tid * (2 * num_cities) + c] = parent1[c]

        parent1 = tournamentSelection(population, population_cost, population_fitness, tid)

        for c in range(num_cities):
            parent_cities[tid * (2 * num_cities) + num_cities + c] = parent1[c]

@ti.kernel
def crossover(population: ti.template(), population_cost: ti.template(),
              population_fitness: ti.template(), parent_cities: ti.template(), citymap: ti.template()):
    
    for tid in range(ISLANDS):

        population[tid*num_cities] = parent_cities[tid*(2*num_cities)]

        parent_city_ptr = ti.Vector.zero(ti.i32,num_cities)
        for i in range(num_cities):
            parent_city_ptr[i] = parent_cities[tid*num_cities*2 + i]

        tourarray = ti.Vector.zero(ti.i32,num_cities)
        for i in range(num_cities):
            tourarray[i] = population[tid*num_cities + i]

        current_city_id = population[tid*num_cities + 10 - 1]

        c1 = get_valid_next_city(parent_city_ptr, tourarray, current_city_id, 10)

        for i in range(num_cities):
            parent_city_ptr[i] = parent_cities[tid*num_cities*2 + num_cities + i]

        c2 = get_valid_next_city(parent_city_ptr, tourarray, current_city_id, 10)

        if citymap[c1*num_cities + current_city_id] <= citymap[c2*num_cities + current_city_id]:
            population[tid*num_cities + 10] = c1
        else:
            population[tid*num_cities + 10] = c2

@ti.kernel
def mutation(population_d: ti.template()):
    for tid in range(ISLANDS):
        if ti.random() < mutation_rate:
            random_num1 = int(1 + ti.random() * (num_cities - 1.00001))
            random_num2 = int(1 + ti.random() * (num_cities - 1.00001))

            city_temp = population_d[tid*num_cities + random_num1]
            population_d[tid*num_cities + random_num1] = population_d[tid*num_cities + random_num2]
            population_d[tid*num_cities + random_num2] = city_temp


#constant
def run():
    max_val = 250
    citymap = [0]*(num_cities*num_cities)

    for i in range(num_cities):
        for j in range(num_cities):
            if(i!=j):
                citymap[i*num_cities+j] = L2_distance(city_x[i], city_y[i], city_x[j], city_y[j])
            else:
                citymap[i*num_cities+j] = max_val*max_val
    
    initialize_random_population(citymap)
    # print("Num islands:", ISLANDS)
    # print("Population size:", ISLANDS*num_cities)

    fittest = get_fittest_score()
    # print("min distance",population_cost[fittest])

    population_d = ti.field(dtype=ti.i32, shape=(ISLANDS*num_cities))
    copy_array(population,population_d)
    population_cost_d = ti.field(dtype=ti.f32, shape=(ISLANDS))
    copy_array(population_cost,population_cost_d)
    population_fitness_d = ti.field(dtype=ti.f32, shape=(ISLANDS))
    copy_array(population_fitness,population_fitness_d)
    citymap_d = ti.field(dtype=ti.f32, shape=(num_cities*num_cities))
    copy_array(citymap,citymap_d)
    parent_cities_d = ti.field(dtype=ti.i32, shape=(ISLANDS*2*num_cities))
    get_population_fitness(population_d, population_cost_d, population_fitness_d, citymap_d)
    start_time = time.time()
    for i in range(num_generations):
        selection(population_d,population_cost_d,population_fitness_d,parent_cities_d)
        for j in range(num_cities):
            crossover(population_d,population_cost_d,population_fitness_d,parent_cities_d,citymap_d)
        mutation(population_d)
        get_population_fitness(population_d, population_cost_d, population_fitness_d, citymap_d)
    end_time = time.time()
    execution_time = end_time - start_time
    print("EA execution time:", execution_time, "seconds")
    copy_array_taichi(population_fitness_d,population_fitness)
    copy_array_taichi(population_cost_d,population_cost)
    fittest = get_fittest_score()
    # print("min distance",population_cost[fittest])

if __name__ == "__main__":
    start_time = time.time()

    run()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Total time:", execution_time, "seconds")