import taichi as ti

ti.init(arch=ti.cpu)

# Constants
chromosome_size = 194
max_population = 100

#data
coords_x = [24748.3333, 24758.8889, 24827.2222, 24904.4444, 24996.1111, 25010.0, 25030.8333, 25067.7778, 25100.0, 25103.3333, 25121.9444, 25150.8333, 25158.3333, 25162.2222, 25167.7778, 25168.8889, 25173.8889, 25210.8333, 25211.3889, 25214.1667, 25214.4444, 25223.3333, 25224.1667, 25233.3333, 25234.1667, 25235.5556, 25235.5556, 25242.7778, 25243.0556, 25252.5, 25253.8889, 25253.8889, 25256.9444, 25263.6111, 25265.8333, 25266.6667, 25266.6667, 25270.5556, 25270.8333, 25270.8333, 25275.8333, 25277.2222, 25278.3333, 25278.3333, 25279.1667, 25281.1111, 25281.3889, 25283.3333, 25283.6111, 25284.7222, 25286.1111, 25286.1111, 25286.6667, 25287.5, 25288.0556, 25290.8333, 25291.9444, 25292.5, 25298.6111, 25300.8333, 25306.9444, 25311.9444, 25313.8889, 25315.2778, 25316.6667, 25320.5556, 25322.5, 25325.2778, 25326.6667, 25337.5, 25339.1667, 25340.5556, 25341.9444, 25358.8889, 25363.6111, 25368.6111, 25374.4444, 25377.7778, 25396.9444, 25400.0, 25400.0, 25404.7222, 25416.9444, 25416.9444, 25419.4444, 25429.7222, 25433.3333, 25440.8333, 25444.4444, 25451.3889, 25459.1667, 25469.7222, 25478.0556, 25480.5556, 25483.3333, 25490.5556, 25492.2222, 25495.0, 25495.0, 25497.5, 25500.8333, 25510.5556, 25531.9444, 25533.3333, 25538.8889, 25545.8333, 25549.7222, 25550.0, 25560.2778, 25566.9444, 25567.5, 25574.7222, 25585.5556, 25609.4444, 25610.2778, 25622.5, 25645.8333, 25650.0, 25666.9444, 25683.8889, 25686.3889, 25696.1111, 25700.8333, 25708.3333, 25716.6667, 25717.5, 25723.0556, 25734.7222, 25751.1111, 25751.9444, 25758.3333, 25765.2778, 25772.2222, 25775.8333, 25779.1667, 25793.3333, 25808.3333, 25816.6667, 25823.6111, 25826.6667, 25829.7222, 25833.3333, 25839.1667, 25847.7778, 25850.0, 25856.6667, 25857.5, 25857.5, 25866.6667, 25867.7778, 25871.9444, 25872.5, 25880.8333, 25883.0556, 25888.0556, 25900.0, 25904.1667, 25928.3333, 25937.5, 25944.7222, 25950.0, 25951.6667, 25957.7778, 25958.3333, 25966.6667, 25983.3333, 25983.6111, 26000.2778, 26008.6111, 26016.6667, 26021.6667, 26033.3333, 26033.3333, 26033.6111, 26033.6111, 26048.8889, 26050.0, 26050.2778, 26050.5556, 26055.0, 26067.2222, 26074.7222, 26076.6667, 26077.2222, 26078.0556, 26083.6111, 26099.7222, 26108.0556, 26116.6667, 26123.6111, 26123.6111, 26133.3333, 26133.3333, 26150.2778]
coords_y = [50840.0, 51211.9444, 51394.7222, 51175.0, 51548.8889, 51039.4444, 51275.2778, 51077.5, 51516.6667, 51521.6667, 51218.3333, 51537.7778, 51163.6111, 51220.8333, 51606.9444, 51086.3889, 51269.4444, 51394.1667, 51619.1667, 50807.2222, 51378.8889, 51451.6667, 51174.4444, 51333.3333, 51203.0556, 51330.0, 51495.5556, 51428.8889, 51452.5, 51559.1667, 51535.2778, 51549.7222, 51398.8889, 51516.3889, 51545.2778, 50969.1667, 51483.3333, 51532.7778, 51505.8333, 51523.0556, 51533.6111, 51547.7778, 51525.5556, 51541.3889, 51445.5556, 51535.0, 51512.5, 51533.3333, 51546.6667, 51555.2778, 51504.1667, 51534.1667, 51533.3333, 51537.7778, 51546.6667, 51528.3333, 51424.4444, 51520.8333, 51001.6667, 51394.4444, 51507.7778, 51003.0556, 50883.3333, 51438.6111, 50766.6667, 51495.5556, 51507.7778, 51470.0, 51350.2778, 51425.0, 51173.3333, 51293.6111, 51507.5, 51333.6111, 51281.1111, 51226.3889, 51436.6667, 51294.7222, 51422.5, 51183.3333, 51425.0, 51073.0556, 51403.8889, 51457.7778, 50793.6111, 50785.8333, 51220.0, 51378.0556, 50958.3333, 50925.0, 51316.6667, 51397.5, 51362.5, 50938.8889, 51383.3333, 51373.6111, 51400.2778, 50846.6667, 50965.2778, 51485.2778, 50980.5556, 51242.2222, 51304.4444, 50977.2222, 51408.3333, 51387.5, 51431.9444, 51433.3333, 51158.6111, 51484.7222, 50958.8889, 51486.3889, 51151.3889, 51092.2222, 51475.2778, 51454.4444, 51450.0, 51372.2222, 51174.4444, 51505.8333, 51468.8889, 51260.8333, 51584.7222, 51591.6667, 51050.0, 51057.7778, 51004.1667, 51547.5, 51449.1667, 50920.8333, 51395.8333, 51019.7222, 51483.3333, 51023.0556, 51449.7222, 51409.4444, 51060.5556, 51133.3333, 51152.5, 51043.8889, 51245.2778, 51072.2222, 51465.2778, 51205.8333, 51033.3333, 51083.3333, 51298.8889, 51441.3889, 51066.6667, 51205.5556, 51354.7222, 51258.3333, 51221.3889, 51185.2778, 51386.3889, 51000.0, 51201.6667, 51337.5, 51313.3333, 51456.3889, 51066.6667, 51349.7222, 51075.2778, 51099.4444, 51283.3333, 51400.0, 51328.0556, 51294.4444, 51083.6111, 51333.3333, 51366.9444, 51116.6667, 51166.6667, 51163.8889, 51200.2778, 51056.9444, 51250.0, 51297.5, 51135.8333, 51316.1111, 51258.6111, 51083.6111, 51166.9444, 51222.2222, 51361.6667, 51147.2222, 51161.1111, 51244.7222, 51216.6667, 51169.1667, 51222.7778, 51216.6667, 51300.0, 51108.0556]


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
    for j in range(generations):
        best_fitness = float('inf')
        best_index = 0
        # TODO: make the best fitness computation on CPU 
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
