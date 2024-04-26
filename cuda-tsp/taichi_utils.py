import math
import random
import numpy as np
import taichi as ti

num_cities = 194
num_generations = 50
tournament_size = 2
mutation_rate = 0.2
ISLANDS = 96

city_x = [24748.3333, 24758.8889, 24827.2222, 24904.4444, 24996.1111, 25010.0000, 25030.8333, 25067.7778, 25100.0000, 25103.3333, 25121.9444, 25150.8333, 25158.3333, 25162.2222, 25167.7778, 25168.8889, 25173.8889, 25210.8333, 25211.3889, 25214.1667, 25214.4444, 25223.3333, 25224.1667, 25233.3333, 25234.1667, 25235.5556, 25235.5556, 25242.7778, 25243.0556, 25252.5000, 25253.8889, 25253.8889, 25256.9444, 25263.6111, 25265.8333, 25266.6667, 25266.6667, 25270.5556, 25270.8333, 25270.8333, 25275.8333, 25277.2222, 25278.3333, 25278.3333, 25279.1667, 25281.1111, 25281.3889, 25283.3333, 25283.6111, 25284.7222, 25286.1111, 25286.1111, 25286.6667, 25287.5000, 25288.0556, 25290.8333, 25291.9444, 25292.5000, 25298.6111, 25300.8333, 25306.9444, 25311.9444, 25313.8889, 25315.2778, 25316.6667, 25320.5556, 25322.5000, 25325.2778, 25326.6667, 25337.5000, 25339.1667, 25340.5556, 25341.9444, 25358.8889, 25363.6111, 25368.6111, 25374.4444, 25377.7778, 25396.9444, 25400.0000, 25400.0000, 25404.7222, 25416.9444, 25416.9444, 25419.4444, 25429.7222, 25433.3333, 25440.8333, 25444.4444, 25451.3889, 25459.1667, 25469.7222, 25478.0556, 25480.5556, 25483.3333, 25490.5556, 25492.2222, 25495.0000, 25495.0000, 25497.5000, 25500.8333, 25510.5556, 25531.9444, 25533.3333, 25538.8889, 25545.8333, 25549.7222, 25550.0000, 25560.2778, 25566.9444, 25567.5000, 25574.7222, 25585.5556, 25609.4444, 25610.2778, 25622.5000, 25645.8333, 25650.0000, 25666.9444, 25683.8889, 25686.3889, 25696.1111, 25700.8333, 25708.3333, 25716.6667, 25717.5000, 25723.0556, 25734.7222, 25751.1111, 25751.9444, 25758.3333, 25765.2778, 25772.2222, 25775.8333, 25779.1667, 25793.3333, 25808.3333, 25816.6667, 25823.6111, 25826.6667, 25829.7222, 25833.3333, 25839.1667, 25847.7778, 25850.0000, 25856.6667, 25857.5000, 25857.5000, 25866.6667, 25867.7778, 25871.9444, 25872.5000, 25880.8333, 25883.0556, 25888.0556, 25900.0000, 25904.1667, 25928.3333, 25937.5000, 25944.7222, 25950.0000, 25951.6667, 25957.7778, 25958.3333, 25966.6667, 25983.3333, 25983.6111, 26000.2778, 26008.6111, 26016.6667, 26021.6667, 26033.3333, 26033.3333, 26033.6111, 26033.6111, 26048.8889, 26050.0000, 26050.2778, 26050.5556, 26055.0000, 26067.2222, 26074.7222, 26076.6667, 26077.2222, 26078.0556, 26083.6111, 26099.7222, 26108.0556, 26116.6667, 26123.6111, 26123.6111, 26133.3333, 26133.3333, 26150.2778]

city_y = [50840.0000, 51211.9444, 51394.7222, 51175.0000, 51548.8889, 51039.4444, 51275.2778, 51077.5000, 51516.6667, 51521.6667, 51218.3333, 51537.7778, 51163.6111, 51220.8333, 51606.9444, 51086.3889, 51269.4444, 51394.1667, 51619.1667, 50807.2222, 51378.8889, 51451.6667, 51174.4444, 51333.3333, 51203.0556, 51330.0000, 51495.5556, 51428.8889, 51452.5000, 51559.1667, 51535.2778, 51549.7222, 51398.8889, 51516.3889, 51545.2778, 50969.1667, 51483.3333, 51532.7778, 51505.8333, 51523.0556, 51533.6111, 51547.7778, 51525.5556, 51541.3889, 51445.5556, 51535.0000, 51512.5000, 51533.3333, 51546.6667, 51555.2778, 51504.1667, 51534.1667, 51533.3333, 51537.7778, 51546.6667, 51528.3333, 51424.4444, 51520.8333, 51001.6667, 51394.4444, 51507.7778, 51003.0556, 50883.3333, 51438.6111, 50766.6667, 51495.5556, 51507.7778, 51470.0000, 51350.2778, 51425.0000, 51173.3333, 51293.6111, 51507.5000, 51333.6111, 51281.1111, 51226.3889, 51436.6667, 51294.7222, 51422.5000, 51183.3333, 51425.0000, 51073.0556, 51403.8889, 51457.7778, 50793.6111, 50785.8333, 51220.0000, 51378.0556, 50958.3333, 50925.0000, 51316.6667, 51397.5000, 51362.5000, 50938.8889, 51383.3333, 51373.6111, 51400.2778, 50846.6667, 50965.2778, 51485.2778, 50980.5556, 51242.2222, 51304.4444, 50977.2222, 51408.3333, 51387.5000, 51431.9444, 51433.3333, 51158.6111, 51484.7222, 50958.8889, 51486.3889, 51151.3889, 51092.2222, 51475.2778, 51454.4444, 51450.0000, 51372.2222, 51174.4444, 51505.8333, 51468.8889, 51260.8333, 51584.7222, 51591.6667, 51050.0000, 51057.7778, 51004.1667, 51547.5000, 51449.1667, 50920.8333, 51395.8333, 51019.7222, 51483.3333, 51023.0556, 51449.7222, 51409.4444, 51060.5556, 51133.3333, 51152.5000, 51043.8889, 51245.2778, 51072.2222, 51465.2778, 51205.8333, 51033.3333, 51083.3333, 51298.8889, 51441.3889, 51066.6667, 51205.5556, 51354.7222, 51258.3333, 51221.3889, 51185.2778, 51386.3889, 51000.0000, 51201.6667, 51337.5000, 51313.3333, 51456.3889, 51066.6667, 51349.7222, 51075.2778, 51099.4444, 51283.3333, 51400.0000, 51328.0556, 51294.4444, 51083.6111, 51333.3333, 51366.9444, 51116.6667, 51166.6667, 51163.8889, 51200.2778, 51056.9444, 51250.0000, 51297.5000, 51135.8333, 51316.1111, 51258.6111, 51083.6111, 51166.9444, 51222.2222, 51361.6667, 51147.2222, 51161.1111, 51244.7222, 51216.6667, 51169.1667, 51222.7778, 51216.6667, 51300.0000, 51108.0556]

population = np.zeros(ISLANDS*num_cities, dtype=np.int32)
population_cost = np.zeros(ISLANDS, dtype=np.float32)
population_fitness = np.zeros(ISLANDS, dtype=np.float32)

def L2_distance(x1, y1, x2, y2):
    x_d = (x1 - x2) ** 2
    y_d = (y1 - y2) ** 2
    return math.sqrt(x_d + y_d)

def initialize_random_population(citymap):
    linear_tour = list(range(num_cities))

    for j in range(num_cities):
        population[j] = j

    temp_tour = linear_tour.copy()

    for i in range(ISLANDS):
        random.shuffle(temp_tour)

        for j in range(num_cities):
            population[i*num_cities + j] = temp_tour[j]
    evaluate_route(citymap, i)

@ti.func
def evaluate_route_taichi(population_d,population_cost_d,population_fitness_d,citymap_d,start_idx,end_idx,last_idx):
    for i in range(start_idx, end_idx):
        distance = 0
        for j in range(num_cities - 1):
            idx_curr = i + j
            idx_next = i + j + 1
            distance += citymap_d[population_d[idx_curr] * num_cities + population_d[idx_next]]
        distance += citymap_d[population_d[i + num_cities - 1] * num_cities + population_d[last_idx - 1]]

        population_cost_d[i // num_cities] = distance

        population_fitness_d[i // num_cities] = 0
        if distance != 0:
            population_fitness_d[i // num_cities] = 1.0 / distance

    # print(type(i))
    # distance = 0
    # for j in range(num_cities-1):
    #     distance += citymap_d[population_d[i*num_cities + j]*num_cities + population_d[i*num_cities + j+1]]
    # distance += citymap_d[population_d[i*num_cities + num_cities-1]*num_cities + population_d[i*num_cities]]

    # population_cost[i] = distance

    # population_fitness[i] = 0
    # if population_cost[i] != 0:
    #     population_fitness[i] = (1.0/population_cost[i])

def evaluate_route(citymap, i):
    # print(type(i))
    distance = 0
    for j in range(num_cities-1):
        distance += citymap[population[i*num_cities + j]*num_cities + population[i*num_cities + j+1]]
    distance += citymap[population[i*num_cities + num_cities-1]*num_cities + population[i*num_cities]]

    population_cost[i] = distance

    population_fitness[i] = 0
    if population_cost[i] != 0:
        population_fitness[i] = (1.0/population_cost[i])


def get_fittest_score():
    idx = np.where(population_fitness == max(population_fitness))
    return idx

def copy_array(np_arr,ta_array):
    for i in range(len(np_arr)):
        ta_array[i] = np_arr[i]

def copy_array_taichi(ta_array,np_arr):
    for i in range(ta_array.shape[0]):
        np_arr[i] = ta_array[i]

@ti.func
def tournamentSelection(population: ti.template(), population_cost:ti.template(), population_fitness:ti.template(), tid:ti.template()) -> ti.template():
    tournament = ti.Matrix.zero(ti.i32, tournament_size, num_cities)
    tournament_cost = ti.Vector.zero(ti.f32, tournament_size)
    tournament_fitness = ti.Vector.zero(ti.f32, tournament_size)

    for i in range(tournament_size):
        # Get random number from global random state on GPU
        random_num = int(ti.random()*ISLANDS)

        for c in range(num_cities):
            tournament[i, c] = population[random_num * num_cities + c]
        tournament_cost[i] = population_cost[random_num]
        tournament_fitness[i] = population_fitness[random_num]

    fittest = get_fittest_tour_index(tournament, tournament_cost, tournament_fitness)

    fittest_route = ti.Vector.zero(ti.i32, num_cities)
    for c in range(num_cities):
        fittest_route[c] = tournament[fittest, c]

    return fittest_route

@ti.func
def get_fittest_tour_index(tournament, tournament_cost, tournament_fitness):
    fittest = 0
    fitness = tournament_fitness[0]

    for i in range(1, tournament_size):
        if tournament_fitness[i] >= fitness:
            fittest = i
            fitness = tournament_fitness[i]

    return fittest

@ti.func
def get_valid_next_city(parent_cities_ptr, tourarray, current_city_id, index):

    # Finding current city in parent
    local_city_index = find_city(current_city_id, parent_cities_ptr, num_cities)

    # Initialize the result
    valid_city = 0
    valid_city_found = False

    # Search for the first valid city (not already in child)
    # occurring after currentCities location in parent tour
    for i in range(local_city_index + 1, num_cities):
        # If not in child already, select it!
        if find_city(parent_cities_ptr[i], tourarray, index) == -1:
            valid_city = parent_cities_ptr[i]
            valid_city_found = True
            break

    # Loop through city ids [1.. NUM_CITIES] and find the first valid city
    # to choose as the next point in construction of the child tour
    if not valid_city_found:
        for i in range(1, num_cities):
            in_tour_already = False
            for j in range(1, index):
                if tourarray[j] == i:
                    in_tour_already = True
                    break

            if not in_tour_already:
                valid_city = getCityN(i, parent_cities_ptr)

    # If no valid city is found, return 0
    return valid_city


@ti.func
def find_city(current_city_id, tour, local_num_cities):
    found_index = -1  # Initialize the index to -1
    for i in range(local_num_cities):
        if current_city_id == tour[i]:
            found_index = i
            break  # Exit the loop early if city is found
    return found_index

@ti.func
def getCityN(n, parent_cities_ptr):
    result = 0  # Initialize the result
    for i in range(num_cities):
        if parent_cities_ptr[i] == n:
            result = parent_cities_ptr[i]
            break  # Exit the loop early if city is found
    return result

