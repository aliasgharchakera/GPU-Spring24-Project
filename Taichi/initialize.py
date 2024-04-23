import taichi as ti
import numpy as np
import random

ti.init(arch=ti.cpu)  # Initialize Taichi with GPU support

@ti.kernel
def initialize_chromosomes(n_chromosomes: ti.i32, chromosome_size: ti.i32, chromosomes: ti.template(), indexes: ti.template()):
    """
    Initialize an array of chromosomes where each chromosome is a list of indexes 1-N
    shuffled randomly.

    Args:
    - n_chromosomes (int): Number of chromosomes to generate.
    - chromosome_size (int): Size of each chromosome.
    - chromosomes (ti.ext_arr()): Array of initialized chromosomes.
    """
    for i in range(n_chromosomes):
        # Generate a list of indexes 1-N and shuffle it randomly
        # indexes = list(range(1, chromosome_size + 1))
        # random.shuffle(indexes)
        # Assign the shuffled indexes to the chromosome array
        for j in range(chromosome_size):
            chromosomes[i, j] = indexes[j]

if __name__ == "__main__":
    # Parameters
    n_chromosomes = 10
    chromosome_size = 5
    indexes = ti.field(dtype=ti.i32, shape=chromosome_size)

    # Initialize Taichi variables
    chromosomes = ti.field(dtype=ti.i32, shape=(n_chromosomes, chromosome_size))

    # Initialize chromosomes
    initialize_chromosomes(n_chromosomes, chromosome_size, chromosomes,indexes)

    # Print initialized chromosomes
    print("Initialized Chromosomes:")
    print(chromosomes.to_numpy())
