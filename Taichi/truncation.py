import taichi as ti

# CUDA kernel to find the top 2 values in an array
@ti.kernel
def find_top_2(values: ti.template(), top_2: ti.template()):
    n = values.shape[0]
    dummy = [0] * top_2.shape[0]

    # Find top 2 values
    for i in range(n):
        val = values[i]
        if val > dummy[0]:
            dummy[1] = dummy[0]
            top_2[1] = top_2[0]
            dummy[0] = val
            top_2[0] = i
        elif val > dummy[1]:
            dummy[1] = val
            top_2[1] = i

@ti.kernel
def init(values: ti.template()):
    for i in range(values.shape[0]):
        values[i] = ti.random() * 100

# Initialize Taichi
ti.init(arch=ti.cpu)

array_size = 100  # Change this to your desired array size

# Initialize array with random values
values = ti.field(ti.i32,shape=array_size)

top = 2

# Create Taichi field to store top 2 values
top_2 = ti.field(ti.i32, shape=top)

init(values)
print(values)
# Launch kernel to find top 2 values
find_top_2(values, top_2)

# Output the top 2 values
print("Top 2 values:", top_2)
