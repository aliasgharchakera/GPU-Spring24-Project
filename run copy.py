# a function that reads qa194.tsp file and returns the data inform of number of cities and their coordinate array of x and y
def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data_x = []
        data_y = []
        nodes_start = False
        for i in lines:
            if i.startswith('NODE_COORD_SECTION'):
                nodes_start = True
                continue
            elif i.startswith('EOF'):
                nodes_start = False
                break
            if nodes_start:
                data = i.split()
                data_x.append(float(data[1]))
                data_y.append(float(data[2]))
        return data_x, data_y

a,b = read_file('qa194.tsp')
# print(a,b,len(b))
print(b)