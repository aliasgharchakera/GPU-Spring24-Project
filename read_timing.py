import re

def extract_execution_times(file_path):
    execution_times = []
    with open(file_path, 'r') as file:
        file_content = file.read()
        matches = re.findall(r'Total execution time:\s+([\d.]+)\s+(\w+)', file_content)
        for match in matches:
            time, unit = match
            if unit == 'ms':
                time = float(time) / 1000  # convert milliseconds to seconds
            elif unit == 's':
                time = float(time)
            execution_times.append(time)
    init_time = (execution_times[0]+execution_times[2]+execution_times[4])/3
    selection_time = (execution_times[1]+execution_times[3]+execution_times[5])/3
    # print(execution_times)
    return init_time, selection_time

# file_path = "profiling_info.txt"  # replace with your actual file path
# execution_times = extract_execution_times(file_path)
# print("Total execution times:", execution_times)
