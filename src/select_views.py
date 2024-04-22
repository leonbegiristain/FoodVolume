import numpy as np
import matplotlib.pyplot as plt
import os
from read_dmap import loadDMAP

def greedy_max_min(set: list, distance_function, k):
    selected = []  # Initialize an empty list to store selected elements
    remaining = set[:]  # Create a copy of the original set
    
    # Iterate until k elements are selected
    for _ in range(k):
        if len(remaining) == 0:
            break  # If no elements remain, exit the loop
        
        max_min_distance = None
        best_candidate = None
        best_candidate_index = None
        
        # Find the element with the maximum minimum distance to selected elements
        for index, candidate in enumerate(remaining):
            min_distance = min(distance_function(candidate, s) for s in selected) if len(selected) > 0 else float('inf')
            if max_min_distance is None or min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
                best_candidate_index = index
        
        selected.append(best_candidate)  # Add the selected element to the list
        remaining.pop(best_candidate_index)  # Remove the selected element from the remaining elements
    
    return selected

# Example usage
# Define a distance function (assuming elements are vectors)
def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def custom_distance(a, b):
    alpha = 1
    beta = 1
    position1, number1 = a
    position2, number2 = b

    print(euclidean_distance(position1, position2), number1)

    return alpha * euclidean_distance(position1, position2) + beta * number1

def depth_map_points(dmap):
    depth = dmap["depth_map"]
    return np.sum(depth > 0)



# Example set of elements (vectors)
output_path = os.path.abspath("./results/orange_coin/")

dmap_paths = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".dmap")]
dmap_list = [loadDMAP(path) for path in dmap_paths]

elements = [[-dmap["R"] @ dmap["C"], depth_map_points(dmap)] for dmap in dmap_list]

numbers_avg = np.mean([e[1] for e in elements])
numbers_std = np.std([e[1] for e in elements])

elements = [[e[0], (e[1] - numbers_avg) / numbers_std] for e in elements]
min_number = min([e[1] for e in elements])
max_number = max([e[1] for e in elements])
elements = [[e[0], (e[1] - min_number) / (max_number - min_number)] for e in elements]

print([e[1] for e in elements])

selected_elements = greedy_max_min(elements, custom_distance, 5)

# Plot the selected elements
selected_elements = np.array([e[0] for e in selected_elements])
elements = np.array([e[0] for e in elements if e[0] not in selected_elements])

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elements[:, 0], elements[:, 1], elements[:, 2], c='b', label='All elements')
ax.scatter(selected_elements[:, 0], selected_elements[:, 1], selected_elements[:, 2], c='r', label='Selected elements')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()