import numpy as np
from typing import Callable


def select_views(dmap_list: list, n_views: int, selected_indeces: list = [], alpha: float = 1, beta: float = 1) -> list:
    """
    Select a subset of views from a list of depth maps based on the maximum minimum distance to selected elements.
    :param dmap_list: A list of depth map dictionaries.
    :param alpha: A float value to weight the Euclidean distance.
    :param beta: A float value to weight the number of points in the depth map.
    """
    def greedy_max_min(points: list, distances: list, k: int, selected_indeces: list = []) -> list[int]:
        """
        Greedy algorithm to select k indeces from a set based on the maximum minimum distance to selected elements.
        """

        k -= len(selected_indeces)

        if k <= 0:
            raise Exception("k must be greater than 0")

        if k > len(points):
            raise Exception("k must be less or equal than the number of elements")
        
        if k == len(points):
            return list(range(len(points)))
        
        if (len(selected_indeces) == 0):
            # Add the point with the largest average distance to all other points
            distances_avg = [np.mean(distances[i]) for i in range(len(points))]
            selected_indeces.append(np.argmax(distances_avg))
            k -= 1
        
        # Iterate until k elements are selected
        for _ in range(k):            
            max_min_distance = None
            best_candidate_index = None
            
            # Find the element with the maximum minimum distance to selected elements
            for index in range(len(points)):
                if index in selected_indeces:
                    continue
                min_distance = min(distances[index][s] for s in selected_indeces) if len(selected_indeces) > 0 else float('inf')
                if max_min_distance is None or min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate_index = index
            
            selected_indeces.append(best_candidate_index)  # Add the selected element to the list
        
        return selected_indeces

    def euclidean_distance(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def custom_distance(a, b):
        position1, number1 = a
        position2, number2 = b

        return alpha * euclidean_distance(position1, position2) + beta * number1
    
    def depth_map_points(dmap):
        depth = dmap["depth_map"]
        return np.sum(depth > 0)



    elements = [[-dmap["R"] @ dmap["C"], depth_map_points(dmap)] for dmap in dmap_list]

    numbers_avg = np.mean([e[1] for e in elements])
    numbers_std = np.std([e[1] for e in elements])
    if numbers_std == 0:
        numbers_std = 1

    elements = [[e[0], (e[1] - numbers_avg) / numbers_std] for e in elements]
    min_number = min([e[1] for e in elements])
    max_number = max([e[1] for e in elements])
    if (max_number - min_number) == 0:
        max_number = min_number + 1
    elements = [[e[0], (e[1] - min_number) / (max_number - min_number)] for e in elements]
    distances = [[custom_distance(e1, e2) for e2 in elements] for e1 in elements]

    selected_indeces = greedy_max_min(elements, distances, n_views, selected_indeces)

    return [dmap_list[i] for i in selected_indeces]


if __name__ == "__main__":
    # Example usage
    dmap_list = [{"depth_map": np.random.rand(10, 10), "R": np.random.rand(3, 3), "C": np.random.rand(3)} for _ in range(100)]
    selected_views = select_views(dmap_list, 3, [], 1, 0)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([(-dmap["R"] @ dmap["C"])[0] for dmap in dmap_list], [(-dmap["R"] @ dmap["C"])[1] for dmap in dmap_list], 'ro')
    plt.plot([(-dmap["R"] @ dmap["C"])[0] for dmap in selected_views], [(-dmap["R"] @ dmap["C"])[1] for dmap in selected_views], 'bo')
    first_selection = selected_views[0]
    plt.plot((-first_selection["R"] @ first_selection["C"])[0], (-first_selection["R"] @ first_selection["C"])[1], 'go')
    plt.show()