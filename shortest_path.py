from christofides import*
import numpy as np
from calc_yield_prior import haversine

def create_distance_matrix(latitudes, longitudes):
    """
    Creates an upper triangular matrix of distances between all pairs of points.
    """
    n = len(latitudes)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        if i < n - 1:  # Only compute distances above the diagonal
            dists = haversine(latitudes[i], longitudes[i], latitudes[i+1:], longitudes[i+1:])
            distance_matrix[i, i+1:] = dists

    return distance_matrix


# Execution:
latitudes = np.array([34.0522, 36.7783, 40.7128])
longitudes = np.array([-118.2437, -119.4179, -74.0060])

distance_matrix = create_distance_matrix(latitudes, longitudes)
print("Distance Matrix:")
print(distance_matrix)

TSP = compute(distance_matrix)
print('TSP', TSP)