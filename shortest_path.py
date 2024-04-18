from christofides import*
import folium
from calc_yield_prior import haversine
import numpy as np


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
latitudes = np.array([34.0522, 36.7783, 40.7128, 38.99, 35.1])
longitudes = np.array([-118.2437, -119.4179, -74.0060, -107., -107.])
lat_lons = np.vstack((latitudes, longitudes)).T
print('lat_lons', lat_lons)

'''
distance_matrix = create_distance_matrix(latitudes, longitudes)
print("Distance Matrix:")
print(distance_matrix)
'''

TSP = tsp(lat_lons)
print('TSP', TSP)


# Map this shit!
'''
# Define the list of coordinates (latitude, longitude)
coordinates = [
    (40.712776, -74.005974),  # Example: New York
    (34.052235, -118.243683), # Example: Los Angeles
    (51.507351, -0.127758),   # Example: London
    (35.689487, 139.691711),  # Example: Tokyo
    (48.856613, 2.352222)     # Example: Paris
]

# Define the order in which to connect these points (using indices)
order = [1, 0, 4, 3, 2, 1]
'''

# Create a map centered around the average of the coordinates
map_center = [np.mean(latitudes), np.mean(longitudes)]
mymap = folium.Map(location=map_center, zoom_start=2)

# Adding markers to the map
order = TSP[1]
for idx, (lat, lon) in enumerate(lat_lons):
    folium.Marker([lat, lon], popup=str(idx)).add_to(mymap)

# Adding lines between the ordered points
for i in range(len(order)-1):
    start_idx, end_idx = order[i], order[i+1]
    start_coord = lat_lons[start_idx]
    end_coord = lat_lons[end_idx]
    folium.PolyLine([start_coord, end_coord], color='blue', weight=2.5, opacity=1).add_to(mymap)

# Save or show the map
mymap.save('maps/map.html')  # This saves the map to a file
# mymap  # If running in a Jupyter notebook, you can simply display the map in the notebook
