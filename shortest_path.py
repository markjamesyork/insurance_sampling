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


def map_results(latitudes, longitudes, order):
    # Create a map centered around the average of the coordinates
    map_center = [np.mean(latitudes), np.mean(longitudes)]
    mymap = folium.Map(location=map_center, zoom_start=2)

    # Adding markers to the map
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


# Execution:
'''
latitudes = np.array([34.0522, 36.7783, 40.7128, 38.99, 35.1])
longitudes = np.array([-118.2437, -119.4179, -74.0060, -107., -107.])
'''
n = 200
latitudes = np.random.random((n,)) * 180 - 90
longitudes = np.random.random((n,)) * 360 - 180
lat_lons = np.vstack((latitudes, longitudes)).T
print('lat_lons', lat_lons)


distance_matrix = create_distance_matrix(latitudes, longitudes)
print("Distance Matrix:")
print(np.round(distance_matrix,0))

TSP = tsp(lat_lons)
print('TSP', TSP)

map_results(latitudes, longitudes, TSP[1])



