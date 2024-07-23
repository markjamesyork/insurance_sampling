import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

def create_grid(lat_min, lon_min, lat_max, lon_max, size_km):
    R = 6371  # Earth's radius in kilometers
    lat_step = (size_km / R) * (180 / np.pi)
    lon_step = lat_step / np.cos(lat_min * np.pi / 180)
    
    grid = []
    grid_id = 0
    for lat in np.arange(lat_min, lat_max, lat_step):
        for lon in np.arange(lon_min, lon_max, lon_step):
            # Define each grid square using `box`
            b = box(lon, lat, lon + lon_step, lat + lat_step)
            # Calculate the centroid of each box
            centroid = b.centroid
            grid.append({
                'grid_id': grid_id,
                'geometry': b,
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x
            })
            grid_id += 1
    
    # Creating a GeoDataFrame from the list of boxes
    grid_gdf = gpd.GeoDataFrame(grid)
    return grid_gdf

def process_yield_data(file_path, grid_size):
    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    lat_min, lat_max = gdf.geometry.y.min(), gdf.geometry.y.max()
    lon_min, lon_max = gdf.geometry.x.min(), gdf.geometry.x.max()

    grid = create_grid(lat_min, lon_min, lat_max, lon_max, grid_size)

    # Save grid data to CSV
    grid[['grid_id', 'centroid_lat', 'centroid_lon']].to_csv('grid_centroids.csv', index=False)

    return grid

# Usage example assuming the data file is located at 'data/kenya_yield_data.csv' and grid size is 500 km
stats = process_yield_data('data/kenya_yield_data.csv', 100)
print(stats)


