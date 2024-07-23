import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

def create_grid(lat_min, lon_min, lat_max, lon_max, size_km):
    R = 6371  # Earth's radius in kilometers
    lat_step = (size_km / R) * (180 / np.pi)
    lon_step = lat_step / np.cos(lat_min * np.pi / 180)
    
    grid = []
    grid_id = 0
    for lat in np.arange(lat_min, lat_max, lat_step):
        for lon in np.arange(lon_min, lon_max, lon_step):
            b = box(lon, lat, lon + lon_step, lat + lat_step)
            centroid = b.centroid
            grid.append({
                'grid_id': grid_id,
                'geometry': b,
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x
            })
            grid_id += 1
    
    grid_gdf = gpd.GeoDataFrame(grid)
    return grid_gdf

def process_yield_data(file_path, grid_size, shapefile_path):
	# Shapefile Source: https://www.naturalearthdata.com/downloads/
    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    lat_min, lat_max = gdf.geometry.y.min(), gdf.geometry.y.max()
    lon_min, lon_max = gdf.geometry.x.min(), gdf.geometry.x.max()

    grid = create_grid(lat_min, lon_min, lat_max, lon_max, grid_size)

    # Load the shapefile for world map and filter for Kenya
    world = gpd.read_file(shapefile_path)
    print('world.columns: ', world.columns)  # Print the column names to find the correct one

    kenya = world[world.NAME == "Kenya"]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 15))
    kenya.plot(ax=ax, color='white', edgecolor='black')
    grid.plot(ax=ax, alpha=0.5, edgecolor='k', cmap='hot', legend=True)  # Random heatmap colors

    plt.title('Grid Overlay on Map of Kenya')
    plt.axis('equal')
    plt.savefig('Kenya_with_grids.png')
    plt.show()

    # Save grid data to CSV
    grid[['grid_id', 'centroid_lat', 'centroid_lon']].to_csv('grid_centroids.csv', index=False)

    return grid

# Example usage assuming the data file and shapefile paths
shapefile_path = 'maps/ne_10m_admin_0_countries'
stats = process_yield_data('data/kenya_yield_data.csv', 100, shapefile_path)
print(stats)
