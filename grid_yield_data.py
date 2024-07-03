import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, box

def create_grid(lat_min, lon_min, lat_max, lon_max, size_km):
    R = 6371  # Earth's radius in kilometers
    lat_step = (size_km / R) * (180 / np.pi)
    lon_step = lat_step / np.cos(lat_min * np.pi / 180)
    
    grid = []
    for lat in np.arange(lat_min, lat_max, lat_step):
        for lon in np.arange(lon_min, lon_max, lon_step):
            grid.append(box(lon, lat, lon + lon_step, lat + lat_step))
    
    # Creating a GeoDataFrame from the list of boxes
    grid_gdf = gpd.GeoDataFrame(geometry=grid)
    return grid_gdf

def process_yield_data(file_path, grid_size):
    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    lat_min, lat_max = gdf.geometry.y.min(), gdf.geometry.y.max()
    lon_min, lon_max = gdf.geometry.x.min(), gdf.geometry.x.max()

    grid = create_grid(lat_min, lon_min, lat_max, lon_max, grid_size)
    gdf = gpd.sjoin(gdf, grid, how='left', predicate='within')

    
    # Visualization
    for year in gdf['year'].unique():
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(f'Average Maize Yield in {year}')
        subset = gdf[gdf['year'] == year]
        subset.plot(column='yield', ax=ax, legend=True, cmap='viridis', edgecolor='black')
        plt.savefig(f'maps/average_yield_{year}.png')
        plt.close()

    return gdf


# Assume the data file is located at 'data/iowa_state_yield_trend.csv' and grid size is 10km
stats = process_yield_data('data/kenya_yield_data.csv', 100)
print(stats)
print('type: ', type(stats))

