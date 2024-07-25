import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import matplotlib.colors as colors

def create_grid(lat_min, lon_min, lat_max, lon_max, size_km):
    R = 6371  # Earth's radius in kilometers
    lat_step = (size_km / R) * (180 / np.pi)
    lon_step = lat_step / np.cos(lat_min * np.pi / 180)
    
    grid = []
    grid_id = 0
    for lat in np.arange(lat_min, lat_max, lat_step):
        for lon in np.arange(lon_min, lon_max, lon_step):
            b = box(lon, lat, lon + lon_step, lat + lat_step)
            grid.append({
                'grid_id': grid_id,
                'geometry': b
            })
            grid_id += 1
    
    grid_gdf = gpd.GeoDataFrame(grid)
    return grid_gdf

def process_yield_data(file_path, grid_size):
    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    lat_min, lat_max = gdf.geometry.y.min(), gdf.geometry.y.max()
    lon_min, lon_max = gdf.geometry.x.min(), gdf.geometry.x.max()

    grid = create_grid(lat_min, lon_min, lat_max, lon_max, grid_size)
    gdf = gpd.sjoin(gdf, grid, how='left', predicate='within')

    # Calculate the average yield for each grid square
    yield_by_grid = gdf.groupby('grid_id').agg({'yield': 'mean'}).rename(columns={'yield': 'average_yield'})
    grid = grid.merge(yield_by_grid, on='grid_id', how='left')

    # Load the shapefile for the world map and filter for Kenya, then load additional layers
    world = gpd.read_file('maps/ne_10m_admin_0_countries') # Shapefile Source: https://www.naturalearthdata.com/downloads/	
    kenya = world[world.NAME == "Kenya"]
    lakes = gpd.read_file('maps/ne_10m_lakes')
    rivers = gpd.read_file('maps/ne_10m_rivers_lake_centerlines')

    # Load cities and filter for large cities within Kenya
    cities = gpd.read_file('maps/ne_10m_populated_places')
    cities['POP_MAX'] = pd.to_numeric(cities['POP_MAX'], errors='coerce')
    large_cities = cities[cities['POP_MAX'] > 250000]  # Adjust the population threshold as needed
    large_cities_in_kenya = gpd.sjoin(large_cities, kenya, how='inner', predicate='intersects')
    print('cities.columns', large_cities_in_kenya.columns)  # This will print out all column names in the dataset

	# Perform spatial join to filter lakes and rivers within Kenya
    lakes_in_kenya = gpd.sjoin(lakes, kenya, how='inner', predicate='intersects')
    rivers_in_kenya = gpd.sjoin(rivers, kenya, how='inner', predicate='intersects')


    # Plotting
    fig, ax = plt.subplots(figsize=(10, 15))
    kenya.plot(ax=ax, color='white', edgecolor='black')
    lakes_in_kenya.plot(ax=ax, color='blue')  # Plot lakes in blue
    rivers_in_kenya.plot(ax=ax, color='blue')  # Plot rivers in blue
    large_cities_in_kenya.plot(ax=ax, marker='o', color='black', markersize=5)  # Plot large cities
    grid.plot(column='average_yield', ax=ax, cmap='RdYlGn', legend=True, 
              legend_kwds={'label': "Average Yield"},
              edgecolor='k', missing_kwds={'color': 'lightgrey'})

    # Annotate city names
    for x, y, label in zip(large_cities_in_kenya.geometry.x, large_cities_in_kenya.geometry.y, large_cities_in_kenya['NAME_left']):
        ax.text(x, y, label, fontsize=9, ha='right', va='top')

    plt.title('Kenya Mean Yield by %d x %d Grid Cell' % (grid_size, grid_size))
    plt.axis('equal')
    plt.savefig('Kenya_with_grids.png')
    plt.show()

    return grid

# Example usage assuming the data file and shapefile paths
stats = process_yield_data('data/kenya_yield_data.csv', 25)
print(stats)
