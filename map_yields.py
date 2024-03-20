#Visualize Kenya Yield Data

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Function to create and save a map with individual yield points
def create_and_save_plot(data, year):
    plt.figure(figsize=(8, 8))
    m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=6, llcrnrlon=33, urcrnrlon=42, lat_ts=0, resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='lightgray',lake_color='aqua')

    # Convert latitude and longitude to x and y coordinates
    x, y = m(data['longitude'].values, data['latitude'].values)

    # Calculate the average yield and count of points
    avg_yield = data['yield'].mean()
    count_points = len(data)

    # Use a scatter plot to add the yield points; color by yield value
    scatter = m.scatter(x, y, c=data['yield'], cmap='RdYlGn', marker='o', edgecolor='k', linewidth=0.5)

    # Add a color bar and title
    plt.colorbar(scatter, label='Yield (mt/ha)')
    plt.title(f'Maize Yields in Kenya for {year}\nAverage Yield: {avg_yield:.2f} mt/ha, Count: {count_points}')
    plt.savefig(f'maps/maize_yields_{year}.png', format='png', bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Load data
df = pd.read_csv('data/kenya_yield_data_only.csv')

# Plot data for each year
years = df['year'].unique()
create_and_save_plot(df, 'all_years')
for year in years:
    yearly_data = df[df['year'] == year]
    create_and_save_plot(yearly_data, year)