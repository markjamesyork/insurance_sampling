import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import certifi
import ssl
import numpy as np

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Read the csv file into a DataFrame
df = pd.read_csv('data/us_county_corn_production_yield.csv')

# Print the first 10 rows
#print(df.head(10))

# Filter the DataFrame for entries with the "State" field equal to "IOWA"
filtered_df = df[df['State'] == 'IOWA']

# Sort the results by "Year"
sorted_df = filtered_df.sort_values(by='Year')
sorted_df.to_csv('output_file.csv', index=False)

# Calculate the covariance matrix for the values in "PYLD" for each unique county
sorted_df.pivot_table(index='Year', columns='County', values='PYLD').to_csv('output_file.csv', index=False)
cov_matrix = sorted_df.pivot_table(index='Year', columns='County', values='PYLD').cov()

# 1. Calculate the average PYLD by county
avg_yield = sorted_df.groupby('County')['PYLD'].mean().reset_index()

# 2. Save the averages to a .csv file
avg_yield.to_csv('county_avg_yield.csv', index=False)

# 3. Plot a color gradient map for Iowa counties

# Load Iowa counties shapefile (geopandas has some datasets built-in, but a detailed shapefile for Iowa might need to be loaded separately)
#url = "https://raw.githubusercontent.com/deldersveld/topojson/master/countries/us-states/IA-19-iowa-counties.json"
url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
us_map = gpd.read_file(url)
us_map.to_csv('us_map.csv', index=False)
iowa_map = us_map[us_map['STATE'] == '19']
iowa_map['NAME'] = iowa_map['NAME'].str.upper().str.replace("'" , " ")

iowa_map.to_csv('iowa_map.csv', index=False)

# Merge the geodataframe with the average yield dataframe

merged = iowa_map.set_index('NAME').join(avg_yield.set_index('County'))
print('merged',merged)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged.plot(column='PYLD', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Average PYLD by County in Iowa')
plt.savefig('iowa_avg_yield.png', format='png', dpi=600)  # Adjust dpi as needed for resolution
plt.show()

