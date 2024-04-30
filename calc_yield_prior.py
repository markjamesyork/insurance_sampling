import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

def haversine(lat1, lon1, lats, lons):
    """
    Vectorized Haversine function to calculate distances between one point (lat1, lon1) a vector of points (lats, lons).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lats, lons = map(np.radians, [lat1, lon1, lats, lons])

    # Haversine formula
    dlat = lats - lat1
    dlon = lons - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lats) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Radius of earth in kilometers
    return km

def calculate_rmse_for_distances(yield_data, target_years, max_dists):
    rmse_results = {max_dist: [] for max_dist in max_dists}

    for target_year in target_years:
        print('target_year', target_year)
        target_data = yield_data[yield_data['year'] == target_year]
        other_years_data = yield_data[yield_data['year'] != target_year]

        # Pre-calculate distances for each target point to all other points from different years
        dist_matrix = np.array([haversine(row['latitude'], row['longitude'], other_years_data['latitude'].values, other_years_data['longitude'].values) for _, row in target_data.iterrows()])

        for max_dist in max_dists:
            predictions = []

            for dists in dist_matrix:
                # Filter points within max_dist and calculate average yield
                close_points_yields = other_years_data['yield'].values[(dists <= max_dist) & (dists > 0)]
                predicted_yield = close_points_yields.mean() if len(close_points_yields) > 0 else np.nan
                predictions.append(predicted_yield)

            # Calculate RMSE for this max_dist and target year
            valid_predictions = ~np.isnan(predictions)
            if np.any(valid_predictions):
                rmse = root_mean_squared_error(target_data['yield'].values[valid_predictions], np.array(predictions)[valid_predictions])
                rmse_results[max_dist].append(rmse)
            else:
                rmse_results[max_dist].append(np.nan)

    # Plotting RMSE for different max_dists across all years
    plt.figure(figsize=(10, 6))
    for max_dist, rmses in rmse_results.items():
        plt.plot(target_years, rmses, marker='o', label=f'Max Dist {max_dist} km')

    plt.xlabel('Year')
    plt.ylabel('RMSE')
    plt.title('Other Year RMSE by Year for Different Max Distances')
    plt.legend()
    plt.grid(True)
    plt.xticks(target_years)
    plt.show()

    #Print results
    for key in rmse_results.keys():
        print('Mean rmse for %d km distance is %f.' % (key, np.mean(rmse_results[key])))

    #Create and save a barchart of yield rmse for each max_dist setting
    # Sort max_dist in ascending order for plotting
    sorted_max_dists = sorted(rmse_results.keys())
    avg_rmses = [np.mean(rmse_results[max_dist]) for max_dist in sorted_max_dists]

    # Creating the line graph
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_max_dists, avg_rmses, marker='o', linestyle='-', color='skyblue')

    # Annotate each point with its RMSE value
    for i, value in enumerate(avg_rmses):
        plt.text(sorted_max_dists[i], value, f"{value:.2f}", ha='center', va='bottom')

    plt.xlabel('Maximum Distance (km)')
    plt.ylabel('Average RMSE')
    plt.title('Average Other Year RMSE vs Maximum Distance')
    plt.grid(True)

    # Save the chart
    plt.savefig('graphs/other_year_yield_rmse_vs_distance.png')

    return rmse_results


# Execution:
yield_data = pd.read_csv('data/kenya_yield_data.csv')  # Update with your actual file path
target_years = [2019, 2020, 2021, 2022, 2023]
max_dists = [10, 25, 50, 100, 150, 250, 500]  # Distances in kilometers
rmse_results = calculate_rmse_for_distances(yield_data, target_years, max_dists)

