import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def read_excel_to_3d_array(filepath):
    # Read the Excel file
    xls = pd.ExcelFile(filepath)
    
    # Initialize a list to hold the data from each sheet
    sheets_data = []

    # Iterate through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name)
        
        # Convert the DataFrame to a 2D array and add it to our list
        sheets_data.append(df.values)
    
    # Combine all the 2D arrays into a 3D array
    # Note: This assumes all sheets have the same shape. If they don't,
    # additional processing will be needed to handle this.
    array_3d = np.array(sheets_data)
    
    return array_3d, df.columns


def ndvi_rsq(array_3d, column_labels):
    # Reverse the order of the years so it matches the sheet order in array_3d
    years = [2023, 2022, 2021, 2020, 2019]  # Reversed corresponding years
    filtered_data = []
    for i, year in enumerate(years):
        # Access layers from first to fifth-to-last (assuming array_3d is in reversed yearly order)
        layer = array_3d[-(i + 1)]  # Changed from array_3d[-(i + 1)]
        df = pd.DataFrame(layer, columns=column_labels)  # Use provided column labels
        filtered_df = df[df['Year'] == year]  # Assuming 'year' is referenced directly
        filtered_data.append(filtered_df)

    concatenated_df = pd.concat(filtered_data)
    concatenated_df.to_csv('data.csv')  # Debugging: save the concatenated data

    # Prepare data for regression
    X_columns = concatenated_df.columns[6:]  # Excluding non-feature columns
    y = concatenated_df['Yield (Mt/'].values  # Direct reference to 'yield'

    r_squared_values = []
    column_number = []  # Storing column names instead of numbers for clarity

    for col in X_columns:
        X = concatenated_df[col].values.reshape(-1, 1)
        X = X.astype(float)
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        r_squared_values.append(r_squared)
        column_number.append(col)

    # 4. Create a line graph of R-squared values
    plt.figure(figsize=(10, 6))
    plt.plot(column_number, r_squared_values, marker='o')

    # Determine the number of labels and set a threshold for how many labels to show
    num_labels = len(column_labels[6:])
    step_size = max(int(num_labels / 10), 1)  # Adjust this to reduce clutter, showing only every nth label

    # Set x-axis labels, rotation, and show only every step_size label to prevent overcrowding
    plt.xticks(ticks=range(num_labels), labels=column_labels[6:], rotation=45, ha='right')
    for index, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if index % step_size != 0:
            label.set_visible(False)

    plt.xlabel('Variable')
    plt.ylabel('R-squared Value')
    plt.title('R-squared for Bi-Weekly NDVI Regressed Against Yield')
    plt.tight_layout()
    plt.savefig(f'graphs/ndvi_rsq.png', format='png', bbox_inches='tight')
    plt.show()

    # 5. Identify the column with the highest R-squared value
    max_r_squared_index = np.argmax(r_squared_values)
    best_column = column_labels[max_r_squared_index+6]
    print(f"The date with the best linear regression fit against 'yield' is: {best_column}")


def ndvi_sliding_window_rsq(array_3d, column_labels):
    # Prepare the data
    years = [2023, 2022, 2021, 2020, 2019]
    filtered_data = []
    for i, year in enumerate(years):
        layer = array_3d[-(i + 1)]
        df = pd.DataFrame(layer, columns=column_labels)
        filtered_df = df[df['Year'] == year]
        filtered_data.append(filtered_df)

    concatenated_df = pd.concat(filtered_data)

    # Response variable
    y = concatenated_df['Yield (Mt/'].values.astype(float)
    
    # Prepare for storing results
    results = {'Window Size': [], 'R-squared': [], 'Columns Averaged': []}

    # Compute average NDVI for sliding windows and regress against yield
    X_columns = concatenated_df.columns[6:]
    max_window = 12
    for window_size in range(2, max_window + 1):
        for start_col in range(len(X_columns) - window_size + 1):
            end_col = start_col + window_size
            averaged_ndvi = concatenated_df[X_columns[start_col:end_col]].astype(float).mean(axis=1).values.reshape(-1, 1)
            model = LinearRegression().fit(averaged_ndvi, y)
            r_squared = model.score(averaged_ndvi, y)
            # Store results
            results['Window Size'].append(window_size)
            results['R-squared'].append(r_squared)
            results['Columns Averaged'].append(X_columns[start_col] + " to " + X_columns[end_col - 1])

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('sliding_window_rsq.csv', index=False)
    
    # Plot the R-squared values for each window size
    plt.figure(figsize=(15, 6))
    for window_size in range(2, max_window + 1):
        window_results = results_df[results_df['Window Size'] == window_size]
        plt.plot(window_results['Columns Averaged'], window_results['R-squared'], marker='o', label=f'Window size {window_size}')

    plt.xticks(rotation=90)
    plt.xlabel('Columns Averaged')
    plt.ylabel('R-squared Value')
    plt.title('R-squared for Different NDVI Averages Regressed Against Yield')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/sliding_window_ndvi_rsq.png', format='png', bbox_inches='tight')
    plt.show()

    # Identify the best window and columns
    best_idx = results_df['R-squared'].idxmax()
    print(f"The best NDVI window is: {results_df.loc[best_idx, 'Columns Averaged']} with window size {results_df.loc[best_idx, 'Window Size']} yielding R-squared: {results_df.loc[best_idx, 'R-squared']}")


def average_columns_and_save(array_3d, column_labels):
    # Define the start and end dates for the averaging period
    start_column_index = 16
    end_column_index = 21

    # Initialize an empty DataFrame for storing the averages
    averages_df = pd.DataFrame()

    # Define the years for each layer
    years = list(range(2001, 2024))  # From 2001 to 2023

    # Iterate through each layer and compute the average for the specified dates
    for i, layer in enumerate(array_3d):
        # Convert the layer to a DataFrame
        df = pd.DataFrame(layer, columns=column_labels)
        
        # Calculate the average of the columns within the specified date range
        average_values = df.iloc[:, start_column_index:end_column_index + 1].mean(axis=1)
        new_df = df.iloc[:, start_column_index:end_column_index + 1]

        # Add the average values as a new column in the averages_df DataFrame
        averages_df[years[i]] = average_values

    # Save the DataFrame to a CSV file in the 'data/' directory
    averages_df.to_csv('data/ndvi_averages_june10_aug29.csv', index=False)



# Code Execution
#filepath = 'data/kenya_yield_ndvi.xlsx'  # Replace with your file path
#array_3d, column_labels = read_excel_to_3d_array(filepath)
#ndvi_rsq(array_3d, columns)
#ndvi_sliding_window_rsq(array_3d, column_labels)

