import pandas as pd
import numpy as np

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
    
    return array_3d

# Usage
filepath = 'data/kenya_yield_ndvi.xlsx'  # Replace with your file path
array_3d = read_excel_to_3d_array(filepath)
print(array_3d)
