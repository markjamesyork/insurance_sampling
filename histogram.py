import matplotlib.pyplot as plt
import pandas as pd

def plot_yield_histogram(df):
    # Plot and save the overall histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['yield'], bins='auto', color='skyblue', alpha=0.7)
    plt.title('All Iowa Yield Data Histogram')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('graphs/all_iowa_yield_hist.png', format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free memory for the next plots
    
    # Plot and save a histogram for each year
    for year in df['year'].unique():
        plt.figure(figsize=(10, 6))
        year_data = df[df['year'] == year]
        plt.hist(year_data['yield'], bins='auto', color='skyblue', alpha=0.7)
        plt.title(f'Iowa Yield Data Histogram for {year}')
        plt.xlabel('Yield')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f'graphs/Iowa_yield_hist_{year}.png', format='png', bbox_inches='tight')
        plt.close()  # Close the plot to free memory for the next plots

# Implementation
df = pd.read_csv('data/iowa_yield_data.csv')
plot_yield_histogram(df)
