#This script reads covariance and yield data, implements a sampling strategy, and stores the results.

'''Sampling Steps:
1. Read yield and (if necessary) covariate data
2. For each year in target years:
	2.1 Create covariance matrix
	2.2 Determine which sample to take next
	2.3 Sample, update yield estimates and covariance matrix
	2.4 Determine whether stopping criteria have been met
	2.5 Repeat steps 2.2 - 2.4 until stopping
3. Create charts and any analysis; save
'''

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class MaizeYieldSampler:
    def __init__(self):
        self.yield_data = None
        self.covariate_data = None
        self.covariance_matrix = None
        self.sampled_indices = set()
        self.predictions = []

    def read_yield_data(self, file_path):
        self.yield_data = pd.read_csv(file_path)

    def read_covariate_data(self, file_path):
        self.covariate_data = pd.read_csv(file_path)

    def initialize_covariance(self, year):
        # Calculate the covariance matrix for a given year before any samples are taken
        filtered_df = self.covariate_data[self.covariate_data['yield_data_year'] == year]
        reduced_df = filtered_df.iloc[:, 2:].to_numpy().T
        # Standardize the data
        scaler = StandardScaler().fit(reduced_df)
        standardized_data = scaler.transform(reduced_df)
        
        # Calculate the covariance matrix of the standardized data
        self.covariance_matrix = np.cov(standardized_data, rowvar=False)

    def next_sample_random(self):
        # Randomly select an unsampled index
        all_indices = set(range(len(self.yield_data)))
        remaining_indices = list(all_indices - self.sampled_indices)
        if remaining_indices:
            next_sample = np.random.choice(remaining_indices)
            self.sampled_indices.add(next_sample)
            return next_sample
        else:
            return None

    def update_predictions(self):
        # Update model's predictions based on newly selected samples
        pass

    def calculate_rmse(self):
        # Calculate RMSE between actual yields and your model's predictions
        if self.predictions:
            actual = self.yield_data.iloc[list(self.sampled_indices)]
            predicted = [self.predictions[i] for i in self.sampled_indices]
            return mean_squared_error(actual, predicted, squared=False)
        else:
            return None

    def higham_approximation(self, A):
        # Finds the nearest positive semidefinite matrix to A using CVXPY.
        #A = A_df.values
        n = A.shape[0]
        A_sym = (A + A.T) / 2
        X = cp.Variable((n, n), symmetric=True)
        objective = cp.Minimize(cp.norm(X - A_sym, 'fro'))
        constraints = [X >> 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return X.value

# Example usage:
yield_data_path = 'data/kenya_yield_data.csv'
covariate_data_path = 'data/kenya_ndvi.csv'

sampler = MaizeYieldSampler()
sampler.read_yield_data(yield_data_path)
sampler.read_covariate_data(covariate_data_path)
sampler.initialize_covariance(2019)
cov_matrix_df = pd.DataFrame(sampler.covariance_matrix)
cov_matrix_df.to_csv('data/covariance_matrix_2019.csv', index=False)

higham_approx = sampler.higham_approximation(sampler.covariance_matrix)
higham_approx_df = pd.DataFrame(higham_approx)
higham_approx_df.to_csv('data/higham_approx_2019.csv', index=False)


distance = np.linalg.norm(sampler.covariance_matrix - higham_approx, 'fro')
print(f"Distance between original and Higham approximation: {distance}")
