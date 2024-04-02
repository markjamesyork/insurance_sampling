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
from calc_yield_prior import haversine #another script in this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler


class MaizeYieldSampler:
    def __init__(self):
        self.yield_data = None
        self.covariate_data = None
        self.covariance_matrix = None
        self.conditional_mu = None
        self.conditional_sigma = None
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
        raw_covariance_matrix =np.cov(standardized_data, rowvar=False)

        # Use Higham approximation to find the closest valid PSD covariance matrix to the raw covariance matrix
        self.covariance_matrix = self.higham_approximation(raw_covariance_matrix)

    def next_sample_random(self, year):
        # Randomly select an unsampled index
        filtered_df = self.yield_data[self.yield_data.year == year]
        all_indices = set(filtered_df.index) #set(range(len(self.yield_data[self.yield_data.year == year])))
        remaining_indices = list(all_indices - self.sampled_indices)
        if remaining_indices:
            next_sample_id = np.random.choice(remaining_indices)
            yield_val = filtered_df.loc[next_sample_id, 'yield']
            self.sampled_indices.add(next_sample_id)
            return next_sample_id, yield_val
        else:
            return None, None

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

    def set_yield_priors(self, year):
        '''Find prior as average yield for all fields within max_dist of target field in all years other than current year'''
        max_dist = 250 #maximum distance yields to consider, in km
        target_data = self.yield_data[self.yield_data['year'] == year]
        other_years_data = self.yield_data[self.yield_data['year'] != year]

        # Pre-calculate distances for each target point to all other points from different years
        dist_matrix = np.array([haversine(row['latitude'], row['longitude'], other_years_data['latitude'].values, other_years_data['longitude'].values) for _, row in target_data.iterrows()])

        prior_yields = []
        for dist in dist_matrix:
            # Filter points within max_dist and calculate average yield
            close_points_yields = other_years_data['yield'].values[dist <= max_dist]
            predicted_yield = close_points_yields.mean() if len(close_points_yields) > 0 else np.nan
            prior_yields.append(predicted_yield)

        self.conditional_mu = prior_yields


    def update_cov_matrix(self):
        #remaining_indices is a list of the indices of the counties which have not yet been measured
        remaining_indices = list(all_indices - self.sampled_indices)
        Sigma11 = self.covariance_matrix[np.ix_(remaining_indices, self.sampled_indices)]
        Sigma12 = self.covariance_matrix[np.ix_(remaining_indices, self.sampled_indices)]
        Sigma22 = self.covariance_matrix[np.ix_(measured_indices, self.sampled_indices)]
        Sigma21 = self.covariance_matrix[np.ix_(measured_indices, self.sampled_indices)]

        # Update the conditional mean and covariance
        #Alternative matrix inversion technique with np.linalg.solve(): https://stackoverflow.com/questions/31256252/why-does-numpy-linalg-solve-offer-more-precise-matrix-inversions-than-numpy-li
        #Sigma22_inv = np.linalg.inv(Sigma22)
        Sigma22_inv = np.linalg.solve(Sigma22, np.identity(Sigma22.shape[0]))
        updated_Sigma = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21

        self.conditional_sigma = updated_Sigma



# Execution:
yield_data_path = 'data/kenya_yield_data.csv'
covariate_data_path = 'data/kenya_ndvi.csv'

sampler = MaizeYieldSampler()
sampler.read_yield_data(yield_data_path)
sampler.read_covariate_data(covariate_data_path)
years_to_sample = [2023]
n_reps = 100
max_samples = 776

for year in years_to_sample:
    #sampler.initialize_covariance(year)
    #cov_matrix_df = pd.DataFrame(sampler.covariance_matrix)
    #cov_matrix_df.to_csv('data/covariance_matrix_2019.csv', index=False)
    sampler.set_yield_priors(year)
    rep = 0
    true_yield_mean = sampler.yield_data[sampler.yield_data['year'] == year]['yield'].mean()
    sample_mean = np.zeros((max_samples, n_reps))

    while rep < n_reps:
        stop = False
        n_samples = 0
        samples = []
        sampler.conditional_sigma = sampler.covariance_matrix #reset conditional sigma matrix to prior covariance matrix before each sampling run

        while stop == False:
            #Pull Sample
            next_sample_idx, yield_val = sampler.next_sample_random(year)

            #Update Yield Estimates
            samples += [yield_val]
            sample_mean[n_samples, rep] = np.mean(samples)

            #sampler.update_cov_matrix()

            #Stopping
            n_samples += 1
            if n_samples >= max_samples: stop = True

        sampler.sampled_indices = set()
        rep += 1

    #Calc results
    squared_error = np.square(sample_mean - true_yield_mean)
    rmse = np.mean(squared_error, axis=1)**.5 #Calculate the root mean squared error of each yield estimate in sample_mean

    # Create a plot of the column means
    plt.figure(figsize=(8, 6))
    plt.plot(rmse, marker='o', linestyle='-', color='b')
    plt.title('RMSE vs Number of Samples Taken - Kenya Maize 2019')
    plt.xlabel('Number of samples taken')
    plt.ylabel('RMSE in tons per hectare')
    plt.grid(True)
    plt.show()

