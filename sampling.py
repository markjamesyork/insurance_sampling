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
import sys


class MaizeYieldSampler:
    def __init__(self):
        self.yield_data = None
        self.covariance_matrix = None
        self.conditional_mu = None
        self.conditional_sigma = None
        self.sampled_indices = set()
        self.predictions = []

    def read_yield_data(self, file_path):
        self.yield_data = pd.read_csv(file_path)

    def initialize_covariance(self, file_path, year):
        # Calculate the covariance matrix for a given year before any samples are taken

        # Read covariate data
        df = pd.read_csv(file_path)

        # Filter data to remove nans and data from target year
        if 'yield_data_year' in df.columns: df = df[df['yield_data_year'] == year] #Case where most locations only have data for one year
        df = df.dropna(subset=[str(year)]) #Remove rows which have no covariate data in the target year
        df = df.drop(str(year), axis=1) #Drop data from the target year
        reduced_df = df.iloc[:, 2:].to_numpy().T

        # Standardize the data
        scaler = StandardScaler().fit(reduced_df)
        standardized_data = scaler.transform(reduced_df)
        
        # Calculate the covariance matrix of the standardized data
        masked_data = np.ma.masked_array(standardized_data, np.isnan(standardized_data)) #mask nan values
        raw_covariance_matrix = np.ma.cov(masked_data, rowvar=False)

        # Use Higham approximation to find the closest valid PSD covariance matrix to the raw covariance matrix
        self.covariance_matrix = self.higham_approximation(raw_covariance_matrix)

    def next_sample_random(self, year):
        # Randomly select an unsampled index

        filtered_df = self.yield_data[self.yield_data.year == year]
        all_indices = set(filtered_df.index)
        remaining_indices = list(all_indices - self.sampled_indices)

        if remaining_indices:
            next_sample_id = np.random.choice(remaining_indices)
            yield_val = filtered_df.loc[next_sample_id, 'yield']
            self.sampled_indices.add(next_sample_id)
            return next_sample_id, yield_val
        else:
            return None, None

    def next_sample_mic_greedy(self, year):
        # Select the unsampled index which greedily maximizes mutual information

        filtered_df = self.yield_data[self.yield_data.year == year]
        filtered_df = filtered_df.reset_index(drop=True)
        all_indices = set(filtered_df.index)
        remaining_indices = list(all_indices - self.sampled_indices)
        sampled_indices = list(self.sampled_indices)

        if remaining_indices:
            # Calculate the mutual information gain from selecting each remaining index and track the higehst one
            Sigma_AA = self.covariance_matrix[np.ix_(sampled_indices, sampled_indices)]
            Sigma_AA_inv = np.linalg.inv(Sigma_AA)
            delta = -10**9
            for index in remaining_indices:
                # Create matrices needed to calculate mutual information
                Sigma_Ay = self.covariance_matrix[np.ix_([index], sampled_indices)] # Covariance matrix slice of A and y
                remaining_less_y = [i for i in remaining_indices if i != index] # Remaining indices with index 
                Sigma_woAy = self.covariance_matrix[np.ix_(remaining_less_y, remaining_less_y)] #Sigma without A and y
                Sigma_woAy_inv = np.linalg.inv(Sigma_woAy)
                Sigma_complAy = self.covariance_matrix[np.ix_([index], remaining_less_y)] # Covariance matrix slice of V \ (A U y) and y

                # Calculate delta of each sample and update
                if len(sampled_indices) == 0:
                    delta_tmp = self.covariance_matrix[index, index] \
                                / (self.covariance_matrix[index, index] - Sigma_complAy @ Sigma_woAy_inv @ Sigma_complAy.T)
                else:
                    delta_tmp = (self.covariance_matrix[index, index] - Sigma_Ay @ Sigma_AA_inv @ Sigma_Ay.T) \
                                / (self.covariance_matrix[index, index] - Sigma_complAy @ Sigma_woAy_inv @ Sigma_complAy.T)
                
                if delta_tmp[0][0] > delta:
                    delta = delta_tmp
                    next_sample_id = index

            # Pull yield value for selected sample and add to list of sampled indices
            yield_val = filtered_df.loc[next_sample_id, 'yield']
            self.sampled_indices.add(next_sample_id)
            return next_sample_id, yield_val

        else: #Return none if all indices have already been sampled
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
        Sigma22_inv = np.linalg.inv(Sigma22)
        updated_Sigma = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21

        self.conditional_sigma = updated_Sigma



# Execution:
# Set region-specific parameters
region = 'Iowa'
if region == 'Iowa':
    yield_data_path = 'data/iowa_yield_data.csv'
    covariate_data_path = 'data/iowa_yield_detrended.csv'
    max_samples = 99 #number of counties in Iowa

elif region == 'Kenya':
    yield_data_path = 'data/kenya_yield_data.csv'
    covariate_data_path = 'data/kenya_ndvi.csv'
    max_samples = 776 #number of samples in Kenya's lowest-sample year

else:
    print('The region you listed has not been paramaterized yet.')
    sys.exit()

# Load data and set region-agnostic parameters
sampler = MaizeYieldSampler()
sampler.read_yield_data(yield_data_path)
sampler.yield_data = sampler.yield_data.dropna()
sample_selection = 'mic_greedy'
years_to_sample = list(np.arange(1980, 2023))
n_reps = 1 # We only need n_reps > 0 when sample_selection == 'random'
rmse_matrix = np.zeros((len(years_to_sample), max_samples))
min_samples = max_samples #tracks the number of samples taken in the lowest-sample year

# Run simulation
for year in years_to_sample:
    print('Sampling %d' % year)
    sampler.initialize_covariance(covariate_data_path, year)
    #cov_matrix_df = pd.DataFrame(sampler.covariance_matrix)
    #cov_matrix_df.to_csv('data/covariance_matrix_2019.csv', index=False)
    #sampler.set_yield_priors(year)
    rep = 0
    true_yield_mean = sampler.yield_data[sampler.yield_data['year'] == year]['yield'].mean()
    sample_mean = np.zeros((max_samples, n_reps))

    while rep < n_reps:
        if rep % 13 == 0: print('Rep number %d' % rep)
        stop = False
        n_samples = 0
        samples = []
        sampler.conditional_sigma = sampler.covariance_matrix #reset conditional sigma matrix to prior covariance matrix before each sampling run

        while stop == False:
            #Pull Sample
            if sample_selection == 'random': next_sample_idx, yield_val = sampler.next_sample_random(year)
            elif sample_selection == 'mic_greedy': next_sample_idx, yield_val = sampler.next_sample_mic_greedy(year) #mic = mutual information criterion

            #Update Yield Estimates
            if yield_val == None: #Stop sampling once all samples have been taken
                stop = True
                if n_samples < min_samples: min_samples = n_samples #tracks the lowest number of samples taken in a year
                break

            samples += [yield_val]
            sample_mean[n_samples, rep] = np.mean(samples)

            #sampler.update_cov_matrix()

            #Stopping
            n_samples += 1
            if n_samples >= max_samples: stop = True

        sampler.sampled_indices = set() #reset to no samples taken
        rep += 1

    # Calc results
    squared_error = np.square(sample_mean - true_yield_mean)
    rmse = np.mean(squared_error, axis=1)**.5 #Calculate the root mean squared error of each yield estimate in sample_mean
    rmse_matrix[year - years_to_sample[0], :] = rmse

# Chart results
rmse_chart_vector = np.mean(rmse_matrix[:,:min_samples], axis=0)
print(rmse_chart_vector, rmse_chart_vector.shape)
plt.figure(figsize=(8, 6))
plt.plot(rmse_chart_vector, marker='o', linestyle='-', color='b')
plt.title('RMSE vs No. of Samples - %s Maize with %s sampling' % (region, sample_selection))
plt.xlabel('Number of samples taken')
if region == 'Iowa': plt.ylabel('RMSE in bushels per acre')
else: plot.ylabel('RMSE in tons per hectare')
plt.grid(True)
plt.savefig('graphs/%s_maize_yield_rmse_%s_sampling' % (region, sample_selection))
plt.show()

