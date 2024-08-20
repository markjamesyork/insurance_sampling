#This script reads covariance and yield data, implements a sampling strategy, and stores the results.

from christofides import*
from shortest_path import*
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import sys


class MaizeYieldSampler:
    def __init__(self):
        self.yield_data = None
        self.covariance_matrix = None
        self.conditional_mu = None
        self.conditional_sigma = None
        self.sampled_indices = []
        self.predictions = []
        self.cost_per_sample_measurement = 10
        self.cost_per_km_traveled = 10

    def read_yield_data(self, file_path):
        self.yield_data = pd.read_csv(file_path)

    def initialize_covariance(self, file_path, year):
        # Calculate the covariance matrix for a given year before any samples are taken
        # Note that the set of location IDs in the covariate data MUST BE EXACTLY THE SAME as the set of locations in the yield data

        # Read covariate data
        df = pd.read_csv(file_path)

        # Filter data to remove nans and data from target year and later
        if 'yield_data_year' in df.columns: df = df[df['yield_data_year'] == year] #Case where most locations only have data for one year
        df = df.dropna(subset=[str(year)]) #Remove rows which have no covariate data in the target year
        for future_year in range(year, 2040):
            if str(year) in df.columns:
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
        #self.covariance_matrix = pd.read_csv('data/cov_tmp.csv').to_numpy()

    def next_sample_random(self, year):
        # Randomly select an unsampled index

        filtered_df = self.yield_data[self.yield_data.year == year]
        all_indices = set(filtered_df.index)
        remaining_indices = list(all_indices - set(self.sampled_indices))

        if remaining_indices:
            next_sample_id = np.random.choice(remaining_indices)
            yield_val = filtered_df.loc[next_sample_id, 'yield']
            self.sampled_indices += [next_sample_id]
            return next_sample_id, yield_val
        else:
            return None, None

    def next_sample_mic_greedy(self, year):
        # Select the unsampled index which greedily maximizes mutual information

        # Define sets of indices
        filtered_df = self.yield_data[self.yield_data.year == year]
        #filtered_df = filtered_df.reset_index(drop=True)
        all_indices = set(filtered_df.index)
        remaining_indices = list(all_indices - set(self.sampled_indices))

        if remaining_indices:
            next_sample_id = highest_mutual_information_gain_sample(self.covariance_matrix, all_indices, self.sampled_indices, remaining_indices)

            # Pull yield value for selected sample and add to list of sampled indices
            yield_val = filtered_df.loc[next_sample_id, 'yield']
            self.sampled_indices += [next_sample_id]
            return next_sample_id, yield_val

        else: #Return none if all indices have already been sampled
            return None, None

    def next_sample_gp_mic_greedy(self, year, gp):
        # Use gp to determine covariance matrix, mutual information, and next sample to greedily select

        # Define sets of indices
        filtered_df = self.yield_data[self.yield_data.year == year]
        all_indices = set(filtered_df.index)
        remaining_indices = list(all_indices - set(self.sampled_indices))

        # Compute the covariance matrix of points using the gp kernel
        if gp != None:
            X_points = filtered_df[['latitude', 'longitude']].values
            covariance_matrix = gp.kernel_(X_points)

        # Find the next sample which maximizes the mutual information
        if remaining_indices:
            # Find the next sample
            if gp != None and np.linalg.det(covariance_matrix) != 0:
                next_sample_id = self.highest_mutual_information_gain_sample(covariance_matrix, all_indices, self.sampled_indices, remaining_indices)
            else: next_sample_id = random.choice(remaining_indices) # If no gaussian process is passed, return a random index

            # Pull yield value for selected sample and add to list of sampled indices
            yield_val = filtered_df.loc[next_sample_id, 'yield']
            self.sampled_indices += [next_sample_id]
            return next_sample_id, yield_val

        else: #Return none if all indices have already been sampled
            return None, None

    def highest_mutual_information_gain_sample(self, cov, all_indices, remaining_indices):
        # Find the unsampled index with the highest mutual information gain

        # Core calculations
        Sigma_AA = cov[np.ix_(self.sampled_indices, self.sampled_indices)]
        Sigma_AA_inv = np.linalg.inv(Sigma_AA)
        delta = -10**9
        for index in remaining_indices:
            # Create matrices needed to calculate mutual information
            Sigma_Ay = cov[np.ix_([index], self.sampled_indices)] # Covariance matrix slice of A and y
            remaining_less_y = [i for i in remaining_indices if i != index] # Remaining indices with index 
            Sigma_woAy = cov[np.ix_(remaining_less_y, remaining_less_y)] #Sigma without A and y
            Sigma_woAy_inv = np.linalg.inv(Sigma_woAy)
            Sigma_complAy = cov[np.ix_([index], remaining_less_y)] # Covariance matrix slice of V \ (A U y) and y

            # Calculate delta of each sample and update
            if len(self.sampled_indices) == 0:
                delta_tmp = cov[index, index] \
                            / (cov[index, index] - Sigma_complAy @ Sigma_woAy_inv @ Sigma_complAy.T)
            else:
                delta_tmp = (cov[index, index] - Sigma_Ay @ Sigma_AA_inv @ Sigma_Ay.T) \
                            / (cov[index, index] - Sigma_complAy @ Sigma_woAy_inv @ Sigma_complAy.T)
            
            if delta_tmp[0][0] > delta:
                delta = delta_tmp
                next_sample_id = index

        return next_sample_id

    def update_predictions(self):
        # Update model's predictions based on newly selected samples
        pass

    def calculate_rmse(self):
        # Calculate RMSE between actual yields and your model's predictions
        if self.predictions:
            actual = self.yield_data.iloc[self.sampled_indices]
            predicted = [self.predictions[i] for i in self.sampled_indices]
            return mean_squared_error(actual, predicted, squared=False)
        else:
            return None

    def interim_estimation_error_loss(self, inferred_yield, true_yield_mean, expected_yield, trend_yield):
        # This function calculates the loss to an insurer for a given estimate. We assume that overpayment and underpayment incur equal cost (e.g. Beta = 1)
        loss_threshold = .85
        guarantee = loss_threshold * (expected_yield + trend_yield) # Numpy array of size [n_years, 1]
        actual_payout = guarantee - np.clip(inferred_yield + trend_yield, 0, guarantee) # Amount the insurer pays the insuree
        ideal_payout = np.clip(guarantee - (true_yield_mean + trend_yield), 0, None) # Amount the insurer would pay the insuree with perfect knowledge of the yield
        loss = np.abs(actual_payout - ideal_payout)

        return loss, actual_payout

    def interim_insurer_loss(self, samples):
        # This function calculates the full insurer loss in expectation before the samples are taken or final yield is known

        # Parameter settings
        per_sample_measurement_cost = 15 # Two workers, two hours, $3.75 per hour, in USD
        per_km_travel_cost = 0.7 # 30 km / hr, 7 km per liter, repairs = fuel * .5, 2 employees @ $3.75/hr, full calc in writeup, in USD
        cost_per_unit_est_error = n_farms * 2 * 250 # Assumes that the average farm size is two hectares, and that the price of grain is $250 per ton (approximately the average 2024 price of maize in Ghana)
        losses = []

        # Create lats / lons array
        filtered_df = sampler.yield_data.loc[indices]
        lat_lon_df = filtered_df[['latitude', 'longitude']]
        lat_lon_array = lat_lon_df.to_numpy()

        # Distance traveled
        distance_traveled, path = tsp(lat_lon_array)

        # Estimation error when loss final yield is known
        interim_estimation_error_loss = interim_estimation_error_loss(inferred_yield_vec[i], true_yield_mean, expected_yield)

        # Add list of all loss types to losses list
        loss += len(samples) * per_sample_measurement_cost + \
                    distance_traveled * per_km_travel_cost +\
                    final_estimation_error_loss    

        return loss

    def final_estimation_error_loss(self, inferred_yield, true_yield, expected_yield):
        # This function calculates the loss to an insurer for a given estimate. We assume that overpayment and underpayment incur equal cost (e.g. Beta = 1)
        loss_threshold = .85
        guarantee = loss_threshold * expected_yield # Numpy array of size [n_years, 1]
        actual_payout = guarantee - np.clip(inferred_yield, 0, guarantee) # Amount the insurer pays the insuree
        ideal_payout = np.clip(guarantee - true_yield, 0, None) # Amount the insurer would pay the insuree with perfect knowledge of the yield
        loss = np.abs(actual_payout - ideal_payout)

        return loss, actual_payout

    def final_insurer_loss(self, samples, n_farms, inferred_yield_vec, true_yield, expected_yield):
        # This function calculates the full insurer loss after the final yield is known. It removes a list of lists, where list i
        # contains the losses after sample i is taken, and each element of the list represents a sub-category of loss
        # [sample-taking cost, travel cost, measurement error cost]

        # Parameter settings
        per_sample_measurement_cost = 15 # Two workers, two hours, $3.75 per hour, in USD
        per_km_travel_cost = 0.7 # 30 km / hr, 7 km per liter, repairs = fuel * .5, 2 employees @ $3.75/hr, full calc in writeup, in USD
        cost_per_unit_est_error = n_farms * 2 * 250 # Assumes that the average farm size is two hectares, and that the price of grain is $250 per ton (approximately the average 2024 price of maize in Ghana)
        losses = []

        # Create lats / lons array
        filtered_df = sampler.yield_data.loc[indices]
        lat_lon_df = filtered_df[['latitude', 'longitude']]
        lat_lon_array = lat_lon_df.to_numpy()

        for i in range(len(samples)):
            # Distance traveled
            distance_traveled, path = tsp(lat_lon_array[:i+1,:])

            # Estimation error when loss final yield is known
            final_estimation_error_loss, actual_payouts = final_estimation_error_loss(inferred_yield_vec[i], true_yield_mean, expected_yield)

            # Add list of all loss types to losses list
            losses += [i * per_sample_measurement_cost, \
                        distance_traveled * per_km_travel_cost, \
                        final_estimation_error_loss]

        return losses # [in-field measurement cost, travel cost, estimation error loss]

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

    def update_mu_sigma(self, year, observed_index, observed_value):
        """
        Update the mean vector and covariance matrix after observing a new sample.
        Assumes that self.covariance has the location_id sorted lexicographically
        
        Parameters:
        year = target year for which we are sampling
        observed_index (int): Index of the observed variable.
        observed_value (float): Value of the observed variable.
        """

        # Filter yield data for this year only
        filtered_df = self.yield_data[self.yield_data.year == year]

        # Find indices for sample to pull and for remaining samples within self.conditional_sigma 
        all_indices = set(filtered_df.index)
        remaining_indices = all_indices - (set(self.sampled_indices) - {observed_index}) # Set contains current sample to evaluate
        obs_index_in_remaining = list(remaining_indices).index(observed_index) # Index of the currently-observed index within all remaining unsampled indices
        remaining_less_obs = list(remaining_indices - {observed_index})

        # Extract relevant parts of the covariance matrix
        sigma_ii = self.conditional_sigma[obs_index_in_remaining, obs_index_in_remaining]
        sigma_oi = np.delete(self.conditional_sigma[:, obs_index_in_remaining], obs_index_in_remaining, 0)
        sigma_oo = np.delete(np.delete(self.conditional_sigma, obs_index_in_remaining, 0), obs_index_in_remaining, 1)
        
        # Compute the Schur complement to update the covariance matrix for unobserved variables
        self.conditional_sigma = sigma_oo - np.outer(sigma_oi, sigma_oi.T) / sigma_ii
        
        # Update the mean vector for unobserved variables
        mu_i = self.conditional_mu[obs_index_in_remaining]
        mu_o = np.delete(self.conditional_mu, obs_index_in_remaining, 0)
        self.conditional_mu = mu_o + (observed_value - mu_i) * (sigma_oi / sigma_ii)

        return


def GP_inference(sampler, year): #sampler is a class
    # Fit a Gaussian Process and infer the yields of unmeasured samples

    # Find indices of what has been sampled and what has not
    filtered_df = sampler.yield_data[sampler.yield_data.year == year]
    all_indices = set(filtered_df.index)
    remaining_indices = list(all_indices - set(sampler.sampled_indices))

    # Slice data according to what has beene sampled
    filtered_df.to_csv('filtered_df.csv')
    df_sampled = filtered_df.reindex(sampler.sampled_indices) # DataFrame of measured samples
    df_sampled.to_csv('df_sampled.csv')
    df_unsampled = filtered_df.drop(sampler.sampled_indices)  # DataFrame of unmeasured samples

    # Fit GP
    gp = fit_GP(df_sampled)

    # Calculate predicted yields
    X_unsampled = df_unsampled[['latitude', 'longitude']].values
    y_pred = gp.predict(X_unsampled, return_std=False)

    return np.mean(np.hstack((y_pred, df_sampled['yield'].values))), gp


def fit_GP(df_sampled):
    # Fit a Gaussian Process to a set of measured samples

    # Prepare the data: Extract coordinates and yields
    X = df_sampled[['latitude', 'longitude']].values
    y = df_sampled['yield'].values

    # Kernel with RBF and adjustable constant
    kernel = C(1.0, (1e-6, 1e1)) * RBF([1, 1], (1e-3, 1e4))

    # Create Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True)

    return gp.fit(X, y) # Returns fit Gaussian Process


def simulate(region, sampling_method, inference_method, n_reps, years_to_sample, log_yield, loss_type):
    # This function calls the MaizeYieldSampler class for various settings of sampling and inference methods

    # Create region-specific settings
    if region == 'Iowa':
        yield_data_path = 'data/iowa_yield_data.csv'
        covariate_data_path = 'data/iowa_yield_detrended.csv'
        max_samples = 99 #number of counties in Iowa

    elif region == 'Kenya':
        yield_data_path = 'data/kenya_yield_data.csv'
        covariate_data_path = 'data/kenya_ndvi.csv'
        max_samples = 100 #number of samples in Kenya's lowest-sample year

    elif region == 'Synthetic':
        yield_data_path = 'data/synthetic_yield_data_column.csv'
        covariate_data_path = 'data/synthetic_yield_data_matrix.csv'
        max_samples = 100

    else:
        print('Region %s is not yet programmed in.' % region)
        sys.exit()

    # Set intermediate variables to be modified during the simulation
    if sampling_method != 'random': n_reps = 1 # Only have more than one rep for non-deterministic sampling methods
    min_samples = max_samples #tracks the number of samples taken in the lowest-sample year
    inferred_yield = np.zeros((max_samples, n_reps, len(years_to_sample)))
    sample_mean = np.zeros((max_samples, n_reps, len(years_to_sample)))
    true_yield_vector = []
    insurer_losses = None

    # Load data
    sampler = MaizeYieldSampler()
    sampler.read_yield_data(yield_data_path)
    sampler.yield_data = sampler.yield_data.dropna()
    if log_yield == True: # Makes lognormal data look normal
        sampler.yield_data['yield'] = np.log1p(sampler.yield_data['yield']) # np.log1p(x) is equivalent to np.log(1 + x) and is more numerically stable

    # Calculate expected yield based on yield data from all years before target year:
    expected_yield = []
    for year in years_to_sample:
        if 'region' == 'Kenya' and year == 2019: expected_yield += [1.7005133] # 2019 trendline Kenyan maize yield based on 2009-2018 FAOSTAT data. The 2009-2018 mean was 1.62366. This is needed because there is no yield data before 2019.
        else: expected_yield += [np.mean(sampler.yield_data[sampler.yield_data['year'] < year]['yield'])]
    print('expected_yield', expected_yield)

    # Run simulation
    for year in years_to_sample:
        print('Sampling %d' % year)
        if sampling_method == 'mic_greedy' or inference_method == 'normal_inference': sampler.initialize_covariance(covariate_data_path, year)
        target_year_yields = sampler.yield_data[sampler.yield_data['year'] == year]['yield']
        true_yield = target_year_yields.mean()
        true_yield_vector += [true_yield]
        expected_yield = sampler.yield_data[sampler.yield_data['year'] != year]['yield'].mean() #calculates expected yield as the mean of yield from all years other than target year
        rep = 0

        # If using a Gaussian Process for sample selection, set an initial default
        if sampling_method == 'GP_mic_greedy':
            if year == years_to_sample[0]:
                gp = None # Passes None to the sample selection function, which will then select a random sample
            else:
                df_past_year = sampler.yield_data[sampler.yield_data.year == (year - 1)]
                gp = fit_GP(df_past_year) # Fits a gaussian process to last year's yield data

        # Check that max_samples is not greater than the number of samples in target year
        target_year_samples = len(target_year_yields)
        if target_year_samples < min_samples: min_samples = target_year_samples

        # Main repetitive sampling loop
        while rep < n_reps:
            # Reset conditional sigma and mu before each sampling run
            if (sampling_method == 'mic_greedy' or inference_method == 'normal_inference'):
                sampler.conditional_sigma = sampler.covariance_matrix 
                sampler.conditional_mu = np.zeros((sampler.covariance_matrix.shape[0],)) # Set conditional mean at zeros - works for detrended yield data as in Iowa
            if rep % 7 == 0: print('Rep number %d' % rep)
            stop = False
            n_samples = 0
            sample_vals = []
            insurer_losses_one_rep = None

            while stop == False:
                # Pull Sample
                if sampling_method == 'random': next_sample_idx, yield_val = sampler.next_sample_random(year)
                elif sampling_method == 'mic_greedy': next_sample_idx, yield_val = sampler.next_sample_mic_greedy(year) #mic = mutual information criterion
                elif sampling_method == 'GP_mic_greedy': next_sample_idx, yield_val = sampler.next_sample_gp_mic_greedy(year, gp)
                sample_vals += [yield_val]

                # Updating yield estimate
                sample_mean[n_samples, rep, year-years_to_sample[0]] = np.mean(sample_vals)
                if inference_method == 'sample_mean':
                    inferred_yield[n_samples, rep, year-years_to_sample[0]] = sample_mean[n_samples, rep, year-years_to_sample[0]]
                elif inference_method == 'normal_inference':
                    sampler.update_mu_sigma(year, next_sample_idx, yield_val)
                    inferred_yield[n_samples, rep, year-years_to_sample[0]] = np.mean(np.hstack((sampler.conditional_mu, np.asarray(sample_vals)))) # Averages sample measurements and conditional mu for unmeasured samples
                elif inference_method == 'GP_inference':
                    inferred_yield[n_samples, rep, year-years_to_sample[0]], gp = GP_inference(sampler, year)

                # Stopping due to max_samples being reached
                n_samples += 1
                if n_samples >= min(target_year_samples, max_samples): stop = True

            # Calculating insurer loss
            if loss_type == 'insurer_loss':
                tmp_losses = final_full_insurer_loss(sampler.sampled_indices, target_year_yields.shape[0], \
                            inferred_yields[:, rep, year-years_to_sample[0]], true_yield, expected_yield) # [in-field measurement cost, travel cost, estimation error loss]
                if insurer_losses_one_rep == None: insurer_losses = np.asarray(tmp_losses)
                else:
                    tmp_losses = np.asarray(tmp_losses)
                    continue
                    '''for ChatGPT: "Please write python code which stacks a new 2D numpy array onto another 2D numpy along the 2 axis, resulting in a 3D numpy array. This code should also work if the base array upon which the new array is being stacked is already 3D, with the first two dimensions being the same as the two dimensions of the new array. The base array is called "insurer_losses", and the new array is called "tmp_losses.""
                    '''

            sampler.sampled_indices = [] #reset to no samples taken
            rep += 1
            if loss_type == 'insurer_loss':
                insurer_losses_one_rep == None #For ChatGPT: "Please write Python code which takes the mean a 3D numpy array along axis 2, resulting in a 2D array."
                if insurer_losses == None: insurer_losses = insurer_losses_one_rep
                else:
                    # Stack arrays here. # Resulting array has shape (3, n_samples, n_years)
                    continue

    # Return predicted yield results
    return inferred_yield, sample_mean, true_yield_vector, min_samples, expected_yield, insurer_losses





def plot_line_graph(var_1, var_2, x_label='X-axis', y_label='Y-axis', title='Line Graph', var_1_label='Variable X', var_2_label='Variable Y'):
    """
    Plots a line graph for two variables passed as numpy arrays.

    Parameters:
    
    x (numpy array): Data for the x-axis.
    y (numpy array): Data for the y-axis.
    
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title of the graph.
    label_x (str): Legend label for the x variable.
    label_y (str): Legend label for the y variable.
    """
    plt.figure(figsize=(10, 6))  # Create a new figure with a custom size
    plt.plot(var_1, label=var_1_label, marker='o', linestyle='-')  # Plot the x variable
    plt.plot(var_2, label=var_2_label, marker='o', linestyle='-')  # Plot the y variable
    plt.xlabel(x_label)  # Set the label for the x-axis
    plt.ylabel(y_label)  # Set the label for the y-axis
    plt.title(title)  # Set the title of the graph
    plt.legend()  # Add a legend to the graph
    plt.grid(True)  # Enable grid for better readability
    plt.show()  # Display the graph



# EXECUTION:

# 0 Simulation Settings
region = 'Kenya'
log_yield = True # If True, read yield data as log(1 + yield). This is to make lognormal data look more normal
sampling_methods = ['random', 'GP_mic_greedy'] # ['random'] # Options: 'random', 'mic_greedy', 'GP_mic_greedy'
inference_methods = ['sample_mean', 'GP_inference'] # Options: 'sample_mean', 'normal_inference', 'GP'
n_reps = 10
loss_type = 'RMSE' # Options: 'RMSE', 'insurer_loss'
years_to_sample = np.arange(2019, 2022)
inferred_yield_dict = {}
loss_dict = {}
insurer_loss_dict = {}
if 'region' == 'Iowa': trend_params = [-4292.5011, 2.21671781] # Needed to calculate insurance threshold for insurer loss; [-4292.5011, 2.21671781] for Iowa, [0, 0] for Kenya
else: trend_params = [0, 0]

# 1 Simulation Loop
for sampling_method in sampling_methods:
    for inference_method in inference_methods:
        print('inference_method', inference_method)
        inferred_yield_dict[sampling_method + '_' + inference_method], sample_mean, true_yield, min_samples, expected_yield, insurer_losses = \
                    simulate(region, sampling_method, inference_method, n_reps, years_to_sample, log_yield, loss_type)
        if loss_type == 'insurer_loss':
            insurer_loss_dict[sampling_method + '_' + inference_method] = insurer_losses


        inferred_chart_array = inferred_yield_dict[sampling_method + '_' + inference_method].mean(axis=(1,2))
        if inference_method != 'sample_mean':
            sample_mean_chart_array = sample_mean.mean(axis=(1,2))
            plot_line_graph(inferred_chart_array, sample_mean_chart_array, x_label='n samples', y_label='Estimate vs. trend, tons/ha', title='Yield Estimates vs. n Samples', var_1_label='GP Inference', var_2_label='Sample Mean')

# 2 Loss Calculation
for sampling_inference in inferred_yield_dict.keys(): # Runs once for each sampling method / inference method pair
    if loss_type == 'RMSE':
        print('true_yield', true_yield)
        squared_error = np.square(inferred_yield_dict[sampling_inference] - true_yield) # Dimension: max_samples, n_runs, n_years
        loss_tmp = np.mean(squared_error, axis=1)**.5 #Calculate the root mean squared error of each yield estimate in sample_mean

    if loss_type == 'insurer_loss':
        trend_yield = np.asarray([trend_params[0] + trend_params[1] * year for year in years_to_sample]) # This is the base trendline yield that should be added to expected yield. If yield data is not trend-adjusged, this will be zero.

    loss_dict[sampling_inference] = np.mean(loss_tmp[:min_samples, :], axis=1)

# 3 Charting & Saving Results
# For ChatGPT: 'Please write python code which makes a stacked area chart with three layers. The code will intake a 3D numpy array, and average across dimension 2. It will then chart such that the layers are in dimension 0, and the x axis is along dimension 1. The height of each layer on the chart is the values in the numpy array.'
# For ChatGPT: 'Please write python code which takes a 3D array and saves it as a .csv file titled 'yyyymmdd_str_.csv'
# For ChatGPT: Please write a numerical integration in python for an integral which can be evaluated at any x value. Then write such a function for a double integral whose value we can assess at any (x,y) value.
# For ChatGPT: Please write a genetic algorithm solver for a function which takes a binary vector of 1s and 0s as input.
# Jake Jilinhoff

# Number of samples taken
samples = np.arange(1, min_samples + 1)




# Plotting RMSE Alone
# Plotting RMSE Comparison
plt.figure(figsize=(8, 6))
for sampling_inference in loss_dict.keys():
    plt.plot(samples, loss_dict[sampling_inference], marker='o', linestyle='-', label=sampling_inference)
#plt.plot(samples, rmse_inference_vector, marker='o', linestyle='-', color='r', label='Inference RMSE')
title_str = list(loss_dict.keys())[0]
for key in list(loss_dict.keys())[1:]: title_str += (' vs. ' + key)
plt.title('%s Maize %s loss: %s' % (region, loss_type, title_str))
plt.xlabel('Number of samples taken')
plt.ylabel('%s in bushels per acre' % loss_type if region == 'Iowa' else '%s in tons per hectare' % loss_type)
plt.grid(True)
plt.legend()
plt.savefig('graphs/%s_maize_%s_%s.png' % (region, loss_type, title_str))
#plt.show()
