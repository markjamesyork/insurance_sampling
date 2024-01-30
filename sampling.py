#This script implements various sampling algorithms for yield measurement
import csv
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import lognorm
from scipy.optimize import minimize
import sys


def simulate_sampling():
	#0 Parameter settings
	sample_noise = .0 #Standard deviation of Gaussian noise as a fraction of the trend-line yield
	samples_per_run = 99 #maximum number of counties to sample, without replacement
	n_runs = 1
	years_to_run = np.arange(1980, 2023) #last year to run is one before last year written here
	state = 'Iowa'
	sampling_methods = ['random'] #['random']#, 'sse']#, 'overall_variance'] #options are 'random', 'sse', 'preset' and 'overall_variance'
	preset_sample_order = [9, 49, 18, 72, 67, 84, 65, 22, 92, 3, 98, 80, 40, 31, 10, 70, 96, 97, 30]
	yield_estimates = {}
	synthetic_yield_data = False
	cov_from_file = True

	#1 Read yield variable data
	if synthetic_yield_data == False:
		df, state_yield = read_and_sort_yield_data(state.upper()) #This function reads actual yield data
	else:
		df = pd.read_csv('synthetic_yield_data.csv') #This function reads synthetic yield data
		state_yield = df.mean(axis=1)

	df, trend_df = detrend_yield_data(df) #df is now detrended
	#df.to_csv('detrended.csv')

	#2 Find negative lognormal parameters and convert detrended data to standard normal
	county_params = []
	for column in df.columns:
		column_array = df[column].to_numpy()
		county_params += [fit_neg_lognormal_params(column_array)]
		df[column] = (np.log(county_params[-1][2] - column_array) - county_params[-1][0]) / county_params[-1][1] #county_params = [mu, sigma, max_value] for each county
	
	if cov_from_file == False:
		cov = df.cov()
	else:
		cov = pd.read_csv('cov_higham_approx.csv')


	cov.to_csv('neg_lognormal_covariance.csv', index=False)

	county_params_df = pd.DataFrame(np.asarray(county_params))
	county_params_df.to_csv('county_params_new.csv')
	df.to_csv('lognormal_standardized.csv')
	county_params = np.asarray(county_params).T #convert county_params from a list of lists to an array of shape (3, n_counties). Values in one-county's row are [mu, sigma, max_value]

	'''
	#Only use selected columns of yield data
	selected_columns = [0, 1, 2, 3, 49]
	df = df.iloc[:, selected_columns]
	'''

	#Find the maximum number of samples to take as the minimum number of counties with valid yields within the set of target years
	for year in years_to_run:
		non_nan_columns = df.loc[year].dropna().index.tolist() # Check which columns do not have a NaN value for the given 'year'
		if samples_per_run > len(non_nan_columns): samples_per_run = len(non_nan_columns)

	#3 Loop through years to run and create yield estimates
	for method in sampling_methods:
		print('Method: ', method)
		if method != random: yield_estimates[method] = np.empty((len(years_to_run), len(df.columns), samples_per_run, 1)) #shape (n_years, n_counties, n_samples_taken, n_runs)
		else: yield_estimates[method] = np.empty((len(years_to_run)), len(df.columns), samples_per_run, n_runs) #only include multiple runs if the sample selection method is random
		yield_estimates[method][:] =  np.nan
		for year in years_to_run:
			print('Year: ', year)
			df_sans_year = df[df.index != year]
			#df_sans_year, original_df_detrended, trend_df = detrend_yield_data(df, df_sans_year, year) #subtract the county trendline yield from each county's yield value
			#std = df_sans_year.std() #Column standard deviations to recalculate actual yield values later
			#df_sans_year = df_sans_year / std #divide each county's de-trended yield by the county-level standard deviation.
			#original_df_std = original_df_detrended / std
			cov = df_sans_year.cov() #covariance matrix for yields from all years except the target year we are running
			non_nan_columns = df.loc[year].dropna().index.tolist() # Check which columns do not have a NaN value for the given 'year'
			trimmed_cov_matrix = cov.loc[non_nan_columns, non_nan_columns] # Select the relevant rows and columns from the covariance matrix

			#Create temporary array which is the shape of the results in the target year. We will intersperse this with nan values later as appropriate to fit it into the overall results matrix.
			if method != random: tmp_results = np.zeros((1, len(non_nan_columns), samples_per_run, 1)) #Temporary results array of shape (1, n_counties_in_target_year, n_samples_taken, 1)
			else: tmp_results = np.zeros((1, len(non_nan_columns), samples_per_run, n_runs)) #Temporary results array of shape (1, n_counties_in_target_year, n_samples_taken, n_runs)
			
			#3.1 Pull samples and calculate estimates
			run = 0
			samples = np.zeros((samples_per_run, n_runs))

			while run < n_runs:
				sample_count = 0
				counties_sampled = [] #trimmed_cov_matrix.columns.tolist() #This list tracks which counties have been sampled
				while sample_count < samples_per_run:
					if method == 'random': counties_sampled += [random.choice([item for item in trimmed_cov_matrix.columns if item not in counties_sampled])]
					elif method == 'preset': counties_sampled += [trend_df.columns[preset_sample_order[sample_count]]] #use trend_df.columns because indices include all counties, not just the ones in the target year
					else:
						counties_sampled += [select_sample(method, trimmed_cov_matrix, counties_sampled)] #Chooses which county to sample next
					sample = df[counties_sampled[-1]][year]*np.random.normal(loc=1, scale=sample_noise)
					samples[sample_count, run] = sample
					#best_est_all_counties = update_conditional(counties_sampled, samples[:sample_count+1, run], np.zeros(len(trimmed_cov_matrix.columns)), trimmed_cov_matrix, year) #We set mu to all zeros, as the mean detrended yield is 0
					best_est_all_counties = inference_simple_mean(counties_sampled, samples[:sample_count+1, run], np.zeros(len(trimmed_cov_matrix.columns)), trimmed_cov_matrix, year) #We set mu to all zeros, as the mean detrended yield is 0
					tmp_results[0, :, sample_count, run] = best_est_all_counties
					sample_count += 1
				run += 1

			#3.2 Fill in blank counties and samples with NaNs to allow storing results for the target year in the overall results matrix
			nan_column_indices = [i for i in range(len(df.columns)) if df.columns[i] not in non_nan_columns] #indices for columns which do not have yield values in the target year
			if method != 'random': nan_insert = np.empty((1, 1, samples_per_run, n_runs))
			else: nan_insert = np.empty((1, 1, samples_per_run, 1))
			nan_insert[:] =  np.nan
			for i in range(len(nan_column_indices)): #index backwards so that the index reference doesn't shift
				tmp_results = np.hstack((tmp_results[:,:nan_column_indices[i],:,:], nan_insert, tmp_results[:,nan_column_indices[i]:,:,:]))

			#3.3 Reinflate normal standardized, detrended county yield estimates to bushels
			tgt_shape = yield_estimates[method].shape
			#Nice explanation of tiling: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
			mu_tile = np.tile(county_params[0,:].reshape(1, tgt_shape[1], 1, 1), (1, 1, tgt_shape[2], tgt_shape[3]))
			sigma_tile = np.tile(county_params[1,:].reshape(1, tgt_shape[1], 1, 1), (1, 1, tgt_shape[2], tgt_shape[3])) 
			max_value_tile = np.tile(county_params[2,:].reshape(1, tgt_shape[1], 1, 1), (1, 1, tgt_shape[2], tgt_shape[3]))
			y_pred = max_value_tile - \
					np.exp(np.multiply(sigma_tile, tmp_results) + \
					mu_tile) #converts standard normal true yield values to original raw bushels minus trend

			#tmp_df = pd.DataFrame(y_pred[method])
			#tmp_df.to_csv('estimates.csv')
			
			#selected_row = trend_df[df.index == year].to_numpy()
			#reshaped_row = selected_row.reshape(1, -1, 1, 1) # Shape: (n_counties)
			#tiled_trend = np.tile(reshaped_row, (1, 1, samples_per_run, n_runs))  # Tiling 1 time along axis 0, and samples_per_run times along axis 2, and n_runs times along axis 3
			#tiled_std = np.tile(std.to_numpy().reshape(1, -1, 1, 1), (1, 1, samples_per_run, n_runs))
			#tmp_results = np.multiply(tmp_results, tiled_std) + tiled_trend #shape (n_years, n_counties, n_samples_taken, n_runs)

			#3.4 Insert results into the target array
			#y_pred = y_pred.reshape((1, y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
			tmp_df = pd.DataFrame(y_pred.reshape(y_pred.shape[1], y_pred.shape[2]).T)
			tmp_df.to_csv('yearly_model_estimates/%d_estimates.csv' % year)
			yield_estimates[method][year - years_to_run[0], :, :, :] = y_pred

	#print('yield_estimates', yield_estimates)
	#4 Calculate, chart & save results
	#y_pred is a dictionary of arrays, each with shape (n_years, n_counties, n_samples_taken, n_runs). Each of these arrays contains estimates of each county's yield after each sample was taken.
	#results_dict = calc_losses(df[df.index.isin(years_to_run)], yield_estimates, county_params)
	y_true = np.tile(county_params[2,:], (len(years_to_run), 1)) - \
			np.exp(np.multiply(np.tile(county_params[1,:], (len(years_to_run), 1)), np.array(df[df.index.isin(years_to_run)])) + \
			np.tile(county_params[0,:], (len(years_to_run), 1))) #converts standard normal true yield values to original raw bushels minus trend
	results_dict = calc_losses(y_true, yield_estimates)
	chart(results_dict, state)

	return


def read_and_sort_yield_data(state):
	df = pd.read_csv('data/us_county_corn_production_yield.csv')
	df = df[df.County != 'OTHER (COMBINED) COUNTIES']
	df = df[df.County != 'OTHER COUNTIES']
	filtered_df = df[df['State'] == state]
	sorted_df = filtered_df.sort_values(by='Year')
	counties = sorted_df["County"].unique()
	county_yields_sorted = sorted_df.pivot_table(index='Year', columns='County', values='PYLD') #county yields sorted by year (rows) and county (columns)
	state_yield = county_yields_sorted.mean(axis=1) #This is not the true state yield, but rather the equal-weighted average of the yields from all counties which reported in a given year.

	return county_yields_sorted, state_yield


def detrend_yield_data(df):
    """
    Detrend the data in each column of the DataFrame, accounting for missing values.

    Args:
    df (pandas.DataFrame): DataFrame with columns containing time series data and possibly missing values.

    Returns:
    tuple: A tuple containing the detrended DataFrame and the DataFrame of trendline values.
    """
    detrended_df = df.copy()
    trendline_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        # Extract the column and drop NA for trend calculation
        y = df[col].dropna()
        x = y.index

        # Fit the polynomial of degree 1 (linear trend) to the data
        coeffs = np.polyfit(x, y, 1)
        trend_poly = np.poly1d(coeffs)

        # Calculate the trendline values across the full range of x
        trendline_df[col] = trend_poly(df.index)

        # Subtract the trend from the original data (NA values are unaffected)
        detrended_df[col] = df[col] - trendline_df[col]

    return detrended_df, trendline_df


def detrend_yield_data_sans_year(df, df_sans_year, year):
    """
    Detrend the data in each column of the DataFrame, accounting for missing values.

    Args:
    df (pandas.DataFrame): DataFrame with columns containing time series data and possibly missing values.

    Returns:
    tuple: A tuple containing the detrended DataFrame and the DataFrame of trendline values.
    """
    detrended_df = df.copy()
    trendline_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        # Extract the column and drop NA for trend calculation
        y = df_sans_year[col].dropna()
        x = y.index

        # Fit the polynomial of degree 1 (linear trend) to the data
        coeffs = np.polyfit(x, y, 1)
        trend_poly = np.poly1d(coeffs)

        # Calculate the trendline values across the full range of x
        trendline_df[col] = trend_poly(df.index)

        # Subtract the trend from the original data (NA values are unaffected)
        detrended_df[col] = df[col] - trendline_df[col]

        #create the detrended df sans year
        detrended_df_sans_year = detrended_df[detrended_df.index != year]

    return detrended_df_sans_year, detrended_df, trendline_df


def select_sample(sample_selection, Sigma, counties_sampled):
	#This function takes as input the sample selection method, the covariance matrix of all counties measurable in the current year, and a list of the counties which have already been sampled.
	min_target = 10**8 #initialization of target variable which we will greedily try to minimize by selecting the next variable to measure
	Sigma_values = Sigma.to_numpy()
	measured_indices = [Sigma.columns.get_loc(item) for item in counties_sampled if item in Sigma.columns]
	remaining_indices = [i for i in range(len(Sigma.columns)) if i not in measured_indices]
	len_unsampled = len(Sigma.columns) - len(counties_sampled) - 1 #Number of counties which will remain unsampled after the next sample is taken

	variables_tested = 0 #counter for the number of counties we've considered
	while variables_tested < len(Sigma.columns):
		if variables_tested in measured_indices:
			variables_tested += 1
			continue #skip county if it has already been measured
		remaining_indices_tmp = [value for value in remaining_indices if value != variables_tested]
		new_cov_matrix = calc_new_cov_matrix(Sigma_values, measured_indices + [variables_tested], remaining_indices_tmp)
		#print('new_cov_matrix', new_cov_matrix)
		if sample_selection == 'sse':
			target = np.trace(new_cov_matrix[:len_unsampled, :len_unsampled])
		elif sample_selection == 'overall_variance':
			target = np.sum(new_cov_matrix[:len_unsampled, :len_unsampled])
		else:
			print('You have chosen an invalid sample_selection method. Please choose a valid sample_selection value and re-run this code.')
			sys.exit(1)
		if target < min_target:
			min_target = target
			county = Sigma.columns[variables_tested]
		variables_tested += 1

	return county

def inference_simple_mean(measured_counties, measured_values, mu, Sigma, year):
	#This function assumes that all unmeasured standardized yield values are the average of all the standardized yield values measured so far.
	mean_measured = np.mean(measured_values)
	full_estimate = np.asarray([mean_measured] * Sigma.shape[0]) #creates a vector of the mean measurement repeated
	measured_indices = [Sigma.columns.get_loc(item) for item in measured_counties if item in Sigma.columns]
	full_estimate[measured_indices] = measured_values

	return full_estimate

def update_conditional(measured_counties, measured_values, mu, Sigma, year):
    """
    Update the conditional mean and covariance for a multivariate normal distribution,
    and return a vector of the full sample space.

    :param measured_indices: Indices of the measured variables.
    :param measured_values: Corresponding values of the measured variables.
    :param mu: Original mean vector of the multivariate normal distribution.
    :param Sigma: Original covariance matrix of the multivariate normal distribution.
    :return: Full sample space vector, updated conditional mean, and covariance matrix.
    """
    #print('Sigma',Sigma)
    #remaining_counties = [i for i in Sigma.columns if i not in measured_counties]
    #print('Sigma.columns', Sigma.columns)
    measured_indices = [Sigma.columns.get_loc(item) for item in measured_counties if item in Sigma.columns]
    remaining_indices = [i for i in range(len(Sigma.columns)) if i not in measured_indices]

    # Partition the mean and covariance matrix
    mu1 = mu[remaining_indices]
    mu2 = mu[measured_indices]
    Sigma = Sigma.to_numpy()

    Sigma11 = Sigma[np.ix_(remaining_indices, remaining_indices)]
    Sigma12 = Sigma[np.ix_(remaining_indices, measured_indices)]
    Sigma22 = Sigma[np.ix_(measured_indices, measured_indices)]
    Sigma21 = Sigma[np.ix_(measured_indices, remaining_indices)]

	# Update the conditional mean and covariance
	#Alternative matrix inversion technique with np.linalg.solve(): https://stackoverflow.com/questions/31256252/why-does-numpy-linalg-solve-offer-more-precise-matrix-inversions-than-numpy-li
	#Sigma22_inv = np.linalg.inv(Sigma22)
    Sigma22_inv = np.linalg.solve(Sigma22, np.identity(Sigma22.shape[0]))
    updated_mu = mu1 + Sigma12 @ Sigma22_inv @ (measured_values - mu2)
	#updated_Sigma = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21 #updated_mu and updated_Sigma weren't being used above, so no need to pass the data. We can turn them on later if needed.
    if len(measured_counties) < 1: print('Sigma: ', Sigma)
    if len(measured_counties) < 1:
    	print('measured_counties', measured_counties)
    	print('measured_values', measured_values)
    	#print('measured_indices', measured_indices)
    	#print('mu1', mu1)
    	print('updated_mu', updated_mu)
    	#print('Sigma12 @ Sigma22_inv', Sigma12 @ Sigma22_inv)
    if year == 2021:
	    arrays_dict = {'Sigma': Sigma, 'Sigma12': Sigma12, 'Sigma22': Sigma22, 'Sigma22_inv': Sigma22_inv, 'measured_values': measured_values.T, 'updated_mu': updated_mu.T }
	    write_csv(arrays_dict)

    # Create the full sample space vector
    full_estimate = np.zeros(Sigma.shape[0])
    full_estimate[measured_indices] = measured_values
    full_estimate[remaining_indices] = updated_mu
    #print('full_estimate', full_estimate)

    return full_estimate #, updated_mu, updated_Sigma #updated_mu and updated_Sigma weren't being used above, so no need to pass the data. We can turn them on later if needed.


def write_csv(arrays_dict): #Sigma, Sigma12, Sigma22, Sigma22_inv, measured_values, updated_mu
	# Path for the CSV file
	csv_file_path = 'tmp.csv'

	with open(csv_file_path, mode='w', newline='') as file:
	    writer = csv.writer(file)

	    # Write each array with its key as the header followed by a blank row
	    for key, array in arrays_dict.items():
	        writer.writerow([key])
	        for row in np.atleast_2d(array):
	            writer.writerow(row)
	        writer.writerow([])  # Blank row

	return


def calc_new_cov_matrix(Sigma, measured_indices, remaining_indices):
	#Sigma is a numpy array of the full covariance matrix for all counties which are measurable in the target year.
	#measured_indices is a list of the indices of the counties which have already been measured, including the prospective county to measure
	#remaining_indices is a list of the indices of the counties which have not yet been measured
	Sigma11 = Sigma[np.ix_(remaining_indices, remaining_indices)]
	Sigma12 = Sigma[np.ix_(remaining_indices, measured_indices)]
	Sigma22 = Sigma[np.ix_(measured_indices, measured_indices)]
	Sigma21 = Sigma[np.ix_(measured_indices, remaining_indices)]

	# Update the conditional mean and covariance
	#Alternative matrix inversion technique with np.linalg.solve(): https://stackoverflow.com/questions/31256252/why-does-numpy-linalg-solve-offer-more-precise-matrix-inversions-than-numpy-li
	#Sigma22_inv = np.linalg.inv(Sigma22)
	Sigma22_inv = np.linalg.solve(Sigma22, np.identity(Sigma22.shape[0]))
	updated_Sigma = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21

	return updated_Sigma


def calc_losses(y_true, y_pred):
	
	'''
	#This function runs once after all samples and forecasts have been run.

	calc_losses takes the following inputs:
	y_true is an nparray of detrended, standardized yields across all years
	y_pred is a dictionary of arrays, each with shape (n_years, n_counties, n_samples_taken, n_runs). Each of these arrays contains estimates of each county's yield after each sample was taken.
	trend_df is a dataframe of the trendline yield values for each county in each year
	std is a dataframe of the standard deviation of detrended yield for each county

	calc_losses returns:
	results_dict, a dictionary of dictionaries. Each dictionary contains a numpy array of the average loss for each sampling method vs number of samples.
	'''

	#Save test data to csv for examinaton
	'''
	for i in range(3):
		tmp_df = pd.DataFrame(y_pred['random'][i, :, :, 0].T) #dataframe of yield estimates for all counties vs number of samples taken
		tmp_df.to_csv('202%d_output_file.csv' %i, index=False)
		tmp_df = pd.DataFrame(y_true)
		tmp_df.to_csv('y_true.csv', index=False)
	'''
	n_years = y_true.shape[0] #number of years simulated
	sampling_methods = y_pred.keys() #the keys of the y_pred dictionary are the different sampling methods used.
	results_dict = {i: {} for i in ['rmse_loss', 'overall_variance_loss', 'full_insurer_loss']}

	#sse loss
	y_true_reshape = y_true.reshape(y_true.shape[0], y_true.shape[1], 1, 1)
	for method in sampling_methods:
		tmp = y_pred[method] - np.tile(y_true_reshape, (1, 1, y_pred[method].shape[2], y_pred[method].shape[3])) #differences between predictions and actuals
		tmp = np.square(tmp) #Squares errors elementwise
		tmp = np.sqrt(np.nanmean(tmp, axis=(1,3))) #averages across counties and samples to generate a 2-d numpy array of square root of the average error for each number of samples in each year
		results_dict['rmse_loss'][method] = np.nanmean(tmp, axis=(0)) #averages the root mean squared error across all years to generate a 1-d array of the average RMSE for each number of samples

	#overall variance loss
	y_true_reshape = np.nanmean(y_true_reshape, axis=(1,2,3)).reshape((y_true_reshape.shape[0], 1)) #y_true averaged across counties to get state-level true yields, with equal-weighting for each county
	state_pred = {}

	for method in sampling_methods:
		state_pred[method] = np.nanmean(y_pred[method], axis=1) #Averages across all counties to get the state-level yield prediction. Shape is n_years, n_samples_taken, n_runs
		tmp = state_pred[method] - np.tile(y_true_reshape, (1, y_pred[method].shape[2])).reshape(y_true_reshape.shape[0], y_pred[method].shape[2], y_pred[method].shape[3]) #differences between predictions and actuals
		tmp = np.square(tmp) #Squares errors elementwise
		results_dict['overall_variance_loss'][method] = np.sqrt(np.nanmean(tmp, axis=(0, 2))) #averages across years, and sample runs to generate a 1-d numpy array of the average error vs. number of samples taken

	'''
	#insurer loss
	#Represetative parameters for maize in Africa
	acres_per_farm = 10
	yield_tons_per_acre = 1
	price_dollars_per_ton = 300
	cost_per_sample = 100
	insurance_threshold = .9 #fraction of trendline yield below which the farmer is insured. We set this purposefully high to account for the averaging that happens across a county.

	#y_true_reshape *= state_std #reinflate true yields to detrended yields with state-level standard deviation

	for method in sampling_methods:
		#The units in this for loop are bushels of yield - trend
		insurance_yield = np.tile(np.asarray(np.mean(trend_df, axis=1)).reshape(y_true.shape[0], 1, 1), (1, state_pred[method].shape[1], state_pred[method].shape[2]))*(insurance_threshold-1) #Detrended insurance-level yield, of shape (n_years, n_samples, n_runs)
		state_pred[method] *= state_std #inflate state-level prediction to detrended values
		y_true_tmp = np.tile(y_true_reshape, (1, insurance_yield.shape[1])).reshape(y_true_reshape.shape[0], insurance_yield.shape[1], insurance_yield.shape[2])
		overpaid_bool = np.where((state_pred[method] < y_true_tmp) \
					& (state_pred[method] < insurance_yield), 1, 0) #1 / 0 matrix for where overpayment occured
		tmp = np.multiply(overpaid_bool, np.minimum(np.subtract(insurance_yield, state_pred[method]), np.subtract(y_true_tmp, state_pred[method])))
		underpaid_bool = np.where((state_pred[method] > y_true_tmp) \
					& (y_true_tmp < insurance_yield), 1, 0) #1 / 0 matrix for where underpayment occured
		tmp += np.multiply(underpaid_bool, np.minimum(np.subtract(state_pred[method], y_true_tmp), np.subtract(insurance_yield, y_true_tmp)))
		tmp *= acres_per_farm*yield_tons_per_acre*price_dollars_per_ton*y_true.shape[1] #overpayments or underpayment times dollars of total production value times the number of counties
		tmp += np.tile(np.arange(1, state_pred[method].shape[1]+1).reshape(1, state_pred[method].shape[1], 1), (state_pred[method].shape[0], 1, state_pred[method].shape[2]))*cost_per_sample #add number of samples * cost per sample to each loss value
		results_dict['full_insurer_loss'][method] = np.nanmean(tmp, axis=(0,2)) #averages across years, and sample runs to generate a 1-d numpy array of the average error vs. number of samples taken
		'''
	return results_dict


def chart(results_dict, state):
	'''This function takes a dictionary called 'results' 
	Creates and saves line charts for each loss type in a dictionary of dictionaries.

	Parameters:
	loss_data (dict): A dictionary of dictionaries. Each sub-dictionary contains 1-D numpy arrays of losses.
	                  The keys of the main dictionary are loss types, and the keys of the sub-dictionaries are sample types.
	'''
	print('results_dict', results_dict)
	for loss_type, samples in results_dict.items():
		print('loss_type: ', loss_type, 'sample', samples)
		plt.figure()
		for sample_type, losses in samples.items():
			#if sample_type != 'random': continue
			plt.plot(range(1, len(losses) + 1), losses, label=sample_type)
			#print('sample_type, losses', sample_type, losses)

		plt.xlabel('Number of samples')
		plt.ylabel('Loss')
		plt.title(f'Loss Chart for {loss_type} in {state}')
		plt.legend()

		# Save the plot with a filename including the datetime, loss_type, and sample_type
		datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"{datetime_str}_{loss_type}_{state}.jpg"
		plt.savefig('graphs/' + filename)
		plt.close()

	return


#This is the function we minimize to find the optimal negative lognormal parameters
def neg_log_likelihood(params, data):
	# Params = [mu, sigma, max_value]
    data_mod = np.log(params[0] - data)
    sigma = np.std(data_mod)
    mu = np.mean(data_mod)
    #return -np.sum(lognorm.logpdf(params[2] - data, s=sigma, scale=np.exp(params[0])))
    return -np.sum(lognorm.logpdf(params[0] - data, s=sigma, scale=np.exp(mu)))

def fit_neg_lognormal_params(data):
	#This function fits the mu, sigma, and max_value for a negative lognormal distribution on detrended county-level yield data
	data = data[~np.isnan(data)] #remove nan values from data
	initial_params = [np.max(data) + 5.]
	result = minimize(neg_log_likelihood, initial_params, args=(data,), method='Nelder-Mead', bounds=[(0,120)])

	max_params = result.x
	#max_params[1] = np.abs(max_params[1]) #ensures that sigma is positive
	data_mod = np.log(max_params[0] - data)
	return [np.mean(data_mod), np.std(data_mod)] + list(max_params)  #[mu, sigma, max_value]


#Testing zone
state = 'IOWA'
df, state_yield = read_and_sort_yield_data(state)
simulate_sampling()

