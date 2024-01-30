#This script creates synthetic yield data for testing of the sampling functions

import numpy as np
import pandas as pd

#0 Parameter setting
n_years = 10000
n_counties = 99

#1 Create random covariance matrix (symmetric, positive semi-definite)
cov = np.random.random((n_counties, n_counties))

for i in range(n_counties):
	cov[i,i] = 1.0
	for j in  range(1, n_counties - i):
		cov[i, j] = cov[j, i]

#1.5 Read in covariance matrix
external_cov = pd.read_csv('covariance.csv')
external_cov = external_cov.to_numpy()

#1.6 Create Random Covariance Matrix that is positive semi-definite
A = np.random.rand(n_counties, n_counties)

#Make sure matrix is symmetric
#A = .5 * (A + A.T)

#Ensure the matrix is positive semi-definite
cov = np.dot(A, A.transpose())

#Normalize the matrix
norms = np.sqrt(np.diag(cov))
cov = cov / np.outer(norms, norms)

#2 Create n_years random draws from MVN with the random covariance matrix
x = np.random.multivariate_normal(np.zeros((n_counties)), cov, size=n_years)

#3 Save results to .csv files
x_df = pd.DataFrame(x)
x_df.to_csv('synthetic_yield_data.csv')

cov_df = pd.DataFrame(cov)
cov_df.to_csv('synthetic_cov.csv')

sample_cov = x_df.cov()
sample_cov.to_csv('sample_cov.csv')