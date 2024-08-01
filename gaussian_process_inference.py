#This script uses a Gaussian process to fit a kernel to lat/lon yield data and to infer unmeasured yields from measured ones.

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

### EXAMPLE 1: ESTIMATING THE VALUE OF X * SIN(X)

def estimate_x_sin_x():
	# Create full x and y data
	X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
	y = np.squeeze(X * np.sin(X))

	# Generate random training data
	rng = np.random.RandomState(1)
	training_indices = rng.choice(np.arange(y.size), size=4, replace=False)
	X_train, y_train = X[training_indices], y[training_indices]

	# Fit GP Kernel
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import RBF

	kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
	gaussian_process.fit(X_train, y_train)
	print(gaussian_process.kernel_)

	# Make predictions from fitted GP
	mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

	# Plot predictions
	plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
	plt.scatter(X_train, y_train, label="Observations")
	plt.plot(X, mean_prediction, label="Mean prediction")
	plt.fill_between(
	    X.ravel(),
	    mean_prediction - 1.96 * std_prediction,
	    mean_prediction + 1.96 * std_prediction,
	    alpha=0.5,
	    label=r"95% confidence interval",
	)
	plt.legend()
	plt.xlabel("$x$")
	plt.ylabel("$f(x)$")
	_ = plt.title("Gaussian process regression on noise-free dataset")
	plt.show()

	return



### EXAMPLE 2: DUMMY YIELD ESTIMATION

def dummy_data_gp():

	# Example data
	# Coordinates of farms (latitude, longitude)
	X_train = np.array([
	    [34.05, -118.25],  # Los Angeles
	    [36.16, -115.15],  # Las Vegas
	    [37.77, -122.42],  # San Francisco
	])

	# Yields from the farms (e.g., tons per hectare)
	y_train = np.array([2.5, 3.0, 2.7])

	# Coordinates of unmeasured farms
	X_predict = np.array([
	    [34.05, -117.15],  # Some location
	    [36.77, -119.42],  # Another location
	])

	# Kernel with RBF and adjustable constant
	kernel = C(1.0, (1e-4, 1e1)) * RBF([1, 1], (1e-4, 1e2))

	# Create Gaussian Process model
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True)

	# Fit the gaussian process
	gp.fit(X_train, y_train)

	# Make Predictions
	y_pred, sigma = gp.predict(X_predict, return_std=True)
	print("Predicted yields:", y_pred)

	print("Standard deviations of predictions:", sigma)

	return


def scatter_test_set(y_pred, y_test):
	# Create the scatter plot
	plt.figure(figsize=(8, 6))  # Set the figure size
	plt.scatter(y_pred, y_test, color='blue', label='Predicted vs Actual')  # Plot predicted vs actual values

	# Add a line of perfect prediction for reference
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='Perfect Prediction Line')

	# Labeling the plot
	plt.xlabel('Predicted Yield')
	plt.ylabel('Actual Yield')
	plt.title('Scatter Plot of Predicted vs Actual Yields')
	plt.legend()

	# Display the plot
	plt.grid(True)
	plt.show()

	return


## EXAMPLE 3: ACTUAL YIELD ESTIMATION ON ONE-YEAR'S DATA

# Load yield data from CSV
df = pd.read_csv('data/kenya_yield_data.csv')

# Filter data for the year 2019
df = df[df['year'] == 2023]

# Prepare the data: Extract coordinates and yields
X = df[['latitude', 'longitude']].values
y = df['yield'].values

# Split data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Kernel with RBF and adjustable constant
kernel = C(1.0, (1e-4, 1e1)) * RBF([1, 1], (1e-4, 1e2))

# Create Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True)

# Fit the Gaussian Process
gp.fit(X_train, y_train)

# Make predictions on the testing set
y_pred, sigma = gp.predict(X_test, return_std=True)
print("Predicted yields:", y_pred)
print("Standard deviations of predictions:", sigma)

# Optionally, evaluate the model performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Comparison with MSE from just assuming the mean yield of all measured sites for all unmeasured sites
average_yield_train = np.mean(y_train)
average_predictions = np.full_like(y_test, fill_value=average_yield_train)
mse_average = mean_squared_error(y_test, average_predictions)
print("Mean Squared Error of average yield predictions:", mse_average)

# Make a scatter plot of the test set versus predicted values
scatter_test_set(y_pred, y_test)




def fit_kernel():

	return

def gausssian_inference():

	return