from __future__ import annotations
from cmdstanpy import CmdStanModel
import numpy as np
from pathlib import Path
import datetime as dt


# Choose which model type to simulate, and the appropriate function will be run
model_type = 'hurdle' # 'normal, 'mvn', 'hurdle'

def normal():
  # 1. Choose true parameters and generate fake data
  np.random.seed(123)

  mu_true = 5.0
  sigma_true = 2.0
  N = 100

  y = np.random.normal(loc=mu_true, scale=sigma_true, size=N)
  data = {"N": N, "y": y}

  # 2. Stan model: normal with unknown mean and sd
  stan_code = """
  data {
    int<lower=0> N;
    array[N] real y;
  }
  parameters {
    real mu;
    real<lower=0> sigma;
  }
  model {
    // weakly informative priors
    mu ~ normal(0, 10);
    sigma ~ normal(0, 5);

    // likelihood
    y ~ normal(mu, sigma);
  }
  """

  # 3. Write the Stan code to a file
  stan_file = "normal_model.stan"
  with open(stan_file, "w") as f:
      f.write(stan_code)

  # 4. Compile the model
  model = CmdStanModel(stan_file=stan_file)

  # 5. Sample from the posterior
  fit = model.sample(
      data=data,
      chains=4,
      iter_warmup=1000,
      iter_sampling=1000,
      seed=42,
  )

  # 6. Extract posterior draws
  mu_samples = fit.stan_variable("mu")
  sigma_samples = fit.stan_variable("sigma")

  print("True mu:     ", mu_true)
  print("Posterior mu mean (sd):     ", mu_samples.mean(), mu_samples.std())
  print()
  print("True sigma:  ", sigma_true)
  print("Posterior sigma mean (sd):  ", sigma_samples.mean(), sigma_samples.std())

  # Optional: quick summary table
  print("\nFull summary:")
  print(fit.summary().loc[["mu", "sigma"], :])

  return


# MVN Helper Functions

def rbf_kernel(X, ell, sigma):
    diffs = X[:, None, :] - X[None, :, :]      # (N, N, D)
    sqdist = np.sum(diffs * diffs, axis=2)     # (N, N)
    return sigma**2 * np.exp(-0.5 * sqdist / ell**2)

def summarize(name, samples, true_val):
    print(f"{name:12s} true={true_val:8.3f}  post_mean={samples.mean():8.3f}  post_sd={samples.std():8.3f}")


def mvn():
  start_time = dt.datetime.now()
  # ----------------------------
  # 1) Simulate 2D GP data (RBF kernel)
  # ----------------------------
  rng = np.random.default_rng(123)   # reproducible randomness

  N = 100
  D = 2

  # Example 2D locations in a square (replace with your own lat/lon or projected coords)
  X = rng.uniform(0.0, 10.0, size=(N, D))   # shape (N, 2)

  mu_true = 1.5
  ell_true = 1.2
  sigma_true = .5
  sigma_noise_true = 0.3

  K = rbf_kernel(X, ell_true, sigma_true)
  L = np.linalg.cholesky(K + 1e-9 * np.eye(N))

  f = mu_true + L @ rng.standard_normal(N)
  #y = f + rng.normal(0.0, sigma_noise_true, size=N) # Generates MVN data, 1st of two changes to switch from mvn to lognormal
  y = np.exp(f + rng.normal(0.0, sigma_noise_true, size=N)) # Generates multivariate lognormal data

  # ----------------------------
  # 2) Stan model (portable RBF covariance, no cov_exp_quad dependency)
  # ----------------------------
  stan_code_mvn = r"""
  data {
    int<lower=1> N;
    int<lower=1> D;
    array[N] vector[D] x;
    vector[N] y;
  }
  parameters {
    real mu;
    real<lower=0> ell;
    real<lower=0> sigma;
    real<lower=0> sigma_noise;
  }
  transformed parameters {
    vector[N] m = rep_vector(mu, N);
  }
  model {
    matrix[N, N] K;

    // Weakly-informative priors
    mu ~ normal(0, 5);
    ell ~ lognormal(0, 1);
    sigma ~ normal(0, 2);
    sigma_noise ~ normal(0, 1);

    // RBF kernel + noise
    for (i in 1:N) {
      K[i, i] = square(sigma) + square(sigma_noise) + 1e-6;
      if (i < N) {
        for (j in (i + 1):N) {
          real sqdist = dot_self(x[i] - x[j]); // works for any D
          real kij = square(sigma) * exp(-0.5 * sqdist / square(ell));
          K[i, j] = kij;
          K[j, i] = kij;
        }
      }
    }

    y ~ multi_normal(m, K);
  }
  """

  stan_code_mvlognormal = r"""
  data {
    int<lower=1> N;
    int<lower=1> D;
    array[N] vector[D] x;
    array[N] real<lower=0> y;   // POSITIVE observations
  }
  parameters {
    real mu;
    real<lower=0> ell;
    real<lower=0> sigma;
    real<lower=0> sigma_noise;  // log-space noise sd
  }
  transformed parameters {
    vector[N] m = rep_vector(mu, N);
  }
  model {
    matrix[N, N] K;

    // Weakly-informative priors
    mu ~ normal(0, 5);
    ell ~ lognormal(0, 1);
    sigma ~ normal(0, 2);
    sigma_noise ~ normal(0, 1);

    // RBF kernel for latent log-field f
    for (i in 1:N) {
      K[i, i] = square(sigma) + 1e-6;   // NO sigma_noise here anymore
      if (i < N) {
        for (j in (i + 1):N) {
          real sqdist = dot_self(x[i] - x[j]);
          real kij = square(sigma) * exp(-0.5 * sqdist / square(ell));
          K[i, j] = kij;
          K[j, i] = kij;
        }
      }
    }

    // Likelihood: log(y) = f + eps, eps ~ Normal(0, sigma_noise)
    // i.e. log(y) ~ MVN(m, K + sigma_noise^2 I)
    {
      matrix[N, N] Ky = K;
      vector[N] logy;
      for (n in 1:N) {
        Ky[n, n] += square(sigma_noise);
        logy[n] = log(y[n]);
      }
      logy ~ multi_normal(m, Ky);
    }
  }
  """

  stan_file = Path("gp_rbf_2d.stan").resolve()
  stan_file.write_text(stan_code_mvlognormal) # 2nd and final change to switch between normal and lognormal

  # CmdStanPy expects x as list-of-lists (array[N] vector[D])
  data = {"N": N, "D": D, "x": X.tolist(), "y": y.tolist()}

  model = CmdStanModel(stan_file=str(stan_file))
  fit = model.sample(
      data=data,
      chains=4,
      iter_warmup=1000,
      iter_sampling=1000,
      seed=42,
  )

  # ----------------------------
  # 3) Compare inferred vs true
  # ----------------------------

  mu_post = fit.stan_variable("mu")
  ell_post = fit.stan_variable("ell")
  sigma_post = fit.stan_variable("sigma")
  sigma_noise_post = fit.stan_variable("sigma_noise")

  print("\nPosterior vs true:")
  summarize("mu", mu_post, mu_true)
  summarize("ell", ell_post, ell_true)
  summarize("sigma", sigma_post, sigma_true)
  summarize("sigma_noise", sigma_noise_post, sigma_noise_true)

  print('Runtime: ', dt.datetime.now() - start_time)

  return


# Hurdle Helper Functions

def simulate_hurdle_gp(
    X: np.ndarray,
    rng: np.random.Generator,
    mu_f: float,
    ell_f: float,
    sigma_f: float,
    sigma_noise: float,
    mu_g: float,
    ell_g: float,
    sigma_g: float,):

  """Simulate (y, z, f, g, p) from the hurdle GP model."""
  N = X.shape[0]

  Kf = rbf_kernel(X, ell_f, sigma_f)
  Lf = np.linalg.cholesky(Kf + 1e-9 * np.eye(N))
  f = mu_f + Lf @ rng.standard_normal(N)

  Kg = rbf_kernel(X, ell_g, sigma_g)
  Lg = np.linalg.cholesky(Kg + 1e-9 * np.eye(N))
  g = mu_g + Lg @ rng.standard_normal(N)

  p = 1.0 / (1.0 + np.exp(-g))  # inv_logit
  z = rng.binomial(1, p, size=N)

  y = np.zeros(N, dtype=float)
  idx = z == 1
  y[idx] = np.exp(f[idx] + rng.normal(0.0, sigma_noise, size=idx.sum()))
  return y, z, f, g, p


stan_code_hurdle = r"""
  data {
    int<lower=1> N;
    int<lower=1> D;
    array[N] vector[D] x;
    array[N] real<lower=0> y;
  }
  parameters {
    // Yield GP (log-yield field)
    real mu_f;
    real<lower=0> ell_f;
    real<lower=0> sigma_f;
    real<lower=0> sigma_noise;

    // Gate GP (logit nonzero probability)
    real mu_g;
    real<lower=0> ell_g;
    real<lower=0> sigma_g;

    // Latent fields
    vector[N] f;
    vector[N] g;
  }
  model {
    matrix[N, N] Kf;
    matrix[N, N] Kg;

    // Priors (tune to your scale)
    mu_f ~ normal(0, 5);
    ell_f ~ lognormal(0, 1);
    sigma_f ~ normal(0, 1);
    sigma_noise ~ normal(0, 1);

    mu_g ~ normal(0, 5);
    ell_g ~ lognormal(0, 1);
    sigma_g ~ normal(0, 1);

    // Build covariance matrices
    for (i in 1:N) {
      Kf[i, i] = square(sigma_f) + 1e-6;
      Kg[i, i] = square(sigma_g) + 1e-6;

      if (i < N) {
        for (j in (i + 1):N) {
          real sqd = dot_self(x[i] - x[j]);
          real kf = square(sigma_f) * exp(-0.5 * sqd / square(ell_f));
          real kg = square(sigma_g) * exp(-0.5 * sqd / square(ell_g));
          Kf[i, j] = kf; Kf[j, i] = kf;
          Kg[i, j] = kg; Kg[j, i] = kg;
        }
      }
    }

    // GP priors on latent fields
    f ~ multi_normal(rep_vector(mu_f, N), Kf);
    g ~ multi_normal(rep_vector(mu_g, N), Kg);

    // Likelihood: zeros vs positives
    for (n in 1:N) {
      if (y[n] == 0) {
        target += bernoulli_logit_lpmf(0 | g[n]);
      } else {
        target += bernoulli_logit_lpmf(1 | g[n]);
        target += normal_lpdf(log(y[n]) | f[n], sigma_noise);
      }
    }
  }
  """


def hurdle():
  start_time = dt.datetime.now()
  # ----------------------------
  # 1) Generate synthetic 2D locations + data
  # ----------------------------
  rng = np.random.default_rng(123)

  N = 50
  D = 2

  # Example "geospatial" coordinates in a 10x10 box.
  # For real lat/lon, consider projecting to meters/km first.
  X = rng.uniform(0.0, 10.0, size=(N, D))

  # True parameters (edit as you like)
  true = dict(
      mu_f=1.5, ell_f=1.2, sigma_f=0.6, sigma_noise=0.4,
      mu_g=-0.5, ell_g=2.0, sigma_g=1.0
  )

  y, z, f_true, g_true, p_true = simulate_hurdle_gp(
      X, rng,
      mu_f=true["mu_f"], ell_f=true["ell_f"], sigma_f=true["sigma_f"], sigma_noise=true["sigma_noise"],
      mu_g=true["mu_g"], ell_g=true["ell_g"], sigma_g=true["sigma_g"]
  )

  print(f"Simulated N={N} points; nonzero fraction = {z.mean():.3f}")

  # ----------------------------
  # 2) Write and compile Stan model
  # ----------------------------
  stan_file = Path("gp_hurdle_lognormal_2d.stan").resolve()
  stan_file.write_text(stan_code_hurdle)

  data = {"N": N, "D": D, "x": X.tolist(), "y": y.tolist()}

  model = CmdStanModel(stan_file=str(stan_file))

  # ----------------------------
  # 3) Sample posterior
  # ----------------------------
  fit = model.sample(
      data=data,
      chains=4,
      iter_warmup=1000,
      iter_sampling=1000,
      seed=42,
      adapt_delta=0.9,  # often helps for GP models
  )

  # ----------------------------
  # 4) Summaries
  # ----------------------------
  mu_f_post = fit.stan_variable("mu_f")
  ell_f_post = fit.stan_variable("ell_f")
  sigma_f_post = fit.stan_variable("sigma_f")
  sigma_noise_post = fit.stan_variable("sigma_noise")

  mu_g_post = fit.stan_variable("mu_g")
  ell_g_post = fit.stan_variable("ell_g")
  sigma_g_post = fit.stan_variable("sigma_g")

  print("\nPosterior vs true:")
  summarize("mu_f", mu_f_post, true["mu_f"])
  summarize("ell_f", ell_f_post, true["ell_f"])
  summarize("sigma_f", sigma_f_post, true["sigma_f"])
  summarize("sigma_noise", sigma_noise_post, true["sigma_noise"])
  summarize("mu_g", mu_g_post, true["mu_g"])
  summarize("ell_g", ell_g_post, true["ell_g"])
  summarize("sigma_g", sigma_g_post, true["sigma_g"])

  print("\nStan summary (key params):")
  print(
      fit.summary().loc[
          ["mu_f", "ell_f", "sigma_f", "sigma_noise", "mu_g", "ell_g", "sigma_g"],
          :
      ]
  )

  # Optional: recover posterior mean nonzero probabilities at each point
  g_post = fit.stan_variable("g")  # shape (draws, N)
  p_post_mean = 1.0 / (1.0 + np.exp(-g_post)).mean(axis=0)
  print(f"\nPosterior mean nonzero prob: min={p_post_mean.min():.3f}, max={p_post_mean.max():.3f}")

  print('Runtime: ', dt.datetime.now() - start_time)

  return


# This section calls the function to simulate the chosen model_type
if model_type == 'normal': normal()
elif model_type == 'mvn': mvn()
elif model_type == 'hurdle': hurdle()

