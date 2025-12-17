from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import warnings

# --------- CONFIG: set these to match your CSV ----------
CSV_PATH = Path("data/kenya_yield_data.csv")  # <-- change
LAT_COL = "latitude"
LON_COL = "longitude"
YIELD_COL = "yield"

# Optional filters
YEAR_COL = "year"       # set to None if not present
TARGET_YEAR = 2023     # e.g. 2023, or None to use all rows
N_SAMPLES_USED = 100		# set the number of samples to be chosen at random for fitting

# If your yields have tiny negatives from preprocessing, clip them to zero
CLIP_NEGATIVE_YIELDS_TO_ZERO = True

# Coordinate transform:
# If you have projected meters already, set USE_EARTH_KM=False and set X,Y columns accordingly.
USE_EARTH_KM = True  # True for lat/lon → approximate km


STAN_CODE = r"""
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

  // Priors (tune to your spatial scale!)
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


def latlon_to_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Roughly convert lat/lon degrees to a local km coordinate system.
    Good enough for small-ish regions; for larger regions use a real projection.
    """
    lat0 = float(np.mean(lat))
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.deg2rad(lat0))
    x = (lon - np.mean(lon)) * km_per_deg_lon
    y = (lat - np.mean(lat)) * km_per_deg_lat
    return np.column_stack([x, y])


def load_real_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(CSV_PATH)

    # Optional year filter
    if YEAR_COL is not None and TARGET_YEAR is not None and YEAR_COL in df.columns:
        df = df.loc[df[YEAR_COL] == TARGET_YEAR].copy()

    # Keep only needed columns and drop missing
    needed = [LAT_COL, LON_COL, YIELD_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.dropna(subset=needed).copy()
    df = df.sample(n=N_SAMPLES_USED, random_state=42) # df.sample(n=N_SAMPLES_USED, random_state=42)

    # Yields must be numeric and >= 0
    df[YIELD_COL] = pd.to_numeric(df[YIELD_COL], errors="coerce")
    df = df.dropna(subset=[YIELD_COL]).copy()

    if CLIP_NEGATIVE_YIELDS_TO_ZERO:
        df.loc[df[YIELD_COL] < 0, YIELD_COL] = 0.0
    else:
        if (df[YIELD_COL] < 0).any():
            raise ValueError("Found negative yields; set CLIP_NEGATIVE_YIELDS_TO_ZERO=True or clean data.")

    y = df[YIELD_COL].to_numpy(dtype=float)

    lat = df[LAT_COL].to_numpy(dtype=float)
    lon = df[LON_COL].to_numpy(dtype=float)

    if USE_EARTH_KM:
        X = latlon_to_km(lat, lon)
    else:
        # If you already have projected coordinates, replace this section accordingly
        X = np.column_stack([lon, lat])  # placeholder: treat lon/lat as planar

    return X, y, df


def print_key_summary(summ, keys):
    """
    Print a stable subset of CmdStanPy summary columns with graceful fallbacks
    across CmdStanPy/CmdStan versions.
    """
    # Only keep keys that exist
    keys_present = [k for k in keys if k in summ.index]
    missing_keys = [k for k in keys if k not in summ.index]
    if missing_keys:
        warnings.warn(f"Missing parameters in summary index (skipping): {missing_keys}")

    if not keys_present:
        print("No requested parameters found in summary index.")
        print("Available rows include e.g.:", list(summ.index)[:20])
        return

    # Preferred columns (some may not exist depending on version)
    candidate_cols = [
        "Mean", "StdDev",
        "2.5%", "50%", "97.5%",
        "5%", "95%",           # fallback quantiles some versions use
        "R_hat",
        "ESS_bulk", "ESS_tail", # may vary by version
        "N_Eff"                 # very old naming
    ]

    cols_present = [c for c in candidate_cols if c in summ.columns]
    if not cols_present:
        print("None of the expected summary columns were found.")
        print("Available columns:", list(summ.columns))
        return

    # If both 2.5/97.5 and 5/95 exist, keep the 2.5/97.5 and drop 5/95 to avoid clutter
    if "2.5%" in cols_present and "97.5%" in cols_present:
        cols_present = [c for c in cols_present if c not in ("5%", "95%")]

    print(summ.loc[keys_present, cols_present])


def main():
    # 1) Load real data
    X, y, df = load_real_data()
    N, D = X.shape
    print(f"Loaded N={N} points. Nonzero fraction={(y > 0).mean():.3f}")

    # 2) Write Stan model file
    stan_file = Path("gp_hurdle_lognormal_2d.stan").resolve()
    stan_file.write_text(STAN_CODE)

    # 3) Build CmdStan data dict
    data = {"N": int(N), "D": int(D), "x": X.tolist(), "y": y.tolist()}

    # 4) Compile + sample
    model = CmdStanModel(stan_file=str(stan_file))
    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=42,
        adapt_delta=0.9,
    )

    # 5) Save summaries
    summ = fit.summary()
    out_csv = Path("hurdle_gp_posterior_summary.csv")
    summ.to_csv(out_csv)
    print(f"Saved posterior summary: {out_csv}")

    # Print a small key-parameter view
    keys = ["mu_f", "ell_f", "sigma_f", "sigma_noise", "mu_g", "ell_g", "sigma_g"]
    print("\nKey parameter summary:")
    print_key_summary(summ, keys) # Prints confidence intervals, R_hat, and other parameter statistics


if __name__ == "__main__":
    main()