import numpy as np
import pandas as pd
import scipy
import copy
from bisect import bisect_left


# 1. Covariance Estimation
# calculate exponential weights
def calculate_exponential_weights(lags, lamb):
  weights = []
  for i in range(1, lags + 1):
    weight = (1 - lamb) * lamb ** (i - 1)
    weights.append(weight)
  weights = np.array(weights)
  normalized_weights = weights / weights.sum()
  return normalized_weights

# calculate exponentially weighted covariance matrix
def calculate_ewcov(data, lamb):
  weights = calculate_exponential_weights(data.shape[1], lamb)
  error_matrix = data - data.mean(axis=1)
  ewcov = error_matrix @ np.diag(weights) @ error_matrix.T
  return ewcov


# 2. Non-PSD Fixes
# Rebonato and Jackel
def near_psd(a, epsilon=0.0):
  n = a.shape[1]
  invSD = None
  out = copy.deepcopy(a)
  # calculate the correlation matrix if we got a covariance
  if (np.sum(np.isclose(1.0, np.diag(out))) != n):
    invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
    out = invSD @ out @ invSD
  # SVD, update the eigen value and scale
  vals, vecs = np.linalg.eigh(out)
  vals = np.maximum(vals, epsilon)
  T = 1.0 / (np.square(vecs) @ vals)
  T = np.diagflat(np.sqrt(T))
  l = np.diag(np.sqrt(vals))
  B = T @ vecs @ l
  out = B @ B.T
  # Add back the variance
  if invSD != None:
    invSD = np.diag(1.0 / np.diag(invSD))
    out = invSD @ out @ invSD
  return out

# Higham
def higham_psd(a, max_iter=100, tol=1e-8):
  # delta_S0 = 0, Y0 = A, gamma0 = max float
  delta_s = 0.0
  y = a
  prev_gamma = np.inf
  # loop k iterations
  for i in range(max_iter):
    r = y - delta_s
    x = projection_s(r)
    delta_s = x - r
    y = projection_u(x)
    gamma = frobenius_norm(y - a)
    if abs(gamma - prev_gamma) < tol:  
      break
    prev_gamma = gamma
  return y

def frobenius_norm(matrix):
  return np.sqrt(np.square(matrix).sum())

def projection_u(matrix):
  out = copy.deepcopy(matrix)
  np.fill_diagonal(out, 1.0)
  return out

def projection_s(matrix, epsilon=0.0):
  vals, vecs = np.linalg.eigh(matrix)
  vals = np.maximum(vals, epsilon)
  return vecs @ np.diag(vals) @ vecs.T


# 3. Simulation Methods
# Calculate Cholesky root.
def chol_psd(a):
  root = np.full(a.shape, 0.0)
  n = a.shape[1]
  # loop over columns
  for j in range(n):
    s = 0.0
    # if we are not on the first column, calculate the dot product of the preceeding row values.
    if j > 0:
      s =  root[j,:j] @ root[j,:j].T
    # Diagonal Element
    temp = a[j,j] - s
    if -1e-8 <= temp <= 0:
      temp = 0.0
    root[j,j] = np.sqrt(temp)
    # Check for the 0 eigan value.  Just set the column to 0 if we have one
    if root[j,j] == 0.0:
      root[j,j:n-1] = 0.0
    else:
      # update off diagonal rows of the column
      for i in range(j+1, n):
        s = root[i,:j] @ root[j,:j].T
        root[i,j] = (a[i,j] - s) / root[j,j]
  return root

def direct_simulation(cov, n_samples=25000):
  B = chol_psd(cov)
  r = scipy.random.randn(len(B[0]), n_samples)
  return B @ r

def pca_simulation(cov, pct_explained, n_samples=25000):
  eigen_values, eigen_vectors = np.linalg.eigh(cov)
  # calculate pca cumulative evr
  sorted_index = np.argsort(eigen_values)[::-1]
  sorted_eigenvalues = eigen_values[sorted_index]
  sorted_eigenvectors = eigen_vectors[:,sorted_index]
  evr = sorted_eigenvalues / sorted_eigenvalues.sum()
  cumulative_evr = evr.cumsum()
  cumulative_evr[-1] = 1
  # find the index for each explain percentage
  idx = bisect_left(cumulative_evr, pct_explained)
  explained_vals = np.clip(sorted_eigenvalues[:idx + 1], 0, np.inf)
  explained_vecs = sorted_eigenvectors[:, :idx + 1]
  B = explained_vecs @ np.diag(np.sqrt(explained_vals))
  r = scipy.random.randn(B.shape[1], n_samples)
  return B @ r


# 4. VaR Calculation Methods
def calculate_var(data, mean=0, alpha=0.05):
  return mean - np.quantile(data, alpha)

def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
  sigma = np.std(data)
  simulation_norm = np.random.normal(mean, sigma, nsamples)
  var_norm = calculate_var(simulation_norm, mean, alpha)
  return var_norm

def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000):
  ew_cov = calculate_ewcov(np.matrix(data).T, 0.94)
  ew_variance = ew_cov[0, 0]
  sigma = np.sqrt(ew_variance)
  simulation_ew = np.random.normal(mean, sigma, nsamples)
  var_ew = calculate_var(simulation_ew, mean, alpha)
  return var_ew

def t_var(data, mean=0, alpha=0.05, nsamples=10000):
  params = scipy.stats.t.fit(data, method="MLE")
  df, loc, scale = params
  simulation_t = scipy.stats.t(df, loc, scale).rvs(nsamples)
  var_t = calculate_var(simulation_t, mean, alpha)
  return var_t

def historic_var(data, mean=0, alpha=0.05):
  return calculate_var(data, mean, alpha)

def kde_var(data, mean=0, alpha=0.05):
  def quantile_kde(x):
    return kde.integrate_box(0, x) - alpha
  kde = scipy.stats.gaussian_kde(data)
  return mean - scipy.optimize.fsolve(quantile_kde, x0=mean)[0]


# 5. ES calculation
def calculate_es(data, mean=0, alpha=0.05):
  var = calculate_var(data, mean, alpha)
  return -np.mean(data[data <= -var])


# 6. Others
def calculate_returns(prices, method="arithmetic"):
  shifted_prices = prices[:-1]
  price_change_percent = []
  for i in range(len(shifted_prices)):
    price_change_percent.append(prices[i+1] / shifted_prices[i])
  price_change_percent = np.array(price_change_percent)
  if method == "arithmetic":
    return price_change_percent - 1
  elif method == "log":
    return np.log(price_change_percent)

# rewrite the return calculation function for pandas
def pd_calculate_returns(prices, method="arithmetic"):
  price_change_percent = (prices / prices.shift(1))[1:]
  if method == "arithmetic":
    return price_change_percent - 1
  elif method == "log":
    return np.log(price_change_percent)