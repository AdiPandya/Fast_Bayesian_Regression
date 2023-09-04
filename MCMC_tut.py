import numpy as np
import pandas as pd

from numba import jit, njit, types, vectorize

@njit(nogil=True)
def ln_likelihood(param, x, y, xerr, yerr):
    """_summary_

    Args:
        param (float array): _description_
        x (array): _description_
        y (array): _description_
        xerr (array): _description_
        yerr (array): _description_

    Returns:
        _type_: _description_
    """
    
    m,b, sig = param[0], param[1], param[2]
    f = m*x + b
    sigma = np.sqrt((yerr**2) + np.square(m * xerr)+ (sig**2))
    
    return -0.5 * np.sum((((y-f) / sigma) ** 2)+ np.log(2*np.pi*(sigma**2)))

@njit(nogil=True)
def ln_prior(param):
    m,b,sig = param[0], param[1], param[2]
    if not (-10 < m < 10):
        return -np.inf
    if not (-10 < b < 10):
        return -np.inf
    if not (0 < sig < 10):
        return -np.inf
    return 0.0

@njit(nogil=True)
def ln_posterior(theta, x, y, xerr, yerr):
    return ln_prior(theta) + ln_likelihood(theta, x, y, xerr, yerr)

@njit(nogil=True)
def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)

@njit(nogil=True)
def metropolis_step(x, y, xerr, yerr, ln_post_0, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4))):
    #q = np.random.multivariate_normal(theta_0, step_cov)
    q = chol_sample(theta_0, step_cov)
    ln_post = ln_posterior(q, x, y, xerr, yerr)
    if ln_post - ln_post_0 > np.log(np.random.rand()):
        return q, ln_post
    return theta_0, ln_post_0

@njit(nogil=True)
def MH(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000):
    lp0 = ln_posterior(theta_0, x, y, xerr, yerr)
    chain = np.empty((n_steps, len(theta_0)))
    for i in range(len(chain)):
        theta_0, lp0 = metropolis_step(x, y, xerr, yerr, lp0, theta_0, step_cov)
        chain[i] = theta_0
    #acc = float(np.any(np.diff(chain, axis=0), axis=1).sum()) / (len(chain)-1)
    return chain
  
@njit(nogil=True)
def get_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000):
    chain_0= MH(x, y, xerr, yerr, theta_0, step_cov, n_steps)
    slope, slope_err = chain_0[burn_in:,0].mean(), chain_0[burn_in:,0].std()
    intercept, intercept_err = chain_0[burn_in:,1].mean(), chain_0[burn_in:,1].std()
    int_sigma, int_sigma_err = chain_0[burn_in:,2].mean(), chain_0[burn_in:,2].std()
    return slope, slope_err, intercept, intercept_err, int_sigma, int_sigma_err

@njit(nogil=True)
def get_raw_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000):
    chain_0= MH(x, y, xerr, yerr, theta_0, step_cov, n_steps)
    slope = chain_0[burn_in:,0]
    intercept = chain_0[burn_in:,1]
    int_sigma = chain_0[burn_in:,2]
    return slope,  intercept,  int_sigma