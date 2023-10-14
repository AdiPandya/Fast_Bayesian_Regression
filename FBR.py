import numpy as np

from numba import jit, njit, types, vectorize

@njit(nogil=True)
def ln_likelihood(param, x, y, xerr, yerr):
    """Functon to setup the log likelihoods 

    Args:
        param (float array eg. np.array((0.0, 0.0, 0.0))): initial values of the 3 parameters
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y

    Returns:
        float: likelihood function in log
    """
    m,b, sig = param[0], param[1], param[2]
    f = m*x + b
    sigma = np.sqrt((yerr**2) + np.square(m * xerr)+ (sig**2))
    
    return -0.5 * np.sum((((y-f) / sigma) ** 2)+ np.log(2*np.pi*(sigma**2)))

@njit(nogil=True)
def ln_prior(param):
    """Function for setting up the log priors

    Args:
        param (float array): initial values of the 3 parameters

    Returns:
        float: returns a 0.0 or negative infinity depending on the prior range
    """
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
    """function to setup the log posterior

    Args:
        param (float array eg. np.array((0.0, 0.0, 0.0))): initial values of the 3 parameters
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y

    Returns:
        float: value of posterior in log
    """
    return ln_prior(theta) + ln_likelihood(theta, x, y, xerr, yerr)

@njit(nogil=True)
def multivariate_sample(mean, cov):
    """Function to get a sample from a multivariate normal distribution

    Args:
        mean (float array eg. np.array((0.0, 0.0, 0.0))): mean for the multivariate normal distribution
        cov (float diagonal matrix eg. np.diag((1e-3, 1e-4, 1e-4))): covariance matrix for the multivariate normal distribution

    Returns:
        float array: random sample from the multivariate normal distribution
    """
    return mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)

@njit(nogil=True)
def metropolis_step(x, y, xerr, yerr, ln_post_0, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4))):
    """Function that takes a step in the mcmc chain using the metropolis hastings algorithm

    Args:
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y
        ln_post_0 (float): initial value of the log posterior
        theta_0 (float, array): initial values of the parameters (used as mean for the random multivariate draw). Defaults to np.array((0.0, 0.0, 0.0)).
        step_cov (float, array): diagonal matrix used to control the step size for mcmc. Defaults to np.diag((1e-3, 1e-4, 1e-4)).

    Returns:
        (array, float): (parameters values, its log posterior)
    """
    #q = np.random.multivariate_normal(theta_0, step_cov)
    q = multivariate_sample(theta_0, step_cov)
    ln_post = ln_posterior(q, x, y, xerr, yerr)
    if ln_post - ln_post_0 > np.log(np.random.rand()):
        return q, ln_post
    return theta_0, ln_post_0

@njit(nogil=True)
def MCMC(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000):
    """Function that runs the mcmc chain

    Args:
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y
        theta_0 (float, array): initial values of the parameters (used as mean for the random multivariate draw). Defaults to np.array((0.0, 0.0, 0.0)).
        step_cov (float, array): diagonal matrix used to control the step size for mcmc. Defaults to np.diag((1e-3, 1e-4, 1e-4)).
        n_steps (int): Number of mcmc samples. Defaults to 20000.

    Returns:
        3 dimensional numpy array: chain of mcmc samples
    """
    lp0 = ln_posterior(theta_0, x, y, xerr, yerr)
    chain = np.empty((n_steps, len(theta_0)))
    for i in range(len(chain)):
        theta_0, lp0 = metropolis_step(x, y, xerr, yerr, lp0, theta_0, step_cov)
        chain[i] = theta_0
    #acc = float(np.any(np.diff(chain, axis=0), axis=1).sum()) / (len(chain)-1)
    return chain
  
@njit(nogil=True)
def get_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000):
    """Function to get the mean and standard deviation of the parameters from the mcmc chain

    Args:
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y
        theta_0 (float, array): initial values of the parameters (used as mean for the random multivariate draw). Defaults to np.array((0.0, 0.0, 0.0)).
        step_cov (float, array): diagonal matrix used to control the step size for mcmc. Defaults to np.diag((1e-3, 1e-4, 1e-4)).
        n_steps (int): Number of mcmc samples. Defaults to 20000.
        burn_in (int): Number of burn in samples. Defaults to 2000.

    Returns:
        float: (slope, slope std, intercept, intercept std, intrinsic scatter, intrinsic scatter std)
    """
    chain_0= MCMC(x, y, xerr, yerr, theta_0, step_cov, n_steps)
    slope, slope_err = chain_0[burn_in:,0].mean(), chain_0[burn_in:,0].std()
    intercept, intercept_err = chain_0[burn_in:,1].mean(), chain_0[burn_in:,1].std()
    int_sigma, int_sigma_err = chain_0[burn_in:,2].mean(), chain_0[burn_in:,2].std()
    return slope, slope_err, intercept, intercept_err, int_sigma, int_sigma_err

@njit(nogil=True)
def get_raw_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000):
    """Function to get the raw values of the parameters from the mcmc chain

    Args:
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y
        theta_0 (float, array): initial values of the parameters (used as mean for the random multivariate draw). Defaults to np.array((0.0, 0.0, 0.0)).
        step_cov (float, array): diagonal matrix used to control the step size for mcmc. Defaults to np.diag((1e-3, 1e-4, 1e-4)).
        n_steps (int): Number of mcmc samples. Defaults to 20000.
        burn_in (int): Number of burn in samples. Defaults to 2000.

    Returns:
        float array: slope, intercept, intrinsic scatter values with size (n_steps-burn_in)
    """
    chain_0= MCMC(x, y, xerr, yerr, theta_0, step_cov, n_steps)
    slope = chain_0[burn_in:,0]
    intercept = chain_0[burn_in:,1]
    int_sigma = chain_0[burn_in:,2]
    return slope,  intercept,  int_sigma