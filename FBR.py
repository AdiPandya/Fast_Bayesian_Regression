import numpy as np

from numba import jit, njit, types, vectorize
import matplotlib.pyplot as plt
import seaborn

@njit(nogil=True)
def ln_likelihood(param, x, y, xerr, yerr, method=0):
    """Functon to setup the log likelihoods 

    Args:
        param (float array eg. np.array((0.0, 0.0, 0.0))): initial values of the 3 parameters
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y
        method (int): 0 for Y|X, 1 for X|Y, and 2 for ODR. Defaults to Y|X or 0.

    Returns:
        float: likelihood function in log
    """
    m,c, sig = param[0], param[1], param[2]
    
    if method == 0:
        f = m*x + c
        sigma = np.sqrt((yerr**2) + np.square(m * xerr)+ (sig**2))
        
        log_likelihood = -0.5 * np.sum((((y-f) / sigma) ** 2)+ np.log(2*np.pi*(sigma**2)))
        
    elif method == 1:
        f = m*y + c
        sigma = np.sqrt((xerr**2) + np.square(m * yerr)+ (sig**2))
        log_likelihood = -0.5 * np.sum((((x-f) / sigma) ** 2)+ np.log(2*np.pi*(sigma**2)))
        
    elif method == 2:
        f = m*x + c
        sigma = np.sqrt(((yerr**2)/(1+m**2)) + (np.square(m * xerr)/(1+m**2))+ (sig**2))
        log_likelihood = -0.5 * np.sum((((y-f)/(np.sqrt(1+m**2)*sigma))**2)+ np.log(2*np.pi*(sigma**2)))
        
    return log_likelihood

@njit(nogil=True)
def ln_prior(param, method=0):
    """Function for setting up the log priors

    Args:
        param (float array): initial values of the 3 parameters
        method (int): 0 for Y|X, 1 for X|Y, and 2 for ODR. Defaults to Y|X or 0.

    Returns:
        float: returns a 0.0 or negative infinity depending on the prior range
    """
    m,c,sig = param[0], param[1], param[2]
    
    if method == 0 or 2:
        if not (-10 < m < 10):
            return -np.inf
        if not (-10 < c < 10):
            return -np.inf
        if not (0 < sig < 10):
            return -np.inf
        return 0.0
    
    if method == 1:
        if not (-10 < 1/m < 30):
            return -np.inf
        if not (-10 < -c/m < 10):
            return -np.inf
        if not (0 < sig < 10):
            return -np.inf
        return 0.0

@njit(nogil=True)
def ln_posterior(theta, x, y, xerr, yerr, method=0):
    """function to setup the log posterior

    Args:
        param (float array eg. np.array((0.0, 0.0, 0.0))): initial values of the 3 parameters
        x (array): x data
        y (array): y data
        xerr (array): error in x
        yerr (array): error in y
        method (int): 0 for Y|X, 1 for X|Y, and 2 for ODR. Defaults to Y|X or 0.

    Returns:
        float: value of posterior in log
    """
    return ln_prior(theta, method) + ln_likelihood(theta, x, y, xerr, yerr, method)

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
def metropolis_step(x, y, xerr, yerr, ln_post_0, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), method=0):
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
    ln_post = ln_posterior(q, x, y, xerr, yerr, method)
    if ln_post - ln_post_0 > np.log(np.random.rand()):
        return q, ln_post
    return theta_0, ln_post_0

@njit(nogil=True)
def diff(arr):
    return arr[1:] - arr[:-1]

@njit(nogil=True)
def any_numba(array):
    check_array = np.ones(array.shape[0]*array.shape[1])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.any(array[i,j,:]) == False:
                check_array[i*array.shape[1]+j] = 0
                
    return check_array.reshape(array.shape[0], array.shape[1])

@njit(nogil=True)
def MCMC(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, method=0, n_chains=1, acc_frac=False):
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
    if n_chains > 1:
        n_step_split = True
    if n_chains == 1:
        n_step_split = False
        
    if n_step_split == True:
        n_steps = n_steps//n_chains
    if n_step_split == False:
        n_steps = n_steps
    
    acc = np.zeros(n_chains)
    chain = np.zeros((n_steps, n_chains,len(theta_0)))
    
    for i in range(n_chains):
        t_0 = theta_0
        lp0 = ln_posterior(t_0, x, y, xerr, yerr, method)
        for j in range(n_steps):
            t_0, lp0 = metropolis_step(x, y, xerr, yerr, lp0, t_0, step_cov, method)
            chain[j,i,:] = t_0
    
    if acc_frac == False:
        return chain, acc
    if acc_frac == True:
        for i in range(n_chains):
            acc[i] = any_numba(diff(chain))[:,i].sum() / (len(chain)-1)
        return chain, acc
  
@njit(nogil=True)
def get_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000, method=0, n_chains=1, acc_frac=False, pow_law=False):
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
    chain0,_ = MCMC(x, y, xerr, yerr, theta_0, step_cov, n_steps, method, n_chains, acc_frac)
    
    burn_in = burn_in//n_chains
    chain_0 = np.zeros((chain0[burn_in:,0,].shape[0]*n_chains, chain0[burn_in:,0,].shape[1]))
    for i in range(n_chains):
        chain_0[(len(chain0)-burn_in)*i:(len(chain0)-burn_in)*(i+1),:] = chain0[burn_in:,i,:]

    if method == 0 or 2:
        slope, slope_err = chain_0[:,0].mean(), chain_0[:,0].std()
        if pow_law == True:
            Norm = 10**(chain_0[:,1])
            intercept, intercept_err = Norm.mean(), Norm.std()
        else:
            intercept, intercept_err = chain_0[:,1].mean(), chain_0[:,1].std()
        
        int_sigma, int_sigma_err = chain_0[:,2].mean(), chain_0[:,2].std()
        
    if method == 1:
        slope, slope_err = (1/chain_0[:,0]).mean(), (1/chain_0[:,0]).std()
        if pow_law == True:
            Norm = 10**(-chain_0[:,1]/chain_0[:,0])
            intercept, intercept_err = Norm.mean(), Norm.std()
        else:
            intercept, intercept_err = -(chain_0[:,1].mean()/chain_0[:,0].mean()), (-chain_0[:,1]/chain_0[:,0].mean()).std()
        int_sigma, int_sigma_err = chain_0[:,2].mean(), chain_0[:,2].std()
   
    return slope, slope_err, intercept, intercept_err, int_sigma, int_sigma_err

@njit(nogil=True)
def get_raw_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000, method=0, n_chains=1, acc_frac=False, pow_law=False):
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
    chain0,_ = MCMC(x, y, xerr, yerr, theta_0, step_cov, n_steps, method, n_chains, acc_frac)

    burn_in = burn_in//n_chains
    chain_0 = np.zeros((chain0[burn_in:,0,].shape[0]*n_chains, chain0[burn_in:,0,].shape[1]))
    for i in range(n_chains):
        chain_0[(len(chain0)-burn_in)*i:(len(chain0)-burn_in)*(i+1),:] = chain0[burn_in:,i,:]
 
    if method == 0 or 2:
        slope = chain_0[:,0]
        if pow_law == True:
            intercept = 10**(chain_0[:,1])
        else:
            intercept = chain_0[:,1]
        int_sigma = chain_0[:,2]
    
    if method == 1:
        slope = (1/chain_0[:,0])
        if pow_law == True:
            intercept = 10**(-chain_0[:,1]/chain_0[:,0])
        else:
            intercept = -(chain_0[:,1]/chain_0[:,0])
        int_sigma = chain_0[:,2]
        
    return slope,  intercept,  int_sigma

@njit(nogil=True)
def get_a_param(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000, method=0, n_chains=1, acc_frac=False, pow_law=False):
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
    chain0,_ = MCMC(x, y, xerr, yerr, theta_0, step_cov, n_steps, method, n_chains, acc_frac)

    burn_in = burn_in//n_chains
    chain_0 = np.zeros((chain0[burn_in:,0,].shape[0]*n_chains, chain0[burn_in:,0,].shape[1]))
    for i in range(n_chains):
        chain_0[(len(chain0)-burn_in)*i:(len(chain0)-burn_in)*(i+1),:] = chain0[burn_in:,i,:]
 
    if method == 0 or 2:
        slope = np.median(chain_0[:,0])
        slope_u, slope_l = np.percentile(chain_0[:,0], [84, 16])
        if pow_law == True:
            Norm = 10**(chain_0[:,1])
            A = np.median(Norm)
            A_u, A_l = np.percentile(Norm, [84, 16])
        else:
            A = np.median(chain_0[:,1])
            A_u, A_l = np.percentile(chain_0[:,1], [84, 16])
        
        int_sigma = np.median(chain_0[:,2])
        int_sigma_u, int_sigma_l = np.percentile(chain_0[:,2], [84, 16])
        
    if method == 1:
        slope = np.median(1/chain_0[:,0])
        slope_u, slope_l = np.percentile(1/chain_0[:,0], [84, 16])
        if pow_law == True:
            Norm = 10**(-chain_0[:,1]/chain_0[:,0])
            A = np.median(Norm)
            A_u, A_l = np.percentile(Norm, [84, 16])
        else:
            A = np.median(-chain_0[:,1]/chain_0[:,0])
            A_u, A_l = np.percentile(-chain_0[:,1]/chain_0[:,0], [84, 16])
        int_sigma = np.median(chain_0[:,2])
        int_sigma_u, int_sigma_l = np.percentile(chain_0[:,2], [84, 16])
   
    return slope, slope_u, slope_l, A, A_u, A_l, int_sigma, int_sigma_u, int_sigma_l

def plot_trace(x, y, xerr, yerr, theta_0=np.array((0.0, 0.0, 0.0)), step_cov=np.diag((1e-3, 1e-4, 1e-4)), n_steps=20000, burn_in=2000, method=0, n_chains=1, acc_frac=False, pow_law=False, m_chains=False):
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
    slope, intercept, int_sigma = get_raw_param(x, y, xerr, yerr, theta_0, step_cov, n_steps, burn_in, method, n_chains, acc_frac, pow_law)
    
    fig, ax = plt.subplots(3, 2, figsize=(8, 6), sharey='row',gridspec_kw={'width_ratios': [3, 1]})

    chain_0,_ = MCMC(x, y, xerr, yerr, theta_0, step_cov, n_steps, method, n_chains, acc_frac)
    burn_in = burn_in//n_chains
    
    if m_chains == True:
        chain_0 = chain_0[burn_in:,:,:]
        slope, intercept, int_sigma = np.zeros((chain_0.shape[0],n_chains)),np.zeros((chain_0.shape[0],n_chains)),np.zeros((chain_0.shape[0],n_chains))

        
    if m_chains==False:  
        chain0 = np.zeros((chain_0[burn_in:,0,:].shape[0]*n_chains,chain_0[:,0,:].shape[1]))
        for i in range(n_chains):
            chain0[(chain_0.shape[0] - burn_in)*i:(chain_0.shape[0] - burn_in)*(i+1)] = chain_0[burn_in:,i,:]
        chain_0 = chain0.reshape(chain_0[burn_in:,0,:].shape[0]*n_chains, 1, chain_0.shape[2])  
        slope, intercept, int_sigma = np.zeros((chain_0.shape[0],1)), np.zeros((chain_0.shape[0],1)), np.zeros((chain_0.shape[0],1))
    
    if method == 0 or 2:
        for i in range(chain_0.shape[1]):
            slope[:,i] = chain_0[:,i,0]
            if pow_law == True:
                intercept[:,i] = 10**(chain_0[:,i,1])
            else:
                intercept[:,i] = chain_0[:,i,1]
            int_sigma[:,i] = chain_0[:,i,2]

    if method == 1:
        for i in range(chain_0.shape[1]):
            slope[:,i] = 1/(chain_0[:,i,0])
            if pow_law == True:
                intercept[:,i] = 10**(-chain_0[:,i,1]/chain_0[:,i,0])
            else:
                intercept[:,i] = -(chain_0[:,i,1]/chain_0[:,i,0])
            int_sigma[:,i] = chain_0[:,i,2]
            
    for i in range(chain_0.shape[1]):
        ax[0,0].plot(slope[:,i])
        ax[1,0].plot(intercept[:,i])
        ax[2,0].plot(int_sigma[:,i])
        
        seaborn.kdeplot(y=slope[burn_in:,i], ax=ax[0,1], label=f'Chain {i+1}')
        seaborn.kdeplot(y=intercept[burn_in:,i], ax=ax[1,1])
        seaborn.kdeplot(y=int_sigma[burn_in:,i], ax=ax[2,1])
        
        ax[2,0].set_xlabel('number of steps')
        ax[0,0].set_ylabel('Slope')
        ax[2,0].set_ylabel('Intrinsic scatter')
        ax[0,1].legend(bbox_to_anchor=(1.9, 1.0))
        ax[0,1].yaxis.tick_right()
        ax[1,1].yaxis.tick_right()
        ax[2,1].yaxis.tick_right()
        ax[0,1].set_xticklabels([])
        ax[1,1].set_xticklabels([])
        ax[2,1].set_xticklabels([])
        # ax[0,1].yaxis.set_label_position("right")
        # ax[2,1].yaxis.set_label_position("right")
        # ax[1,1].yaxis.set_label_position("right")
        ax[0,1].set_ylabel('Slope', rotation=270, labelpad=20)
        ax[2,1].set_ylabel('Intrinsic scatter', rotation=270, labelpad=20)

        if pow_law == True:
            ax[1,0].set_ylabel('Normalization')
            ax[1,1].set_ylabel('Normalization', rotation=270, labelpad=20)
        else:
            ax[1,0].set_ylabel('Intercept')
            ax[1,1].set_ylabel('Intercept', rotation=270, labelpad=20)

    fig.subplots_adjust(wspace=.05, hspace=.0)
    return 