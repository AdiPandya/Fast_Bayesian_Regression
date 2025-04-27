# Fast_Bayesian_Regression

by **Aditya Pandya**

This is the Python script for performing fast Bayesian linear regression.<br /> 
It accepts input data (x and y) along with their respective errors and provides best-fit results, including the slope, intercept, intrinsic scatter, and their respective errors.  

The script contains various function to perform the task and generate plots. 

## Key Functions for Users

Here are the important functions provided in this script:

- **`MCMC()`**  
    Performs the Markov Chain Monte Carlo (MCMC) simulation to sample from the posterior distribution of the model parameters. It returns the complete chain of sampled values.

- **`get_params()`**  
    Executes the `MCMC()` function and extracts the best-fit parameters after discarding the burn-in steps. This function supports three modes of output:  
    1. Return the best-fit parameters along with their standard deviations.  
    2. Return the full parameter chain values.  
    3. Return the best-fit parameters with their upper and lower bounds.

- **`plot_chain()`**  
    Runs the `MCMC()` function and visualizes the chain output. It supports two modes:  
    1. Creating a trace plot of the parameter chains.  
    2. Creates a corner plot to visualize parameter correlations.

## Required Libraries

This script requires the following Python packages:  
- **NumPy**: For numerical computations.  
- **Numba**: For just-in-time (JIT) compilation, which translates Python functions into optimized machine code at runtime to accelerate numerical computations. 
- **Matplotlib, Seaborn, and Corner**: For data visualizations.  

To install Numba, use the following command in your terminal:  `pip install numba`

For detailed documentation on Numba, visit the [official Numba documentation](https://numba.pydata.org/numba-doc/latest/index.html).

## Tutorial

Along with the Python script, a Jupyter Notebook ([FBR_tutorial.ipynb](./FBR_tutorial.ipynb)) is provided as a tutorial for understanding the basics of Bayesian linear regression. This notebook contains simplified functions from the script to provides step-by-step guidance for performing Bayesian analysis.

### Algorithm Flowchart

Below is the flowchart of the algorithm used for Bayesian linear regression. 
![Algorithm Flowchart][def]

[def]: Images/flowchart_2.png
