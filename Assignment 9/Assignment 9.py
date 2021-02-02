import numpy as np 
from numpy import random as rd 
from scipy.stats import t
import matplotlib.pyplot as plt
import time 

#!#################
#*Whittle estimator
#!#################


def transfer(b, lambda_j): 
    '''
    Transfer function of MA(1) process as given in class.
    '''
    return((1 + b**2 + 2*b*np.cos(lambda_j)))

def Ix(x_t, lambda_j): 
    '''
    I_x as defined on slide 144.
    '''
    T = len(x_t)
    factor = 1/(2*np.pi*T)
    cos_term = np.sum([x_t[t] * np.cos(lambda_j*t) for t in range(T)])
    sin_term = np.sum([x_t[t] * np.sin(lambda_j*t) for t in range(T)])
    return(factor * (cos_term**2 + sin_term**2))

def omega(M, b, x_t, lambdajs):
    '''
    singled out sigma divided by 2pi
    '''
    Ixs = np.array([Ix(x_t = x_t, lambda_j = lam) for lam in lambdajs])
    deltas = np.array([transfer(b = b, lambda_j = lam) for lam in lambdajs])
    sum_fractions = np.sum(Ixs/deltas)
    return(1/M*sum_fractions)

def I_m(x_t, b, lambdajs, sigma):
    '''
    Objective function as discussed in class, log part split up 
    '''
    first = transfer(b, lambdajs)
    second = sigma
    third = np.array([Ix(x_t = x_t, lambda_j = lam) for lam in lambdajs])
    val = - np.log(first) - np.log(second) - third/(second*first)
    summation = np.sum(val)
    return(summation)

def optimization(x_t, bs, lambdajs, M):
    '''
    Calculate all I_m for a grid of points b and find associated value in bs for max(I_m)
    '''
    sigmas = np.array([omega(M, param, x_t, lambdajs) for param in bs])
    Ims = np.array([I_m(x_t = x_t, b = param, lambdajs = lambdajs, sigma = sig) for param, sig in zip(bs, sigmas)])
    max_i = np.where(Ims == max(Ims))
    param_hat = bs[max_i]
    variance_hat = sigmas[max_i] * 2 * np.pi
    return(param_hat, variance_hat)

#!bias functions
def bias(estimate, true_val, T): 
    '''
    Calculate bias as given in assignment.
    '''
    reps = len(estimate)
    summation = np.sum(estimate)
    bias = 1/reps * summation - true_val
    return(bias)

def MSE(bias, parameter, T):
    '''
    *parameter = parameter estimates
    '''
    bias2 = bias**2 
    variance = np.var(parameter)
    return((bias2 + variance)*T)


#!Simulation study 
T = 100
M = int((T-1)/2)
b = 0.25
lambdas = np.array([(2*np.pi*j)/T for j in range(M)])
bs = np.arange(-0.9, 0.9, 0.05)
n_specs = 3
S = 100
#empty matrices for DGP
eps = np.empty((n_specs, T+1))
eps_lag = np.empty((n_specs, T+1))
#preparing empty arrays for estimates
params_hat = np.empty((n_specs, S))
variances_hat = np.empty((n_specs, S))
start = time.time()

for s in range(S):
    eps[0] = rd.normal(0, 1, T+1)
    eps[1] = t.rvs(df = 5, size = T+1)
    eps[2] = rd.uniform(0, 1, T+1)
    eps_lag = eps[:, :T]        #each array in array of arrays eps is taken up to T, so last obs is dropped
    eps_lag = eps[:, :T]
    x = eps[:, 1:] + b * eps_lag  #take only elements i = 1 to i = T-1 of epsilon arrays
    for i, k in enumerate(x):
        results = np.array(optimization(k, bs, lambdas, M))
        params_hat[i][s] = results[0]
        variances_hat[i][s] = results[1]
    print(s)
end = time.time() 
print(end-start)
print(params_hat)
print(variances_hat)

#!calculating biases 
bias_params = np.empty(n_specs)
bias_variance = np.empty(n_specs)
true_variances = np.array((1, 5/3, 1/12)) #define true variances (per hand since easier)

for i, param in enumerate(params_hat):
    bias_params[i] = bias(param, b, T)
print(bias_params)

for i, param, true in zip(range(n_specs), variances_hat, true_variances):
    bias_variance[i] = bias(param, true, T)
print(bias_variance)

#!calculating MSEs 
param_hat_MSE = np.empty(n_specs)
for i, bias, param in zip(range(n_specs), bias_params, params_hat):
    param_hat_MSE[i] = MSE(bias, param, T)
print(param_hat_MSE)

variance_hat_MSE = np.empty(n_specs)
for i, bias, param in zip(range(n_specs), bias_variance, variances_hat):
    variance_hat_MSE[i] = MSE(bias, param, T)
print(variance_hat_MSE)