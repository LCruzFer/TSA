import numpy as np 
from numpy import random as rd 
from scipy.stats import t
import time
import pandas as pd
import matplotlib.pyplot as plt


#!MLE functions 
def get_eps_t(xt, b, T): 
    '''
    Note: this cannot handle single array (due to reshape)
    If want single array of x to be handled, remove reshaping of res in loop
    Get *all* epsilon_t following summation approach. 
    Returns an array with all eps_t and and array with all epsilon_t^2, which we need for sigma^2.
    *xt = 1D array of x
    *b = scalar parameter 
    *T = # of periods, i.e. len(xt)
    '''
    final_norm = np.empty((n_specs, 1))
    for t in range(T+1): 
        #get x_{t-s} up to x_t
        #turn around order such that x_t is first element and x_1 is last 
        sub_x = np.flip(xt[:, :t], axis = 1)
        sub_x_II = np.empty((n_specs, len(sub_x[0])))
        #loop over all different x-specifications and calculate epsilon as summation of x_t following formula
        for j in range(n_specs):
            sub_x_II[j] = np.array([b**i * x_t * (-1)**i for i, x_t in enumerate(sub_x[j])])
        res = np.sum(sub_x_II, axis = 1)
        res = np.reshape(res, (n_specs, 1))
        final_norm = np.hstack((final_norm, res))
    #drop first observation since this is just initializing, empty array
    final_norm = final_norm[:, 2:]
    return(final_norm)

def lll_hood_opt(xt, bs, T):
    values = np.empty((n_specs, 1))
    for param in bs:
        eps_hat = get_eps_t(xt, param, T = T)
        sigma = 1/(T-1) * np.sum(eps_hat**2, axis = 1)
        #can get sum of sqrd epsilon directly and save it
        res = (T-1)*np.log(1/np.sqrt(2*np.pi*sigma)) - (1/(2*sigma)) * np.sum(eps_hat**2)
        res = np.reshape(res, (n_specs, 1))
        values = np.hstack((values, res))
    #discard initializing empty array again
    values = values[:, 1:]
    max_i = np.argmax(values, 1)
    opt_param = np.empty((n_specs, 1))
    for i in range(n_specs): 
        opt_param[i] = bs[max_i[i]]
    return(opt_param)

#!replication study 
#set general parameters
S = 100
T = 1000
n_specs = 3
b = 0.25
bs = np.arange(-0.9, 0.9+0.001, 0.05)
#set up empty array where everything is saved to 
#this time using hstack, maybe faster but in the end need to discard first element 
opt_params = np.empty((n_specs, 1))
start = time.time()
for s in range(S):
    eps = np.empty((n_specs, T+1))
    eps[0] = rd.normal(0, 1, T+1)
    eps[1] = t.rvs(df = 5, size = T+1)
    eps[2] = rd.uniform(0, 1, T+1)
    eps[:, 0] = 0
    eps_lag = eps[:, :T]
    #start epsilon from second element because we added eps[0] = 0
    x = eps[:, 1:] + b*eps_lag
    results = lll_hood_opt(x, bs, T)
    opt_params = np.hstack((opt_params, results))
    print(s)
end = time.time() 
print(end-start)

#get sigmas for optimal parameters by calculating them again for the given b 

#!calculating biases 
def bias(estimate, true_val, T): 
    '''
    Calculate bias as given in assignment.
    '''
    reps = len(estimate)
    summation = np.sum(estimate)
    bias = 1/reps * summation - true_val
    return(bias)

bias_params = np.empty(n_specs)
bias_variance = np.empty(n_specs)
true_variances = np.array((1, 5/3, 1/12)) #define true variances (per hand since easier)
bias(opt_params)
for i, param in enumerate(opt_params):
    bias_params[i] = bias(param, b, T)
print(bias_params)
