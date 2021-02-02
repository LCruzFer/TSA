import numpy as np 
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acovf
from statsmodels.tools import tools 
from numpy import random as rd
import pandas as pd 

'''
Our goal is to repeat the following 1000 times 
1) simulate 1000 observations of x_t as it is defined in the assignment 
2) calculate the different t-statistics 
    - normal t-stat not accounting for dynamics in e 
    - PP t-stat with long-run variance calculated with q = T^(4/5)
    - PP t-stat with long-run variance calculated with q = T^(1/3)
and then to calculate the different Power statistics P_1, P_2 and P_3 (average number of times t-stat 
is below -2.86 for each t-stat).
For the OLS regression and to get various estimates from the regression I will use the 'statsmodels' module. 
However, we first need to define some functions to calculate the t-statistics we want.
'''

#get variance of the time series, i.e. gamma(0)
def gamma_0(TS):
    return(acovf(TS)[0])

#kernel for long run variance estimation following Andrews (1991) 
def kernel(z):
    c = 6 * np.pi * z / 5
    value = 3/c**2 * (np.sin(c) / c - np.cos(c))
    return(value)

#function acovf from statsmodels module returns all autocovariances of time series in one array
#use it to build a function for the long-run variance estimator 
#long run variance estimator due to Andrews (1991)

def long_run_var(TS, T, q, k):
    factor = T/(T-k)
    var = gamma_0(TS = TS) #first element is return is the variance of the series, i.e. gamma(0)
    cov_sum = 0 #this will be the kernel-weighted sum of autocovariances
    for h in range(1, T-1):            #iterate over all lags from 1 to T-1
        z = h / (q-1)
        kern = kernel(z)     
        cov_sum = cov_sum + kern * 2 * acovf(TS, demean = True, missing = 'drop', nlag = h)[h]
    lr_var = factor * (var + cov_sum)
    return(lr_var)

#next up: the standard t-statistic, which is not corrected for error term dynamics
def t_stat(results, coefficient, H_0 = 1): #define which coefficient we are interested in and supply results of estimation and H_0 value of coefficient
    coeff_estim = results.params[coefficient]
    se_estim = results.bse[coefficient]
    return((coeff_estim - H_0)/se_estim)
#se_estim, the standard error of the estimator, is returned as an element of the OLS regression 
#using the statsmodels module later

#now the corrected t-statistics due to Philips-Perron
def PP_t_stat(TS, T, results, q, k, H_0 = 1, coefficient = 1): #need to supply q and k as an argument as we apply long_run_var() inside, need to define which coefficient we are interested in (coeff_no)
    gamma_e_0 = gamma_0(TS)
    omega_e = long_run_var(TS = TS, T = T, q = q, k = k)
    se_estim = results.bse[coefficient] 
    stat_1 = np.sqrt(gamma_e_0/omega_e) * t_stat(results = results, coefficient = coefficient, H_0 = H_0) 
    stat_2 = (omega_e - gamma_e_0)/(2*np.sqrt(omega_e)) * (T * se_estim)/gamma_e_0
    return(stat_1 - stat_2)

#now, the last function: computing the empirical size of the test compared to the DF-test 
#t_stats argument needs to be list of t_statistics from ALL iterations 
def size(t_stats, S = 1000, crit_val = -2.86): #using 1000 iterations and critical value of -2.86 
    summation = np.sum(np.asarray(t_stats) < -2.86)
    return(1/S * summation)

'''
Now that we have the functions ready we can start the data generating process.
We will create the series x_t as described in the assignment, then calculate the 3 t-statistics we 
are interested in, append them to a predefined list and then start this process from the beginning. 
We do this 1000 times. 
'''

#create empty lists we can add the t-statistic of each iteration to 
t_stat_1_list = []
t_stat_2_list = []
t_stat_3_list = []

for j in range(0, 1000):
    #*First: data generating process
    #first, we create 1002 N(0,1) shocks 
    epsilon = rd.normal(0, 1, 1002)
    #now, we create the e_t with short run dynamics
    e = np.empty(1000)
    for i in range(0, len(e)): 
        e[i] = epsilon[i+2] - 5/6 * epsilon[i+1] + 1/6 * epsilon[i]
    #now: u_t 
    u = np.empty(1000)
    a = 1
    u[0] = e[0]
    for i in range(1, len(u)): 
        u[i] = a * u[i-1] + e[i]
    #now x_t 
    beta_0 = 1
    x = np.array([i + beta_0 for i in u])
    
    #!Alternative path 
    #*Second: regress x on a constant to demean it, resiudals of this regression is u_hat 
    #x_df = pd.DataFrame({'x': x})
    #x_df = tools.add_constant(x_df, prepend = False)
    #model_u = sm.OLS(x_df['x'], x_df['const'], missing = 'drop')
    #results_u = model_u.fit()
    #u_hat = results_u.resid
    #u_df = pd.DataFrame({'u': u_hat})
    #u_df['u_hat_lag'] = u_df['u'].shift(1)
    #u_df = tools.add_constant(u_df, prepend = False)
    #model_OLS = sm.OLS(u_df['u'], u_df[['const', 'u_hat_lag']], missing = 'drop')
    #results = model_OLS.fit() 
    #residuals = results.resid
    #*Second: make the OLS estimation and get estimate, residuals, and standard errors 
    #first, create a dataframe with x_t and x_(t-1) as columns 
    x_df = pd.DataFrame({'x': x})
    x_df['x_lag'] = x_df['x'].shift(1)
    #add a constant 
    x_df = tools.add_constant(x_df, prepend = False)
    model_OLS = sm.OLS(x_df['x'], x_df[['const', 'x_lag']], missing = 'drop')
    results = model_OLS.fit()
    residuals = results.resid
    
    #*Third: calculate t-statistics 
    #1) 
    t_stat_1 = t_stat(results = results, coefficient = 1, H_0 = 1) 
    #2) 
    t_stat_2 = PP_t_stat(TS = residuals, T = 1000, results = results, q = 251, k = 1, coefficient = 1, H_0 = 1)
    #3)
    t_stat_3 = PP_t_stat(TS = residuals, T = 1000, results = results, q = 9, k = 1, coefficient = 1, H_0 = 1)

    #append t_statistics to respective lists 
    t_stat_1_list.append(t_stat_1)
    t_stat_2_list.append(t_stat_2)
    t_stat_3_list.append(t_stat_3)
    print(j)

P_1 = size(t_stat_1_list, S = 11)
P_2 = size(t_stat_2_list, S = 11)
P_3 = size(t_stat_3_list, S = 11)

print(P_1, P_2, P_3)