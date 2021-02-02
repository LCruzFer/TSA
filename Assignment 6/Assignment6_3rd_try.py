import numpy as np 
import pandas as pd 
import statsmodels.formula.api as sm 
from statsmodels.tsa.stattools import acovf, adfuller
from statsmodels.tools import tools
from numpy import random as rd 


def kernel(z):
    c = 6 * np.pi * z / 5
    value = 3/c**2 * (np.sin(c)/c - np.cos(c))
    return(value)

def t_test(coeff_hat, se_hat, H_0): 
    return((coeff_hat - H_0)/se_hat)

def long_run_var(variable, T, q, k):
    factor = T/(T-k)
    gamma0 = np.var(variable)
    autocov = acovf(variable, missing = 'drop')
    cov_sum = np.sum([kernel(h/(q+1)) * 2 * autocov[h] for h in range(1, T-1)])
    lr_var = factor * (gamma0 + cov_sum)
    return(lr_var)

def PP_test(t_val, se_hat, omega_e, gamma_e_0, T):
    stat1 = np.sqrt(gamma_e_0/omega_e)*t_val
    stat2 = (omega_e - gamma_e_0)/(2*np.sqrt(omega_e)) 
    stat3 = (T * se_hat)/gamma_e_0
    return(stat1 - stat2*stat3)

#First, we create the data to be able to test our functions 
#create one dataframe where all variables and their repsective lags are saved 
t_stat_1 = []
t_stat_2 = []
t_stat_3 = []
t_stat_4 = []
PP_module = []
ADF_module = []
ERS_module = []
for S in range(0, 1000):
    data_df = pd.DataFrame()
    #start with epsilon
    #we need 1002 epsilons so we have a lag for the first e
    data_df['epsilon_t'] = rd.normal(0, 1, 1002)
    data_df['epsilon_t_1'] = data_df['epsilon_t'].shift(1)
    data_df['epsilon_t_2'] = data_df['epsilon_t'].shift(2)
    #now the e 
    #epsilon[2] is the first period, epsilon[1] and [0] are the 0th and -1st period, respectively
    data_df['e_t'] = data_df['epsilon_t'] - 5/6 * data_df['epsilon_t_1'] + 1/6 * data_df['epsilon_t_2']
    #remove first 2 rows since they have missing e values and reset the index to start at 0 again
    data_df = data_df[2:].reset_index()
    data_df = data_df.drop(['index'], axis = 1)
    #now to the u 
    #first set all to zero
    data_df['u_t'] = [0] * len(data_df)
    #set the first value equal to first value of e, since there is no u_0
    data_df.loc[0, 'u_t'] = data_df.loc[0, 'e_t']
    #data_df.loc[0, 'u_t'] = rd.normal(0, 1)
    for i in range(1, len(data_df)):
        data_df.loc[i, 'u_t'] = data_df.loc[i-1, 'u_t'] + data_df.loc[i, 'e_t']

    data_df['x_t'] = 1 + data_df['u_t']
    data_df['x_t_1'] = data_df['x_t'].shift(1)

    #now do the regression of x_t on x_t_1 and a constant
    #this is the same as regressing x on a constant to get u_hat and then regressing u_hat on its lag
    model = sm.ols('x_t ~ x_t_1', data = data_df, missing = 'drop')
    model = model.fit()
    
    #get the necessary estimates for calculating the t statistics we are interested in 
    residuals = model.resid
    param = model.params[1]
    std_err = model.bse[1]
    gamma_e_0 = np.var(residuals)
    omega_e_1 = long_run_var(residuals, T = 1000, q = int(1000**(4/5)), k = 2)
    omega_e_2 = long_run_var(residuals, T = 1000, q = int(1000**(1/3)), k = 2)
    omega_e_3 = long_run_var(residuals, T = 1000, q = 4.5, k = 2)
    t_val = t_test(coeff_hat = param, se_hat = std_err, H_0 = 1)
    t_stat_1.append(t_val)
    t_stat_2.append(PP_test(t_val, std_err, omega_e_1, gamma_e_0, 1000))
    t_stat_3.append(PP_test(t_val, std_err, omega_e_2, gamma_e_0, 1000))
    t_stat_4.append(PP_test(t_val, std_err, omega_e_3, gamma_e_0, 1000))
    #PP_module.append(PhillipsPerron(data_df['x_t']).stat)
    #ADF_module.append(adfuller(data_df['x_t'])[0])
    #ERS_module.append(DFGLS(data_df['x_t']).stat)
    print(S)

def size(t_statistics, iterations):
    P = 1/iterations * np.sum(np.asarray(t_statistics) < -2.86)
    return(P)

P1 = size(t_stat_1, 1000)
P2 = size(t_stat_2, 1000)
P3 = size(t_stat_3, 1000)
P4 = size(t_stat_4, 1000)
#P4 = size(PP_module, 1000)
#P5 = size(ADF_module, 1000)
#P6 = size(ERS_module, 1000)
#print(P1, P2, P3, P4, P5, P6)
print(P1, P2, P3)
print(P1, P2, P3, P4)

T = 1000
q1 = int(1000**(4/5))
q2 = int(1000**(1/3))
h = np.arange(1, T-1)
z1 = h/(q1 + 1)
z2 = h/(q2 + 1)
kernel_list_1 = kernel(z1)
kernel_list_2 = kernel(z2)

from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show, reset_output
output_notebook()
fig = figure(background_fill_color = 'white', background_fill_alpha = 0.9, 
                plot_height = 300, plot_width = 500, 
                x_axis_label = 'Horizon', 
                y_axis_label = 'Kernel',  
                title = 'Kernel values for different q',
                title_location = 'above')
show(fig)

fig.circle(x = h, y = kernel_list_1, alpha = 2, size = 1, color = 'red', legend = 'q = strategy 2')
fig.circle(x = h, y = kernel_list_2, alpha = 2, size = 1, color = 'blue', legend = 'q = strategy 3')
fig.circle(x = h, y = acovf(residuals), alpha = 2, size = 1, color = 'gray', legend = 'Estimated autocovariance')
show(fig)