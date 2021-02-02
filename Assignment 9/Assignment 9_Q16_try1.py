import numpy as np 
from scipy.integrate import quad
import pandas as pd
from numpy import random as rd
from statsmodels.tsa.stattools import acovf
import matplotlib.pyplot as plt

def f(x, z): 
    val = x**(z-1)*np.exp(-x)
    return(val)

def intf(z):
    if z!= 0: 
        val = quad(f, 0, np.inf, args = z)[0]
    else:
        val = quad(f, 0.001, np.inf, args = z)[0]
    return(val)

def psi(j, d):
    if d<0:
        upper = j**(-d-1)
        lower = intf(abs(d))
    else:
        upper = j**(d-1)
        lower = intf(d)
    value = upper/lower
    return(value)

def get_data(length, d_list, epsilon):
    y = np.empty((len(d_list), length))
    for i in range(len(d_list)):
        result = np.empty(length)
        for t in range(length):
            result[t] = np.sum([psi(j, d_list[i]) * epsilon[t-j] for j in range(1, t)])
            print(t)
        y[i, :] = result
    return(y)

def get_gamma0(data, d_list, d):
    '''
    data must be an array
    '''
    pos = np.where(d_list == d)
    sigma2 = np.var(data[pos, :])
    upper = intf(1 - 2*d)
    lower = intf(1-d)**2
    final = sigma2*(upper/lower)
    return(final)

def get_corrs(data, laglength, d_list):
    autocorrs = np.empty((len(d_list), laglength + 1))
    estimated = np.empty((len(d_list), laglength + 1))
    for i in range(len(d_list)):
        lags = np.empty((len(d_list), laglength+1))
        gamma_0 = get_gamma0(data, d_list, d_list[i])
        lags[i][0] = gamma_0
        for h in range(1, laglength + 1):
            factor = (h-1+d_list[i])/(h-d_list[i])
            gammah = factor * lags[i][h-1]
            lags[i][h] = gammah
        #getting autocorrs from autocovs
        autocorrs[i, :] = lags[i, :]/lags[i, 0]
        estimated[i, :] = acovf(data[i, :], demean = False, nlag = laglength)
    final = np.array((autocorrs, estimated))
    return(final)

M = 2000
TS = 1000
overall_length = M + TS
eps = rd.normal(0, 1, M+TS)
d_list = np.array((-0.45, -0.25, 0, 0.25, 0.45))

y = get_data(overall_length, d_list, eps)
y_cutoff = y[:, M:]
df = pd.DataFrame()
for i in range(len(y)):
    df[str(i)] = y[i]
df.to_csv('ys.csv')
#first element of a is arrays of true autocorrelations/approximated autocorrelations 
#second element are estimated autocorrelations 
#element i of first or second respectively are the autocorrs for d == d_list[i]
lag_len = 25
a = get_corrs(y, lag_len, d_list)

#now I only need to plot them 
#! Plotting 
for i in range(len(d_list)): 
    fig = plt.figure()
    plt.grid()
    label1 = r'$\rho(h)$'  + ', d = ' + str(d_list[i])
    plt.scatter(np.arange(0, lag_len+1), a[0][i], label = label1)
    label2 = r'$\hat{\rho}(h)$'  + ', d = ' + str(d_list[i])
    plt.scatter(np.arange(0, lag_len+1), a[1][i], label = label2)
    plt.legend()
    plt.title('d = ' + str(d_list[i]))
    plotname = 'd_' + str(d_list[i]) + '.pdf'
    fig.savefig(plotname, bbox_inches = 'tight')
    plt.show()

fig2 = plt.figure()
plt.grid() 
for i in range(len(y_t))
    label = r'$\{y_t\}$' + ', d = ' + str(d_list[i])
    plt.plot(np.arange(0, 1001), y[i], label = label) 
fig2.savefig('ytplot, bbox_inches = 'tight')