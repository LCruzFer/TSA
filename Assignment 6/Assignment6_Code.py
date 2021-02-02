import numpy as np 
import pandas as pd 
from statsmodels.tsa.stattools import acovf 
import statsmodels.api as sm 
from numpy import random as rd 

#generating some data 
epsilon = rd.normal(0, 1, 1002)

e = np.empty(1000) 
e = [epsilon[i+2] - 5/6 * epsilon[i+1] + 1/6 * epsilon[i] for i in range(len(e))]

u = np.empty(1000)
u[0] = e[0]
u = [u[i-1] + e[i] for i in range(1, len(u))]

x = np.empty(1000)
x = [x + 1 for x in u]

x_lag = np.empty(1000)
x_lag = [x[i-1] for i in range(len(x))]
x_lag[0] = 'nan'

df = pd.DataFrame({'x': x, 'const': [1]*len(x)})

model = sm.OLS(df['x'], df['const'], missing = 'drop')
results = model.fit()
params = results.params
df['u_hat'] = results.resid
df['u_hat_lag'] = df['u_hat'].shift(1)

model2 = sm.OLS(df['u_hat'], df['u_hat_lag'], missing = 'drop')
results2 = model2.fit()
params2 = results2.params
