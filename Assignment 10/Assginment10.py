import numpy as np 
import time 
from numpy import random as rd
import statsmodels 
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import pandas as pd
#!Data generation functions
def get_psis(j_max, d):
    '''
    Get an array of the psi_j/pi_j, which are MA coefficients of fractional process
    '''
    psi = np.empty(j_max)
    psi[0] = 1
    for j in range(1, j_max):
        psi[j] = (j-1+d)/j * psi[j-1] 
    return(psi)

def data_gen(total_len, d):
    '''
    This function creates T observation array of y_t as defined in the Assignment.
    '''
    #first, get x_t
    eps = rd.normal(0,1,total_len+1)
    x = np.empty(total_len)
    x[0] = eps[1] + 0.3*eps[0]
    #epsilon is one step ahead of xt
    for t in range(1, total_len):
        x[t] = 0.5*x[t-1] + eps[t+1] + 0.3*eps[t]
    #now, get the psis corresponding to d supplied
    psis_s = get_psis(j_max=total_len, d=d)
    #lastly, calculating the y's
    y = np.empty(total_len)
    for t in range(total_len):
        psi_oi = psis_s[0:t+1]
        x_oi = x[0:t+1][::-1]
        y[t] = np.sum(psi_oi * x_oi)
    return(y)

#!Whittle estimator and optimization functions
#from Assignment 9 we get the function for I_y (previously I_x)
def Ix(x_t, lambda_j): 
    '''
    I_x as defined on slide 144.
    '''
    T = len(x_t)
    factor = 1/(2*np.pi*T)
    cos_term = np.sum([x_t[t] * np.cos(lambda_j*t) for t in range(T)])
    sin_term = np.sum([x_t[t] * np.sin(lambda_j*t) for t in range(T)])
    return(factor * (cos_term**2 + sin_term**2))

def get_lambdas(m, T):
    '''
    Get lambda_j grid given m and T
    '''
    lambdas = np.array([2*np.pi*j/T for j in range(1, m+1)])
    return(lambdas)

def obj_fct(lambdas, Iy, d, m):
    '''
    Generate the value of the objective function for given m (which implictly defines lambdas, and thus Iy) and d
    '''
    lambdas_transform = lambdas**(-2*d)
    first = np.sum(Iy/lambdas_transform)
    second = np.sum(np.log(lambdas))
    final = np.log(1/m*first) - 2*d/m * second
    return(final)

def optimization(d_grid, lambdas, Iy, m):
    '''
    Minimize objective function on d grid
    '''
    values = np.asarray([obj_fct(d = d, lambdas=lambdas, Iy = Iy, m = m) for d in d_grid])
    min_i = np.where(values == min(values))
    result = d_grid[min_i]
    return(result)

#!Bias and coverage rate functions
def bias(estimates, true_val): 
    '''
    Calculate the (unscaled) bias of the estimates. 
    *true_val 
    *estimates
    '''
    #get number of estimates for a certain alpha
    S = estimates.shape[1]
    sum = np.sum(estimates, axis = 1)
    val = 1/S * sum - true_val 
    return(val)

def sigma_theory_w(m):
    '''
    Get theoretical sigma given m following slide 175.
    '''
    sigma2 = 1/(4*m)
    sigma = np.sqrt(sigma2)
    return(sigma)

def sigma_theory_pr(m):
    '''
    get theoretical variance of PR from slide 184
    '''
    sigma2 = np.pi**2/(24*m)
    sigma = np.sqrt(sigma2)
    return(sigma)

def CI_theory(true_val, sigma, n):
    '''
    Get theoretical CI.
    '''
    upper = true_val + 1.96*sigma/np.sqrt(n)
    lower = true_val - 1.96*sigma/np.sqrt(n)
    CI = np.vstack((lower, upper))
    return(CI)

def coverage_prob(CIs, estimates):
    cov_prob = np.empty(CIs.shape[1])
    CIs = CIs_whittle
    for i in range(CIs.shape[1]):
        lower, upper = CIs[0, i], CIs[1, i]
        in_CI = np.asarray([(val >= lower) & (val <= upper) for val in estimates[i, :]])
        summation = np.sum(in_CI)
        cov_prob[i] = summation/estimates.shape[1]
    return(cov_prob)

#!Replication study
M = 2000
T = 1000
S = 1000
alpha_list = np.arange(0.2, 0.8+0.01, 0.1)
d_grid = np.arange(-0.45, 0.5, 0.05)
#*Generating data before the actual loop
start = time.time()
data = np.empty((S, M+T))
for s in range(S):
    yt = data_gen(M+T, d=0.25)
    data[s, :] = yt
data = data[:, M:]
#*Result arrays 
opt_d_whittle = np.empty((len(alpha_list), S))
opt_d_regress = np.empty((len(alpha_list), S))
for i, alpha in enumerate(alpha_list):
    print(i)
    little_m = int(T**alpha)
    lambdas_a = get_lambdas(little_m, T)
    for s in range(S):
        yt_s = data[s, :]
        Iy_s = [Ix(yt_s, lam) for lam in lambdas_a]
        result = optimization(d_grid, lambdas_a, Iy_s, little_m)
        opt_d_whittle[i, s] = result
        #*log periodogram 
        Ij = np.log(Iy_s)
        rj = 4 * np.sin(lambdas_a/2)**2
        Rj = np.reshape(- np.log(rj), (len(rj), 1))
        const = np.ones(Rj.shape)
        exog = np.hstack((const, Rj))
        model = OLS(Ij, exog)
        results = model.fit()
        param = results.params[1]
        opt_d_regress[i, s] = param
        print(s)
end = time.time() 
print(end-start)

#get biases 
bias_whittle = abs(bias(opt_d_whittle, 0.25))
bias_pr = abs(bias(opt_d_regress, 0.25))
biases = np.vstack((bias_whittle, bias_pr))
print(biases)
#get variances 
var_whittle = np.var(opt_d_whittle, axis = 1)
var_pr = np.var(opt_d_regress, axis = 1)
vars = np.vstack((var_whittle, var_pr))
print(vars)

#!Coverage rate 
little_ms = np.array([int(T**a) for a in alpha_list])
#get sigmas
sigmas_whittle = sigma_theory_w(little_ms)
sigmas_pr = sigma_theory_pr(little_ms)
#get theoretical CIs
CIs_whittle = CI_theory(0.25, sigmas_whittle, n = T)
CIs_pr = CI_theory(0.25, sigmas_pr, n = T)
#Get coverage rate
cov_prob_w = coverage_prob(CIs_whittle, opt_d_whittle)
cov_prob_pr = coverage_prob(CIs_pr, opt_d_regress)
print(cov_prob_w)
print(cov_prob_pr)
#make quick latex table
cov_prob_df = pd.DataFrame((cov_prob_w, cov_prob_pr)).T
cov_prob_df = cov_prob_df.rename(columns = {0:'LW', 1: 'PR'})
cov_prob_df.to_latex('Cov_prob_table.tex')
#!Plotting
#bias plot
fig = plt.figure() 
plt.grid()
plt.plot(alpha_list, bias_whittle, label = r'${d}_{LW}$')
plt.plot(alpha_list, bias_pr, label =  r'${d}_{PR}$')
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel(r'bias')
plt.title('(Absolute) Biases of estimators')
plt.show()
fig.savefig('Biases.pdf', bboc_inches = 'tight')
#variance plots
fig2 = plt.figure() 
plt.grid()
plt.plot(alpha_list, var_whittle, label = r'${d}_{LW}$')
plt.plot(alpha_list, var_pr, label =  r'${d}_{PR}$')
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Variance')
plt.title('Variances of estimators')
plt.show()
fig2.savefig('Variances.pdf', bbox_inches = 'tight')
