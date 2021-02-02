import numpy as np 
from numpy import random as rd
import pandas as pd 
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#!--------------------
#!Defining functions 
#!--------------------
#these contain complex numbers, so to be able to plot them we can 
#calculate the Euclidean norm 

def euclid_norm(complex): 
    real = complex.real 
    imag = complex.imag
    norm = np.sqrt(real**2 + imag **2)
    return(norm)

#spectral density
#have shown what it looks like on paper 
#just hardcode it 

def spec_dens(lam, sigma = 1, q = 3):
    double_sum = 0
    for k in range(-q,q+1):
        for i in range(-q,q+1):
            power = (i-k)*1j*lam
            e_term = np.exp(power)
            double_sum = double_sum + e_term
    f = sigma/(2*np.pi) * 1/((2*q +1)**2) * (double_sum)
    return(f)

#defining the integral function as given in the assignment using scipy.integrate.quad which
#is standard integration, takes function, limits and list of values for other arguments then the 
#first one of the supplied function over which is automatically integrated
def s(lambda_0, sigma, q): 
    integral = integrate.quad(spec_dens, 0, lambda_0, args = (sigma, q))
    value = 2*integral[0]
    return(value)

#*create a grid of lambdas
lambdas = np.arange(0, np.pi, 0.001)
#*define list of q's we are interested in 
qs = [1, 3, 10]

#now to the task
#*first part
#generate an empty array that is of dimension len(q)xlen(lambdas), 
#i.e. this is an aray consisting of 3 arrays, each of length = len(lambdas), 
#such that we have one array for each q and all together in one array
#loop over the list of qs 
#enumerate creates a tuple of index and value of supplied list, e.g. (0, val[0]), (1, val[1]) 
#and so on if supplied list is val
spec_densities = np.empty((len(qs), len(lambdas)))
for i, q in enumerate(qs):
    spec_densities[i] = [spec_dens(lam = l, q = q) for l in lambdas]

#now same structure of data as before but we apply the euclidean norm function 
#to each element 
spec_dens_norm = np.empty((len(qs), len(lambdas)))
for i in range(len(spec_densities)):
    spec_dens_norm[i] = [euclid_norm(x) for x in spec_densities[i]]

#data is now ready to be plotted which is at end of code

#*second part 
#again same idea of structure of data
#create an array of dim 3x(number of lambdas) which stores s values for each 
#q in one sub array (1, 2, 3)
#have 3 different q
#results containes the arrays of the integration results
results = np.empty((3, len(lambdas)))
for i, q in enumerate(qs):
    results_q = [s(lam0, sigma = 1, q = q) for lam0 in lambdas]
    results[i] = results_q
    print(i)
#so element i of results array is array containing values of s(lambda_0) for a given q

#the last element of each of these arrays is s(pi; q)
#thus we get VR(lam0;q) by dividing the whole array by its last element 
VR_q = [results[i]/results[i][-1] for i in range(len(results))]

#!--------------------
#!Plotting 
#!--------------------
#*first part
f = plt.figure()
plt.grid()
#plotting the lines
for i in range(len(results)):
    label = 'q = ' + str(qs[i])
    plt.plot(lambdas, spec_dens_norm[i], label = label)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$f_x(\lambda)$')
plt.legend()
plt.show()
f.savefig("spec_densities.pdf", bbox_inches = 'tight')

#*second part
f2 = plt.figure()
plt.grid()
#plotting the different VR(lam0)
for i in range(len(results)):
    label = r'$VR(\lambda_0; q = $' + str(qs[i]) + ')'
    plt.plot(lambdas, VR_q[i], label = label)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$VR(\lambda_0; q)$')
plt.legend()
plt.show()
f2.savefig("VR.pdf", bbox_inches = 'tight')