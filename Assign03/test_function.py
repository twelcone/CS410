import numpy as np
import math
def Sphere(ind):
    sum = 0
    for i in ind:
        sum += i**2
    return sum

def Rosenbrock(ind):
    sum = 0
    for i in range(len(ind) - 1):
        sum += 100 * (ind[i + 1] - ind[i]**2)**2 + (ind[i] - 1)**2
    return sum 

def Ackley(d):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = 0
    sum2 = 0
    for i in range(len(d)):
        sum1 += d[i] ** 2
        sum2 += np.cos(c * d[i])
    term1 = -a * np.exp(-b * np.sqrt(sum1 / len(d)))
    term2 = -np.exp(sum2 / len(d))

    return term1 + term2 + a + np.exp(1)
    
def Zakharov(ind):
    d = len(ind)
    sum1 = 0
    sum2 = 0
    for i in range(d):
        sum1 += ind[i]**2
        sum2 += 0.5 * (i+1) * ind[i]

    return sum1 + sum2**2 + sum2**4

def Michalewicz(ind):
    sum = 0
    for i in range(len(ind)):
        sum += np.sin(ind[i]) * (np.sin(((i+1) * ind[i]**2)/np.pi))**20
    return (-1*sum)
