import numpy as np
import copy
from scipy.stats import multivariate_normal

X = np.array([
[5.9,3.2],
[4.6,2.9],
[6.2,2.8],
[4.7,3.2],
[5.5,4.2],
[5.0,3.0],
[4.9,3.1],
[6.7,3.1],
[5.1,3.8],
[6.0,3.0]])

u = np.array([
[6.2,3.2],
[6.6,3.7],
[6.5,3.0]])

var = np.array([
[[0.5,0],
[0,0.5]],
[[0.5,0],
[0,0.5]],
[[0.5,0],
[0,0.5]]
])

pi = np.array([1.0/3,1.0/3,1.0/3])

def Expectation(localu,localvar,localpi):
    probability = []
    for i in range(0,len(X)):
        total = 0.0
        for j in range(0,len(u)):
            total += localpi[j]*multivariate_normal.pdf(X[i],mean=localu[j],cov=localvar[j])
        prob = []
        for j in range(0,len(u)):
            up = localpi[j]*multivariate_normal.pdf(X[i],mean=localu[j],cov=localvar[j])
            prob.append(up/total)
        probability.append(prob)
    return probability

def Maximization(probability,localpi):
    newu = copy.deepcopy(u)
    newvar = copy.deepcopy(var)
    newpi = copy.deepcopy(pi)
    for j in range(0,len(newu)):
        Nk = 0.0
        up = [0.0,0.0]
        for i in range(0,len(X)):
            Nk += probability[i][j]
            up += probability[i][j]*X[i]
        newu[j] = up/Nk
        varup = np.array([[0.0,0.0],[0.0,0.0]])
        for i in range(0,len(X)):
            delta = X[i]-newu[j]
            delta_axis = delta[np.newaxis,:]
            varup += probability[i][j]*np.dot(np.transpose(delta_axis),delta_axis)
        newvar[j] = varup/Nk
        newpi[j] = Nk/len(X)
    return newu,newvar,newpi

def GMMOneiter():
    localu = copy.deepcopy(u)
    localvar = copy.deepcopy(var)
    localpi = copy.deepcopy(pi)
    probablity = Expectation(localu,localvar,localpi)
    newu,newvar,newpi = Maximization(probablity,localpi)
    print newu

if __name__ == "__main__":
    GMMOneiter()
