import numpy as np
import os
import copy
from scipy.stats import multivariate_normal
from error_ellipse import plot_point_cov
import matplotlib.pyplot as plt

u = np.array([
[4.0,81],
[2.0,57],
[4.0,71]])

var = np.array([
[[1.3,13.98],
[13.98,184.82]],
[[1.3,13.98],
[13.98,184.82]],
[[1.3,13.98],
[13.98,184.82]]
])

pi = np.array([1.0/3,1.0/3,1.0/3])

rgbcolor = ['r','g','b']

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def readfile(path):
    alldata = []
    with open(path) as file:
        line = file.readline()
        while line:
            oneline = line.split()
            data = []
            for i in range(1,len(oneline)):
                data.append(float(oneline[i]))
            alldata.append(data)
            line = file.readline()
    return np.array(alldata)

def Expectation(localu,localvar,localpi,alldata):
    probability = []
    for i in range(0,len(alldata)):
        total = 0.0
        for j in range(0,len(u)):
            total += localpi[j]*multivariate_normal.pdf(alldata[i],mean=localu[j],cov=localvar[j])
        prob = []
        for j in range(0,len(u)):
            up = localpi[j]*multivariate_normal.pdf(alldata[i],mean=localu[j],cov=localvar[j])
            prob.append(up/total)
        probability.append(prob)
    return probability


def Maximization(probability,localpi,alldata):
    newu = copy.deepcopy(u)
    newvar = copy.deepcopy(var)
    newpi = copy.deepcopy(pi)
    for j in range(0,len(newu)):
        Nk = 0.0
        up = [0.0,0.0]
        for i in range(0,len(alldata)):
            Nk += probability[i][j]
            up += probability[i][j]*alldata[i]
        newu[j] = up/Nk
        varup = np.array([[0.0,0.0],[0.0,0.0]])
        for i in range(0,len(alldata)):
            delta = alldata[i]-newu[j]
            delta_axis = delta[np.newaxis,:]
            varup += probability[i][j]*np.dot(np.transpose(delta_axis),delta_axis)
        newvar[j] = varup/Nk
        newpi[j] = Nk/len(alldata)
    return newu,newvar,newpi

def OneIter(localu,localvar,localpi,alldata,count):
    probability = Expectation(localu,localvar,localpi,alldata)
    newu,newvar,newpi = Maximization(probability,localpi,alldata)
    kind2index = classify(probability)
    draw(alldata,kind2index,count)
    return newu,newvar,newpi

def classify(probability):
    kind2index = dict()
    for i in range(0,len(probability)):
        kind = -1
        maxprob = 0.0
        for j in range(0,len(probability[i])):
            if probability[i][j]>maxprob:
                maxprob = probability[i][j]
                kind = j
        if kind not in kind2index:
            kind2index[kind] = []
        kind2index[kind].append(i)
    return kind2index

def draw(alldata,kind2index,count):
    for elem in kind2index:
        data = []
        for index in kind2index[elem]:
            data.append(alldata[index])
        data = np.array(data)
        x,y = np.array(data).T
        plt.plot(x, y, 'ro',color=rgbcolor[elem])
        plot_point_cov(data, nstd=3, alpha=0.5, color=rgbcolor[elem])
    plt.savefig("bonus_result/task3_gmm_iter"+str(count)+".jpg")
    plt.clf()

if __name__ == "__main__":
    folderpath = "bonus_result"
    mkdir(folderpath)
    alldata = readfile("data.txt")
    localu = copy.deepcopy(u)
    localvar = copy.deepcopy(var)
    localpi = copy.deepcopy(pi)
    count = 1
    while count<=5:
        newu,newvar,newpi = OneIter(localu,localvar,localpi,alldata,count)
        localu = newu
        localvar = newvar
        localpi = newpi
        count += 1

     
