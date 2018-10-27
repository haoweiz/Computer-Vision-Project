UBIT = "haoweizh"
import numpy as np
import os
import sys
import cv2
from matplotlib import pyplot as plt
np.random.seed(sum([ord(c) for c in UBIT]))

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

center = np.array([
[6.2,3.2],
[6.6,3.7],
[6.5,3.0]])

color = ['r','g','b']

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def reclassify(savepath):
    kind2index = dict()
    for i in range(0,X.shape[0]):
        dist = sys.maxint
        kind = -1
        for j in range(0,center.shape[0]):
            d = (X[i][0]-center[j][0])**2+(X[i][1]-center[j][1])**2
            if d<dist:
                dist = d
                kind = j
        if kind2index.has_key(kind)==False:
            kind2index[kind] = []
        kind2index[kind].append(X[i])
    ax = plt.subplot()
    for elem in kind2index:
        x_list = []
        y_list = []
        pts = kind2index[elem]
        for pt in pts:
            x_list.append(pt[0])
            y_list.append(pt[1])
        ax.scatter(x_list,y_list,c=color[elem],marker='^',s=50,alpha=1)
    print kind2index
    plt.savefig(savepath)
    plt.clf()
    return kind2index

def recompute_mean(kind2index,meanpath):
    ax = plt.subplot()
    for elem in kind2index:
        x_list = [0.0]
        y_list = [0.0]
        pts = kind2index[elem]
        for pt in pts:
            x_list[0] = x_list[0]+pt[0]
            y_list[0] = y_list[0]+pt[1]
        x_list[0] = x_list[0]/len(pts)
        y_list[0] = y_list[0]/len(pts)
        center[elem][0] = x_list[0]
        center[elem][1] = y_list[0]
        ax.scatter(x_list,y_list,c=color[elem],s=50,alpha=1)
        print x_list[0],y_list[0]
    plt.savefig(meanpath)
    plt.clf()

def initcenter(img,k):
    r,c,d = img.shape
    center = np.zeros(shape=(0,d))
    for i in range(0,k):
        x = np.random.randint(0,r)
        y = np.random.randint(0,c)
        p = img[x][y]  
        core = p[np.newaxis,:]
        center = np.r_[center,core]
    return center

def classifyimg(center,img):
    result = dict()
    r,c,d = img.shape
    for i in range(0,r):
        for j in range(0,c):
            dist = sys.maxint
            kind = -1
            for e in range(0,center.shape[0]):
                dt = 0
                for k in range(0,d):
                    dt += (img[i][j][k]-center[e][k])**2
                if dt<dist:
                    dist = dt
                    kind = e
            if result.has_key(kind)==False:
                result[kind] = []
            result[kind].append([i,j])
    return result

def recompute_center(img,classify,center):
    r,c,d = img.shape
    newcenter = np.zeros(shape=(0,d))
    for elem in classify:
        pts = classify[elem]
        ct = np.array([0,0,0])
        for pt in pts:
            for t in range(0,d):
                ct[t] = ct[t]+img[pt[0]][pt[1]][t]
        for t in range(0,d):
            ct[t] = ct[t]/len(pts)
        ct = ct[np.newaxis,:]
        newcenter = np.r_[newcenter,ct]
    return newcenter

def compare(oldcenter,newcenter):
    return (oldcenter==newcenter).all()

def color_quant(imgpath,outpath,k):
    img = cv2.imread(imgpath)
    oldcenter = initcenter(img,k)
    flag = True
    while flag==True:
        classify = classifyimg(oldcenter,img)
        newcenter = recompute_center(img,classify,oldcenter)
        if compare(oldcenter,newcenter)==True:
            r,c,d = img.shape
            for elem in classify:
                pts = classify[elem]
                for pt in pts:
                    for t in range(0,d):
                        img[pt[0]][pt[1]][t] = newcenter[elem][t]
            flag = False
        else:
            oldcenter = newcenter
    cv2.imwrite(outpath,img)

if __name__ == "__main__":
    folder = "part3_result"
    mkdir(folder)
    savepath = "part3_result/task3_iter1_a.jpg"
    kind2index = reclassify(savepath)
    meanpath = "part3_result/task3_iter1_b.jpg"
    recompute_mean(kind2index,meanpath)
    savepath = "part3_result/task3_iter2_a.jpg"
    kind2index = reclassify(savepath)
    meanpath = "part3_result/task3_iter2_b.jpg"
    recompute_mean(kind2index,meanpath)
    imgpath = "data/baboon.jpg"
    outpath = "part3_result/task3_baboon_3.jpg"
    color_quant(imgpath,outpath,3)
    outpath = "part3_result/task3_baboon_5.jpg"
    color_quant(imgpath,outpath,5)
    outpath = "part3_result/task3_baboon_10.jpg"
    color_quant(imgpath,outpath,10)
    outpath = "part3_result/task3_baboon_20.jpg"
    color_quant(imgpath,outpath,20)


