import cv2
import numpy as np
import copy
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def erosion(img,template):
    result = copy.deepcopy(img)
    row,column = img.shape
    tr,tc = template.shape
    stripr = int(tr/2)
    stripc = int(tc/2)
    for i in range(stripr,row-stripr):
        for j in range(stripc,column-stripc):
            flag = True;
            for k in range(0,tr):
                for t in range(0,tc):
                    flag = flag&(img[i+k-stripr][j+t-stripc]!=0)
            if flag==True:
                result[i][j] = 255;
            else:
                result[i][j] = 0
    return result


def dilation(img,template):
    result = copy.deepcopy(img)
    row,column = img.shape
    tr,tc = template.shape
    stripr = int(tr/2)
    stripc = int(tc/2)
    for i in range(stripr,row-stripr):
        for j in range(stripc,column-stripc):
            flag = True;
            for k in range(0,tr):
                for t in range(0,tc):
                    flag = flag&(img[i+k-stripr][j+t-stripc]==0)
            if flag==True:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

def opening(img,template):
    erosionimg = erosion(img,template)
    dilationimg = dilation(erosionimg,template)
    return dilationimg

def closing(img,template):
    dilationimg = dilation(img,template)
    erosionimg = erosion(dilationimg,template)
    return erosionimg

def RemoveNoise(noise,path1,path2):
    img = cv2.imread(noise,0)
    template = np.ones(shape=(3,3))
    openimg = opening(img,template)
    closeimg = closing(img,template)
    cv2.imwrite(path1,openimg)
    cv2.imwrite(path2,closeimg)
    cv2.waitKey(0)
    
def ExtractBoundary(path1,path2,path3,path4):
    template = np.ones(shape=(3,3))
    res_noise1 = cv2.imread(path1,0)
    res_noise2 = cv2.imread(path2,0)
    erosionimg1 = erosion(res_noise1,template)
    res1 = res_noise1-erosionimg1;
    cv2.imwrite(path3,res1)
    erosionimg2 = erosion(res_noise2,template)
    res2 = res_noise2-erosionimg2
    cv2.imwrite(path4,res2)

if __name__ == "__main__":
    noise = "original_imgs/noise.jpg"
    folder = "part1_result"
    path1 = folder+"/res_noise1.jpg"
    path2 = folder+"/res_noise2.jpg"
    path3 = folder+"/res_bound1.jpg"
    path4 = folder+"/res_bound2.jpg"
    mkdir(folder)
    RemoveNoise(noise,path1,path2)
    ExtractBoundary(path1,path2,path3,path4)
