import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def addpad(image,pad):
    (rows,cols) = image.shape
    result = image
    rowzero = np.zeros(shape=(1,cols))
    for i in range(0,pad):
        result = np.insert(result,0,values=rowzero,axis=0)
    for i in range(0,pad):
        result = np.insert(result,result.shape[0],values=rowzero,axis=0)
    colzero = np.zeros(shape=(1,rows+2*pad))
    for j in range(0,pad):
        result = np.insert(result,0,values=colzero,axis=1)
    for j in range(0,pad):
        result = np.insert(result,result.shape[1],values=colzero,axis=1)
    return result

def scaling(image,percent):
    (rows,cols) = image.shape
    result = np.zeros(shape=(int(rows*percent),int(cols*percent)))
    (targetrows,targetcols) = result.shape
    for i in range(0,targetrows):
        for j in range(0,targetcols):
            sourceX = int(round(float(i)/targetrows*rows))
            sourceY = int(round(float(j)/targetcols*cols))
            result[i][j] = image[sourceX][sourceY]
    return result    

def genGaussianKernel(sigma,size):
    result = np.zeros(shape=(size,size),dtype=np.float)
    pivot = size/2;
    for i in range(0,size):
        for j in range(0,size):
            up = math.exp(-((i-pivot)**2+(j-pivot)**2)/(2.0*(sigma**2)))
            down = 2.0*math.pi*(sigma**2)
            result[i][j] = float(up/down)
    return result

def blur(img,GaussianMatrix):
    (rows,cols) = img.shape
    size = GaussianMatrix.shape[0]
    pivot = size/2
    result = np.zeros(shape=(rows,cols))
    padimg = addpad(img,pivot)
    (rows,cols) = padimg.shape
    for i in range(pivot,rows-pivot):
        for j in range(pivot,cols-pivot):
            value = 0.0
            for k in range(0,size):
                for l in range(0,size):
                    value = value+(padimg[i-(pivot-k)][j-(pivot-l)]*GaussianMatrix[k][l])
            result[i-pivot][j-pivot] = value
    return result

def generateimg(img,octave,size,sigma):
    result = []
    scalepercent = 1.0
    for i in range(0,octave-1):
        scalepercent = scalepercent*0.5
    for i in range(0,len(sigma)):
        GaussianMatrix = genGaussianKernel(sigma[i],size)
        scaleimg = scaling(img,scalepercent)
        print "Octave"+str(octave)+str(i+1)+":"+str(scaleimg.shape)
        blurimg = blur(scaleimg,GaussianMatrix)
        result.append(blurimg)
        plt.imshow(blurimg,'gray')
        plt.show()
    return result

def getfeatures(DoG):
    scalepercent = 1
    feature = []
    number = len(DoG)
    (rows,cols) = DoG[0].shape
    for k in range(1,number-1):
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                value = DoG[k][i][j]
                maximalleft = np.max(DoG[k-1][i-1:i+2,j-1:j+2])
                minimalleft = np.min(DoG[k-1][i-1:i+2,j-1:j+2])
                maximalright = np.max(DoG[k+1][i-1:i+2,j-1:j+2])
                minimalright = np.min(DoG[k+1][i-1:i+2,j-1:j+2])
                maximal = np.max(DoG[k][i-1:i+2,j-1:j+2])
                minimal = np.min(DoG[k][i-1:i+2,j-1:j+2])
                maximal = max(maximal,max(maximalleft,maximalright))
                minimal = min(minimal,min(minimalleft,minimalright))
                if value==maximal or value==minimal:
                    feature.append([i,j])
    return feature

if __name__ == "__main__":
    img = cv2.imread('task2.jpg',0)
    allimage = []
    octave1 = generateimg(img,1,7,[1.0/math.sqrt(2),1,math.sqrt(2),2,2*math.sqrt(2)])
    octave2 = generateimg(img,2,7,[math.sqrt(2),2,2*math.sqrt(2),4,4*math.sqrt(2)])
    octave3 = generateimg(img,3,7,[2*math.sqrt(2),4,4*math.sqrt(2),8,8*math.sqrt(2)])
    octave4 = generateimg(img,4,7,[4*math.sqrt(2),8,8*math.sqrt(2),16,16*math.sqrt(2)])
    allimage.append(octave1)
    allimage.append(octave2)
    allimage.append(octave3)
    allimage.append(octave4)
    
    rows = len(allimage)
    cols = len(allimage[0])
    allfeatures = []
    for i in range(0,rows):
        DoG = []
        for j in range(1,cols):
            difference = allimage[i][j]-allimage[i][j-1]
            DoG.append(difference)
            print "DoG"+str(i+1)+str(j)+":"
            plt.imshow(abs(difference),'gray')
            plt.show()
        feature = getfeatures(DoG)
        showimg = allimage[i][0]
        scalepercent = 1
        for count in range(0,i):
            scalepercent = scalepercent*2
        for elem in feature:
            showimg[elem[0]][elem[1]] = 255
            allfeatures.append([elem[0]*scalepercent,elem[1]*scalepercent])
        plt.imshow(showimg,'gray')
        plt.show()
    img_rgb = cv2.imread('task2.jpg')
    for elem in allfeatures:
        img_rgb[elem[0],elem[1]] = [255,255,255]
    cv2.imshow("All features",img_rgb)
    cv2.waitKey(0)

