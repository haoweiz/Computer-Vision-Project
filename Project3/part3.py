import cv2
import numpy as np
import os
import math

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def sobelX(img):
    m,n = img.shape
    result = np.zeros(img.shape)
    for i in range(1,m-1):
        for j in range(1,n-1):
            result[i][j] = img[i-1][j+1]+2*img[i][j+1]+img[i+1][j+1]-img[i-1][j-1]-2*img[i][j-1]-img[i+1][j-1]
            if result[i][j] < 0:
                result[i][j] = -result[i][j]
    return result

def sobelY(img):
    m,n = img.shape
    result = np.zeros(img.shape)
    for i in range(1,m-1):
        for j in range(1,n-1):
            result[i][j] = img[i-1][j-1]+2*img[i-1][j-1]+img[i-1][j+1]-img[i+1][j-1]-2*img[i+1][j]-img[i+1][j+1]
            if result[i][j] < 0:
                result[i][j] = -result[i][j]
    return result


def Sobel(img):
    m,n = img.shape
    result = np.zeros(img.shape)
    sobelx = sobelX(img)
    sobely = sobelY(img)
    for i in range(1,m-1):
        for j in range(1,n-1):
            result[i][j] = int(0.5*sobelx[i][j]+0.5*sobely[i][j])
    for i in range(1,m-1):
        for j in range(1,n-1):
            if result[i][j]>100:
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result

def DrawOneLine(theta,r,result):
    rows,cols = result.shape
    for x in range(0,rows):
        y = int((r-x*math.cos(theta))/math.sin(theta))
        if y<cols and y>=0:
            result[x][y] = 255

def DrawRedLines(count,img,threshold):
    rows,cols = img.shape
    result = np.zeros(img.shape).astype(np.uint8)
    for key in count.keys():
        if count[key]<threshold:
            continue
        if key[0]<2:
            DrawOneLine(key[0],key[1],result)
    return result

def DrawBlueLines(count,img,threshold):
    rows,cols = img.shape
    result = np.zeros(img.shape).astype(np.uint8)
    for key in count.keys():
        if count[key]<threshold:
            continue
        if key[0]>2:
            DrawOneLine(key[0],key[1],result)
    return result

def HoughTransform(x,y,theta):
    return x*math.cos(theta)+y*math.sin(theta)

def detectLine(img,threshold,redpath,bluepath):
    edge = Sobel(img)
    edge = edge.astype(np.uint8)
    rows,cols = edge.shape
    count = dict()
    for i in range(0,rows):
        for j in range(0,cols):
            if edge[i][j]<128:
                continue
            theta = 0.1
            while theta<2*math.pi:
                r = HoughTransform(i,j,theta)
                r = int(r)
                if (theta,r) not in count:
                    count[(theta,r)] = 0
                count[(theta,r)] += 1
                theta += 0.1
    redLines = DrawRedLines(count,img,threshold)
    blueLines = DrawBlueLines(count,img,threshold)
    cv2.imwrite(redpath,redLines)
    cv2.imwrite(bluepath,blueLines)
    cv2.waitKey(0)

if __name__ == "__main__":
    folder = "part3_result"
    mkdir(folder)
    hough = cv2.imread("original_imgs/hough.jpg",0)
    redpath = folder+"/red_line.jpg"
    bluepath = folder+"/blue_line.jpg"
    detectLine(hough,180,redpath,bluepath)

