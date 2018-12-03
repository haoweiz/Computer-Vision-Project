import cv2
import numpy as np
import os
import copy
import matplotlib.pyplot as plt

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def pointDetection(point,threshold):
    template = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    rows,cols = point.shape
    result = copy.deepcopy(point)
    tr,tc = template.shape
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            value = 0
            for k in range(0,tr):
                for t in range(0,tc):
                    value += template[k][t]*point[i+k-1][j+t-1]
            if value > threshold or value < -threshold:
                if i>200 and i<300:
                    print i,j
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result

def drawHistogram(img):
    data = img.flatten()
    plt.hist(data,bins=255)
    plt.show()

def Segmentation(segment,threshold):
    rows,cols = segment.shape
    result = copy.deepcopy(segment)
    for i in range(0,rows):
        for j in range(0,cols):
            if segment[i][j]<threshold:
                result[i][j] = 0
    left = cols
    up = rows
    right = 0
    down = 0
    for i in range(0,rows):
        for j in range(0,cols):
            if result[i][j] != 0:
                left = min(left,j)
                right = max(right,j)
                up = min(up,i)
                down = max(down,i)
    print left,up,right,down
    cv2.rectangle(segment,(left,up),(right,down),(128,128,128),4)
    #cv2.rectangle(result,(left,up),(right,down),(128,128,128),4)
    #cv2.imshow("result",result)
    #cv2.waitKey(0)
    #drawHistogram(segment)
    return result
                

if __name__ == "__main__":
    folder = "part2_result"
    mkdir(folder)
    point = cv2.imread("original_imgs/point.jpg",0)
    res1 = pointDetection(point,350)
    cv2.imwrite(folder+"/poros.jpg",res1)
    segment = cv2.imread("original_imgs/segment.jpg",0)
    res2 = Segmentation(segment,210)
    cv2.imwrite(folder+"/object.jpg",segment)
