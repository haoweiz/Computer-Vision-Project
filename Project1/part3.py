import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def sobel(img):
    gridimgx = cv2.Sobel(img,cv2.CV_16S,1,0)
    gridimgy = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(gridimgx)
    absY = cv2.convertScaleAbs(gridimgy)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst

if __name__ == "__main__":
    threshold = 5600000
    cursor = cv2.imread('task3/cursor.png',0)
    (currows,curcols) = cursor.shape
    template = cv2.resize(cursor,(curcols*2,currows*2))
    gridtemplate = sobel(template)
    print gridtemplate
    plt.imshow(gridtemplate,'gray')
    plt.show()
    for i in range(1,16):
        filename = "pos_"+str(i)+".jpg"
        imgfilename = "task3/"+filename
        img = cv2.imread(imgfilename,0)

        (imgrows,imgcols) = img.shape
        img2size = cv2.resize(img,(imgcols*2,imgrows*2))
        gridimg = sobel(img2size)

        result = cv2.matchTemplate(gridimg,gridtemplate,cv2.TM_SQDIFF)
        (minVal,maxVal,minLoc,maxLoc) = cv2.minMaxLoc(result)
        if minVal >= threshold:
            print minVal
            continue
        topleft = minLoc
        bottomright = (topleft[0]+template.shape[1],topleft[1]+template.shape[0])
        img_rgb = cv2.imread(imgfilename)
        img_rgb = cv2.resize(img_rgb,(imgcols*2,imgrows*2))
        cv2.rectangle(img_rgb,topleft,bottomright,255,2)
        print topleft,bottomright
        cv2.imwrite("./task3_positive_result/result_"+filename,img_rgb)
        #cv2.imshow("./result_"+filename,img_rgb)
        #cv2.waitKey(0)
