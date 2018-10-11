import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('task2.jpg',0)
m = img.shape[0]
n = img.shape[1]

def showx():
    result = np.zeros(img.shape)
    for i in range(1,m-1):
        for j in range(1,n-1):
            result[i][j] = img[i-1][j+1]+2*img[i][j+1]+img[i+1][j+1]-img[i-1][j-1]-2*img[i][j-1]-img[i+1][j-1]
            if result[i][j] < 0:
                result[i][j] = -result[i][j]
    plt.imshow(result,'gray')
    plt.show()

def showy():
    result = np.zeros(img.shape)
    for i in range(1,m-1):
        for j in range(1,n-1):
            result[i][j] = img[i-1][j-1]+2*img[i-1][j-1]+img[i-1][j+1]-img[i+1][j-1]-2*img[i+1][j]-img[i+1][j+1]
            if result[i][j] < 0:
                result[i][j] = -result[i][j]
    plt.imshow(result,'gray')
    plt.show()
if __name__ == "__main__":
    showx()
    showy()
