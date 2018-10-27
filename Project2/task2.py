UBIT = "haoweizh"
import numpy as np
import cv2
import os
np.random.seed(sum([ord(c) for c in UBIT]))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def generate_Keypoints(imgpath,outputpath):
    img = cv2.imread(imgpath)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(img,None)
    imgkp = cv2.drawKeypoints(img,kp,img)
    cv2.imwrite(outputpath,imgkp)
    img = cv2.imread(imgpath)
    return img,kp,des

def draw_match(img1,kp1,des1,img2,kp2,des2,outputmatch):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good1 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good1.append(m)
    good2 = np.expand_dims(good1,1) 
    img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good2,None,flags=2)
    cv2.imwrite(outputmatch,img)
    return good1,good2,img,matches

def get_fundamental_matrix(kp1,kp2,matches):
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
    print F
    return F,mask,pts1,pts2

def drawlines(img1,img2,lines,pts1,pts2):
    r,c,d = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def get_epiline(numpairs,img1,img2,pts1,pts2,leftpath,rightpath):
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    ptsnum = pts1.shape[0]
    partpts1 = np.zeros(shape=(0,2),dtype='int32')
    partpts2 = np.zeros(shape=(0,2),dtype='int32')
    for i in range(0,numpairs):
        index = np.random.randint(0,ptsnum)
        pt1 = pts1[index]
        pt1 = pt1[np.newaxis,:]
        pt2 = pts2[index]
        pt2 = pt2[np.newaxis,:]
        partpts1 = np.r_[partpts1,pt1]
        partpts2 = np.r_[partpts2,pt2]
    lines1 = cv2.computeCorrespondEpilines(partpts2.reshape(-1,1,2),2,F)
    lines1 = lines1.reshape(-1,3)
    img3,img4 = drawlines(img1,img2,lines1,partpts1,partpts2)
    lines2 = cv2.computeCorrespondEpilines(partpts1.reshape(-1,1,2),1,F)
    lines2 = lines2.reshape(-1,3)
    img5,img6 = drawlines(img2,img1,lines2,partpts2,partpts1)
    cv2.imwrite(leftpath,img3)
    cv2.imwrite(rightpath,img5)

def disparity(img1,img2,disparitypath):
    stereo = cv2.StereoBM_create(numDisparities=48,blockSize=15)
    disparity = stereo.compute(img1,img2)
    cv2.imwrite(disparitypath,disparity)

if __name__ == "__main__":
    folder = "part2_result"
    mkdir(folder)
    imgpath1 = "data/tsucuba_left.png"
    outputpath1 = "part2_result/task2_sift1.jpg";
    img1,kp1,des1 = generate_Keypoints(imgpath1,outputpath1);
    imgpath2 = "data/tsucuba_right.png"
    outputpath2 = "part2_result/task2_sift2.jpg";
    img2,kp2,des2 = generate_Keypoints(imgpath2,outputpath2);
    outputmatch = "part2_result/task2_matches_knn.jpg"
    good1,good2,img,matches = draw_match(img1,kp1,des1,img2,kp2,des2,outputmatch)
    F,mask,pts1,pts2 = get_fundamental_matrix(kp1,kp2,matches)
    leftpath = "part2_result/task2_epi_left.jpg"
    rightpath = "part2_result/task2_epi_right.jpg"
    numpairs = 10
    get_epiline(numpairs,img1,img2,pts1,pts2,leftpath,rightpath)
    disparitypath = "part2_result/task2_disparity.jpg"
    img1 = cv2.imread(imgpath1,0)
    img2 = cv2.imread(imgpath2,0)
    disparity(img1,img2,disparitypath)

