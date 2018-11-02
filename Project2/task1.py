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
    return good1,good2,img

def get_homography_matrix(kp1,kp2,good1):
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2)
    H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    print H
    return H,mask
  

def draw_inliers(ranmatchnum,img1,kp1,img2,kp2,good1,H,mask,inlierpath):
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    masklen = len(matchesMask)
    partmatchesMask = []
    partgood1 = []
    for i in range(0,ranmatchnum):
        index = np.random.randint(0,masklen)
        partmatchesMask.append(matchesMask[index])
        partgood1.append(good1[index])
    draw_params = dict(matchesMask = partmatchesMask,flags = 2)
    img = cv2.drawMatches(img1,kp1,img2,kp2,partgood1,None,**draw_params)
    cv2.imwrite(inlierpath,img)

def splice(img1,img2,H,panopath):
    wrap = cv2.warpPerspective(img2,H,(img1.shape[1]+img1.shape[1],img2.shape[0]+img2.shape[0]))
    wrap[0:img1.shape[0],0:img1.shape[1]] = img1
    rows,cols = np.where(wrap[:,:,0] != 0)
    min_row,max_row = min(rows),max(rows)+1 
    min_col,max_col = min(cols),max(cols)+1 
    wrap = wrap[min_row:max_row,min_col:max_col,:]
    cv2.imwrite(panopath,wrap)

if __name__ == "__main__":
    folder = "part1_result"
    mkdir(folder)
    imgpath1 = "data/mountain1.jpg"
    outputpath1 = "part1_result/task1_sift1.jpg";
    img1,kp1,des1 = generate_Keypoints(imgpath1,outputpath1);
    imgpath2 = "data/mountain2.jpg"
    outputpath2 = "part1_result/task1_sift2.jpg";
    img2,kp2,des2 = generate_Keypoints(imgpath2,outputpath2);
    outputmatch = "part1_result/task1_matches_knn.jpg"
    good1,good2,img = draw_match(img1,kp1,des1,img2,kp2,des2,outputmatch)
    H,mask = get_homography_matrix(kp1,kp2,good1)
    inlierpath = "part1_result/task1_matches.jpg"
    ran_match_number = 15
    draw_inliers(ran_match_number,img1,kp1,img2,kp2,good1,H,mask,inlierpath)
    panopath = "part1_result/task1_pano.jpg"
    splice(img1,img2,H,panopath)
