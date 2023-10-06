# Tom Mulroy
# CS 549
# Prof. Lioulemes
# Lab 6
# Note: Some code is sourced from OpenCV Documentation
import cv2 as cv
import numpy as np

# Instantiation
sift = cv.SIFT_create()
bf = cv.BFMatcher()
MIN_MATCH_COUNT = 10

# Images
boston1 = cv.imread('boston1.jpeg')
boston2 = cv.imread('boston2.jpeg')
gray_boston1 = cv.cvtColor(boston1, cv.COLOR_BGR2GRAY)
gray_boston2 = cv.cvtColor(boston2, cv.COLOR_BGR2GRAY)

# PART 1
kp_boston1, des_boston1 = sift.detectAndCompute(gray_boston1, None)
kp_boston2, des_boston2 = sift.detectAndCompute(gray_boston2, None)
img_boston1 = cv.drawKeypoints(gray_boston1, kp_boston1, boston1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_boston2 = cv.drawKeypoints(gray_boston2, kp_boston2, boston2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Part 2
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_boston1,des_boston2,k=2)
good = []
for m,n in matches:
 if m.distance < 0.7*n.distance:
    good.append(m)

# Part 3
H = None
if len(good)>MIN_MATCH_COUNT:
 src_pts = np.float32([ kp_boston1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
 dst_pts = np.float32([ kp_boston2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
 matchesMask = mask.ravel().tolist()
 h,w,_ = boston1.shape
 pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
 dst = cv.perspectiveTransform(pts,H)
 img2 = cv.polylines(boston2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
 print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
 matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
 singlePointColor = None,
 matchesMask = matchesMask, # draw only inliers
 flags = 2)

img3 = cv.drawMatches(boston1,kp_boston1,boston2,kp_boston2,good,None,**draw_params)
# cv.imshow('Sift Matches', img3)
# cv.waitKey()

# Part 4
boston1_original = cv.imread('boston1.jpeg')
boston2_original = cv.imread('boston2.jpeg')
rows,cols,ch = boston2_original.shape
# print(f'boston1 shape: {boston1.shape}')
# print(f'boston2 shape: {boston2.shape}')

# When I use the H matrix from cv.findHomography(), it cuts off a lot of the image.
pts1 = np.float32([[0,0],[600,1],[1,958],[600,959]])
pts2 = np.float32([[0,0],[600,0],[0,900],[600,1200]])
M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(boston2_original,M,(boston1_original.shape[1],rows))
print(f'dst shape: {dst.shape}')
result = cv.hconcat([boston1_original[:, 0:650], dst])
cv.imshow('boston1', result)
cv.waitKey()

