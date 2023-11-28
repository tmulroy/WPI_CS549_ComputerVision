import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
# Part 0: Calibration3.3

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('left_images/*.jpg')
# images_2 = glob.glob('calibration_data_/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)


# Calibration Matrix
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print Camera Calibration Matrix, Distortion Coeff, and Re-Projection Error
print(f'Camera Calibration Matrix: {mtx}')
print(f'Distortion Coefficients: {dist}')

# Un-distort
for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    name = fname.partition('/')[2]
    img_name = name.partition('.')[0]
    path = f'./calibration_data_result/{img_name}_undistorted.png'
    cv.imwrite(path, dst)

# Re-Projection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print(f'total re-projection error: {mean_error/len(objpoints)}')

# Save Calibration Matrix, Distortion Coefficients, and Reprojection Error as npy files
with open('./B.npz', 'wb') as f:
    np.savez(f, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
f.close()

cv.destroyAllWindows()

# Part 1
# Capture two images

# Part 2
# Feature Extraction
sift = cv.SIFT_create()
img1 = cv.imread('image0.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('image1.jpg', cv.IMREAD_GRAYSCALE)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# kp3, des3 = sift.detectAndCompute(img3,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# matches_center_right = flann.knnMatch(des2,des3,k=2)


pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Now we have the list of best matches from both the images. Let's find the Fundamental Matrix.
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

# plt.subplot(121),plt.imshow(img5)
# plt.title('Left')
# plt.subplot(122),plt.imshow(img3)
# plt.title('Center')
# plt.show()

# Part 3: Fundamental Matrix
print(f'Fundamental Matrix: {F}')

# Part 4: Calculate Essential Matrix
# E = np.transpose(mtx)*F*mtx
# print(f'Essential Matrix: {E}')
# print(f'det(E)=0: {np.abs(np.linalg.det(E)) < 0.01}')
E,mask_ = cv.findEssentialMat(pts1,pts2,mtx)
E2,mask_2 = cv.findEssentialMat(pts2,pts1,mtx)
print(f'E_cv:\n{E}')
print(f'E2:\n{E2}')
print(f'E_cv det == 0: {np.abs(np.linalg.det(E)) < 0.001}')

# Part 5: Recover Rotation and Translation Vectors
points_12, R_12, t_12, mask_RP_12 = cv.recoverPose(E, pts1, pts2, mask=mask_)
points_21, R_21, t_21, mask_RP_21 = cv.recoverPose(E2,pts2,pts1,mask=mask_)
print(f'Rotation 1->2:\n{R_12}')
print(f'Translation 1->2:\n{t_12}')
print(f'Rotation 2->1:\n{R_21}')
print(f'Translation 2->1: {t_21}')

# Part 6: Get Projection Matrices
Rt_12 = np.concatenate((R_12,t_12), axis=1)
Rt_21 = np.concatenate((R_21,t_21), axis=1)

print(f'Rt_12:\n{Rt_12}')
print(f'Rt_21:\n{Rt_21}')

P0= np.dot(mtx,Rt_12)
P1= np.dot(mtx, Rt_21)
print(f'P0:\n{P0}')
print(f'P1:\n{P1}')

# Part 7
# u0 = cv.undistortPoints(pts1,mtx,dist)
# Need to undistort for each camera image
print(f'pts1 shape: {pts1[0]}')
# triangulated_points = cv.triangulatePoints(P0,P1,pts1,pts2)
def LinearLSTriangulation(u0, P0, u1, P1):
    pass

# Part 8: Re-projection Error
