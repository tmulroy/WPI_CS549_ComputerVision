import numpy as np
import cv2 as cv
import glob
# Note: some code used from OpenCV documentation

# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# images = glob.glob('*.jpg')
#
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (7,6), None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)
#
# # Camera Calibration
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# # Un-distortion
# img = cv.imread('left12.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)
#
# # Save camera calibration, dist coeff
# with open('./B.npz', 'wb') as f:
#     np.savez(f, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
# f.close()
#
# # Re-projection Error
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objpoints)) )

# Week 9 Code
# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def draw(img, corners, imgpts):

    corner = tuple((corners[0].ravel()))
    # Convert float to int for corner and imgpts
    new_imgpts = []
    for imgpt in imgpts:
        x = tuple([int(i) for i in imgpt[0]])
        new_imgpts.append(x)

    corner = tuple([int(x) for x in corner])

    img = cv.line(img, corner, new_imgpts[0], (255,0,0), 5)
    img = cv.line(img, corner, new_imgpts[1], (0,255,0), 5)
    img = cv.line(img, corner, new_imgpts[2], (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for fname in glob.glob('left*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        print(f'fname: {fname}')
        corners2 = cv.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        cv.imshow('img', img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)
cv.destroyAllWindows()