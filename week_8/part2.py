import numpy as np
import cv2 as cv
import glob

# Your code should print and save the calibration matrix,
# the distortion coefficients
# and the re-projection error in a NumPy format file (npy).

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('left_images/*.jpg')
images_2 = glob.glob('calibration_data_/*.jpg')
good_images = {}

for fname in images_2:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    for i in range(3,11):
        for j in range(3,11):
            ret, corners = cv.findChessboardCorners(gray, (i,j), None)
            if ret:
                print(f'image: {fname}')
                print(f'i,j: {i},{j}')
                # good_images[(i,j)] = img
    # ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    # print(ret)


    # If found, add object points, image points (after refining them)
    # if ret == True:
    #     good_images.append(img)
    #     objpoints.append(objp)
    #     corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #     imgpoints.append(corners2)
    #     # Draw and display the corners
    #     cv.drawChessboardCorners(img, (7,5), corners2, ret)
    #     cv.imshow('img', img)
    #     cv.waitKey(500)


# Calibration
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# # Print Camera Calibration Matrix, Distortion Coeff, and Re-Projection Error
# print(f'Camera Calibration Matrix: {mtx}')
# print(f'Distortion Coefficients: {dist}')
#
# # Un-distort
# for fname in images:
#     img = cv.imread(fname)
#     h, w = img.shape[:2]
#     newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#
#     dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#     # crop the image
#     x, y, w, h = roi
#     dst = dst[y:y + h, x:x + w]
#     path = f'./calibresult/{fname[12:-4]}.png'
#     cv.imwrite(path, dst)
#
# # Re-Projection Error
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print(f'total re-projection error: {mean_error/len(objpoints)}')

cv.destroyAllWindows()