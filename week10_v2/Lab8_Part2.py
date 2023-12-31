import numpy as np
import cv2 as cv
import glob

# Your code should print and save the calibration matrix,
# the distortion coefficients
# and the re-projection error in a NumPy format file (npy).

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('left_images/*.jpg')
images_2 = glob.glob('calibration_data_/*.jpg')

for fname in images_2:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (11,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (11,7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)


# Calibration Matrix
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print Camera Calibration Matrix, Distortion Coeff, and Re-Projection Error
print(f'Camera Calibration Matrix: {mtx}')
print(f'Distortion Coefficients: {dist}')

# Un-distort
for fname in images_2:
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
# Save camera calibration, dist coeff
with open('./B.npz', 'wb') as f:
    np.savez(f, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
f.close()

cv.destroyAllWindows()