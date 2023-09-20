import cv2 as cv
import numpy as np


img = cv.imread('UnityHall.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# Image Information
rows, cols, ch = img.shape
smaller_img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
print(f'rows: {rows}')
print(f'cols: {cols}')

# Helper Functions
def harris_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)
    corners = cv.cornerHarris(gray_float32, 3, 5, 0.04)
    dilated = cv.dilate(corners, None)
    image[dilated>0.04*dilated.max()] = [0, 0, 255]
    return image

# ORIGINAL
# cv.imshow('Original', img)

# SCALE UP BY 20%
scaled_up_img = cv.resize(img, None, fx=1.2, fy=1.2, interpolation = cv.INTER_CUBIC)
# cv.imshow('Scaled Up', scaled_up_img)

# SCALE DOWN BY 20%
scaled_down_img = cv.resize(img, None, fx=0.8, fy=0.8, interpolation = cv.INTER_CUBIC)
# cv.imshow('Scaled Down', scaled_down_img)

# ROTATE by 10 degrees
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),10,1)
rotated_img = cv.warpAffine(img,M,(cols,rows))
# cv.imshow('Rotated',rotated_img)

# AFFINE TRANSFORM
# Scale down to fit in window
pts1 = np.float32([[0,0],[0,590],[984,0]])
pts2 = np.float32([[0,50],[0,640],[984,-50]])
M = cv.getAffineTransform(pts1,pts2)
affine_img = cv.warpAffine(smaller_img,M,(cols,rows))
# cv.imshow('Affine', affine_img)

# PERSPECTIVE TRANSFORM
# Using scaled-down image from affine transform
pers_pts1 = np.float32([[0,0], [984,10],[10,590],[590,980]])
pers_pts2 = np.float32([[50,50], [3000,0],[10,590],[700,1500]])
M = cv.getPerspectiveTransform(pers_pts1, pers_pts2)
perspective_img = cv.warpPerspective(smaller_img,M,(cols,rows))
# cv.imshow('Perspective', perspective_img)

# EDGE DETECTION
# Same sift for each image:
sift = cv.SIFT_create()

# Original Image Harris Corner Detection
harris_original = harris_detection(img)
cv.imshow('Harris Original', harris_original)

# Original Image SIFT Feature Detection
# kp_original = sift.detect(gray_original,None)
# sift_original = cv.drawKeypoints(gray_original,kp_original,img)
# cv.imshow('Original SIFT', sift_original)


# Rotated Image Harris Corner Detection
rotated_harris = harris_detection(rotated_img)
cv.imshow('Corners for Rotated Image', rotated_harris)

# Rotated SIFT Feature Detection
kp = sift.detect(rotated_img, None)
gray_sift_rotated = cv.cvtColor(rotated_img,cv.COLOR_BGR2GRAY)
rotated_sift_img = cv.drawKeypoints(gray_sift_rotated, kp, rotated_img)
cv.imshow('SIFT Rotated', rotated_sift_img)

# Scaled Up Harris Corner Detection
scaled_up_harris = harris_detection(scaled_up_img)
cv.imshow('Scaled Up Harris Corner Detection', scaled_up_harris)

# Scaled Up SIFT Feature Detection

k = cv.waitKey()
if k == 27:  # Exit Program
    print('Exiting program')
    cv.destroyAllWindows()