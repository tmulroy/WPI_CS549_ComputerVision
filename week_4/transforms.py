import cv2 as cv
import numpy as np


img = cv.imread('UnityHall.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# Determines if application should save images to local directory
save_flag = True

# Image Information
rows, cols, ch = img.shape
smaller_img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

# Helper Functions
def harris_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)
    corners = cv.cornerHarris(gray_float32, 3, 5, 0.04)
    dilated = cv.dilate(corners, None)
    image[dilated>0.04*dilated.max()] = [0, 0, 255]
    return image

sift = cv.SIFT_create()

def sift_detection(image, sift=sift):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    sifted_img = cv.drawKeypoints(gray, kp, image)
    return sifted_img

# SCALE UP BY 20%
scaled_up_img = cv.resize(img, None, fx=1.2, fy=1.2, interpolation = cv.INTER_CUBIC)

# SCALE DOWN BY 20%
scaled_down_img = cv.resize(img, None, fx=0.8, fy=0.8, interpolation = cv.INTER_CUBIC)

# ROTATE by 10 degrees
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),10,1)
rotated_img = cv.warpAffine(img,M,(cols,rows))

# AFFINE TRANSFORM
# Scale down to fit in window
pts1 = np.float32([[0,0],[0,590],[984,0]])
pts2 = np.float32([[0,50],[0,640],[984,-50]])
M = cv.getAffineTransform(pts1,pts2)
affine_img = cv.warpAffine(smaller_img,M,(cols,rows))

# PERSPECTIVE TRANSFORM
# Using scaled-down image from affine transform
pers_pts1 = np.float32([[0,0], [984,10],[10,590],[590,980]])
pers_pts2 = np.float32([[50,50], [3000,0],[10,590],[700,1500]])
M = cv.getPerspectiveTransform(pers_pts1, pers_pts2)
perspective_img = cv.warpPerspective(smaller_img,M,(cols,rows))

# Show images

images = {
    'Original': img,
    'Rotated': rotated_img,
    'Scaled Up': scaled_up_img,
    'Scaled Down': scaled_down_img,
    'Affine': affine_img,
    'Perspective': perspective_img
}

if save_flag:
    for name, img in images.items():
        cv.imwrite(f'{name}.jpg', img)
        harris = harris_detection(img)
        cv.imwrite(f'{name}_Harris.jpg', harris)
        sift = sift_detection(img)
        cv.imwrite(f'{name}_SIFT.jpg', sift)

k = cv.waitKey()
if k == 27:  # Exit Program
    print('Exiting program')
    cv.destroyAllWindows()
else:
    pass