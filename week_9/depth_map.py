import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('aloeL.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('aloeR.jpg', cv.IMREAD_GRAYSCALE)
stereo = cv.StereoBM.create(numDisparities=128, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity)
plt.show()

# cv.imwrite('disparity.jpg', disparity)
# cv.imshow('disp', disparity)
# cv.waitKey()