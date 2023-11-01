import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('texas.png', cv.IMREAD_GRAYSCALE)
img_color = cv.imread('texas.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# Canny Edge Detection
edges = cv.Canny(img, 100, 200, apertureSize=3)

# Hough Transform Line Detection
lines = cv.HoughLines(edges, 0.85, np.pi/180, 300)
r,_,c = lines.shape

A = []
B = []
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    A.append([rho,theta])
    B.append(b)
    cv.line(img_color,(x1,y1),(x2,y2),(0,0,255),2)
# cv.imwrite('houghlines3.jpg', img)

A = np.array(A)
B = np.array(B)

# Calculate t = ((A^T) * A)^1 * ((A^T) * b)
term1 = np.power(np.dot(np.transpose(A),A),-1)
term2 = np.dot(np.transpose(A), B)
t = np.dot(term1, term2)
print(t.shape)
coords = (t[0].astype(np.int64), t[1].astype(np.int64))
print(coords)
cv.circle(img_color, coords, 20, (0,0,255), cv.FILLED)


# lines = cv.HoughLinesP(edges,1,np.pi/180,200,minLineLength=100,maxLineGap=5)
# lines = cv.HoughLinesP(edges,2,np.pi/180,200,minLineLength=200,maxLineGap=9)
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv.imwrite('houghlines5.jpg',img)


cv.imshow('vanishing point', img_color)
cv.waitKey()
# Show Image
# plt.subplot(121),plt.imshow(img_color)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()