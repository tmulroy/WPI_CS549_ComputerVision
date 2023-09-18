import cv2 as cv
from sobel import custom_sobel


def custom_laplacian(image):
    '''
    This image performs two passes of custom_sobel operator to
    detect horizontal and vertical edges
    It then combines the two images.
    :param image: numpy.array([x,y,3])
    :return:
    '''

    d_x = custom_sobel(image, dx=1, dy=0)
    ddx = custom_sobel(d_x, dx=1, dy=0)
    d_y = custom_sobel(image, dx=0, dy=1)
    ddy = custom_sobel(d_y, dx=0, dy=1)
    alpha = 0.5
    beta = 0.5
    output = cv.addWeighted(ddx,alpha,ddy,beta, 0.0)

    return output