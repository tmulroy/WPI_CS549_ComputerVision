import cv2
import numpy as np
import cv2 as cv


def custom_sobel(image, dx=0, dy=0):
    '''
    :param image: numpy.array
    :param dx: 1=detect vertical edges
    :param dy: 1=detect horizontal edges
    :return: numpy.array
    '''

    # Create output of same dimensions as input image
    image_height = image.shape[0]
    image_width = image.shape[1]
    output = np.array([image_height,image_width])
    # Define Kernels
    # X kernel detects vertical edges
    G_x = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ))
    # Y kernel detects horizontal edges
    G_y = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ))


    if dx == 1:
        output = abs(cv.filter2D(image, ddepth=-1,kernel=G_x))
    elif dy == 1:
        output = abs(cv.filter2D(image, ddepth=-1, kernel=G_y))

    return output
