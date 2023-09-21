import numpy as np
import math
import cv2 as cv
from skimage.exposure import rescale_intensity

'''
This class implements the first two steps in the SIFT feature detection algorithm
1. scale-space extrema detection
    a. Compute Differnce of Gaussians
    b. Local Extrama Detection
2. Accurate Keypoint Localization

Resources:
https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
'''

class Sift:
    def __init__(self, image, sigma, s):
        self.image = image
        self.gaussian_imgs = []
        self.diff_of_gauss_imgs = []
        self.sigma = sigma
        self.s = s
        self.k = 2**(1/s)



    def scale_space(self):
        pass

    def generate_diff_of_gaussians(self):
        gauss_cascade = 
        for i in range(self.s+3):
            print(f'i: {i+1}')


    def generate_gaussian_kernel(self, x, y, sigma):
        '''
        :param x: int number of rows
        :param y: int number of columns
        :param sigma: int
        :return: numpy.array((dimension, dimension))
        '''
        # Allocate space for kernel
        kernel = np.zeros((x,y))
        x_vals = np.arange(-(x-1)/2, ((x-1)/2)+1)
        y_vals = np.arange(-(y-1)/2, ((y-1)/2)+1)
        # print(f'x_vals: {x_vals}')

        # Iterate through each item in the kernel and input its Gaussian
        for i in np.arange(0, kernel.shape[0]):
            for j in np.arange(0, len(kernel[i])):
                e_numerator = -(x_vals[i]**2 + y_vals[j]**2)
                e_denominator = 2*sigma**2
                gaussian = (1/(2*math.pi*sigma**2)) * math.e**(e_numerator/e_denominator)
                rounded_gaussian = round(gaussian, 3)
                kernel[i][j] = rounded_gaussian
        return kernel

    def convolve(self, image, kernel=np.ones((3,3))):
        '''
        :param image:
        :return: convolved image
        Assumes a NxN kernel
        '''
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        rows,cols = image.shape
        output = np.zeros((rows,cols), dtype='float32')
        cv.imshow('Before',image)

        kernel = np.ones((21,21),dtype='float')
        # Extract Kernel Information
        kernel_height,kernel_width = kernel.shape
        multiplier = 1/np.size(kernel)
        kernel = kernel*multiplier

        # Add Padding to Original Image
        pad = (kernel_width-1)//2
        image = cv.copyMakeBorder(image,pad,pad,pad,pad,cv.BORDER_REPLICATE)

        # Perform convolution
        for y in np.arange(pad, rows+pad):
            for x in np.arange(pad, cols+pad):
                roi = image[y-pad:y+pad+1, x-pad:x+pad+1]
                result = (roi*kernel).sum()
                output[y-pad,x-pad] = result

        # Rescale back to 0-255 range
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        return output


if __name__ == '__main__':
    img = cv.imread('lenna.png')
    sift = Sift(img,4,5)
    # output = sift.convolve(img)
    # cv.imshow('After', output)
    # cv.waitKey()
    kernel = sift.generate_gaussian_kernel(5,5,1)
    sift.generate_diff_of_gaussians()
