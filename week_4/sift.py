import numpy as np
import math
import cv2 as cv
from skimage.exposure import rescale_intensity
# from sklearn.preprocessing import Normalizer
import sklearn.preprocessing
from scipy.ndimage import gaussian_filter

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
        kernel = [[1,4,6,4,1],
                  [4,16,24,16,4],
                  [6,24,36,24,6],
                  [4,16,24,16,4],
                  [1,4,6,4,1]]
        self.gauss_kernel = np.array(kernel)*(1/256)



    def scale_space(self):
        pass

    def generate_diff_of_gaussians(self):
        row,col,ch = self.image.shape
        gauss_imgs = [cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)] # Set first image as the grayscale of image

        # Generate an octave of Cascading Gaussian-Images
        # REFACTOR: make async as this could run into timing issues
        for i in range(self.s+3):
            print(f'i: {i+1}\n')
            if i == 0:
                gauss_imgs.append(gaussian_filter(gauss_imgs[0],1))
            else:
                gauss_imgs.append(gaussian_filter(gauss_imgs[i-1], 1))

        cv.imshow('gauss_imgs[0]', gauss_imgs[0])
        cv.imshow('gauss_imgs[1]', gauss_imgs[1])
        cv.imshow('gauss_imgs[2]', gauss_imgs[2])
        cv.imshow('gauss_imgs[3]', gauss_imgs[3])
        cv.imshow('gauss_imgs[4]', gauss_imgs[4])
        cv.imshow('gauss_imgs[5]', gauss_imgs[5])
        cv.imshow('gauss_imgs[6]', gauss_imgs[6])
        cv.imshow('gauss_imgs[7]', gauss_imgs[7])
        cv.imshow('gauss_imgs[8]', gauss_imgs[8])
        cv.waitKey()


    def generate_gaussian_kernel(self, x, y, sigma):
        '''
        :param x: int number of rows
        :param y: int number of columns
        :param sigma: int
        :return: numpy.array((x, y))
        '''
        # Allocate space for kernel
        kernel = np.zeros((x,y))
        x_vals = np.arange(-(x-1)/2, ((x-1)/2)+1)
        y_vals = np.arange(-(y-1)/2, ((y-1)/2)+1)

        # Iterate through each item in the kernel and input its Gaussian
        for i in np.arange(0, kernel.shape[0]):
            for j in np.arange(0, len(kernel[i])):
                e_numerator = -(x_vals[i]**2 + y_vals[j]**2)
                e_denominator = 2*sigma**2
                gaussian = (1/(2*math.pi*sigma**2)) * math.e**(e_numerator/e_denominator)
                rounded_gaussian = round(gaussian, 3)
                kernel[i][j] = rounded_gaussian

        # Need to normalize kernel
        # kernel = sklearn.preprocessing.normalize(kernel, norm='max')
        return kernel


    def convolve(self, image, kernel):
        '''
        :param image: MUST BE GRAYSCALE
        :return: convolved image
        Assumes a NxN kernel
        '''
        rows,cols = image.shape
        output = np.zeros((rows,cols), dtype='float32')

        # kernel = np.ones((21,21),dtype='float')*(1/(21*21))
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
                output[y-pad, x-pad] = result

        # Rescale back to 0-255 range
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")
        return output


if __name__ == '__main__':
    img = cv.imread('lenna.png')
    sift = Sift(img,4,5)
    sift.generate_diff_of_gaussians()
