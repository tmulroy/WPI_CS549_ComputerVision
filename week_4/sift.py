import numpy as np
import math
import cv2 as cv
from skimage.exposure import rescale_intensity
import sklearn.preprocessing
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean

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
        # self.k = (2*self.sigma)/self.s

    def generate_diff_of_gaussians(self):
        octaves = []
        gray = cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
        gauss_imgs = [gray] # Set first image as the grayscale of image
        diff_of_gauss = []

        # Generate an octave of Cascading Gaussian-Images
        # REFACTOR: make async as this could run into timing issues

        sigma_incr = self.sigma
        # print(f'sigma_cr before: {sigma_incr}')
        # print(f'self.k: {self.k}')
        for i in range(self.s+3):
            if i == 0:
                gauss_imgs.append(gaussian_filter(gauss_imgs[0],sigma=sigma_incr))
            else:
                gauss_imgs.append(gaussian_filter(gauss_imgs[i-1], sigma=sigma_incr))
            # print(f'sigma_incr: {sigma_incr}')
            sigma_incr += self.k
        # print(f'sigma_cr after: {sigma_incr}')

        # Calculate Differnce of Gaussians
        for idx,x in enumerate(gauss_imgs):
            if idx != len(gauss_imgs)-1:
                # print(f' gauss_imgs[{idx}] - gauss_imgs[{idx + 1}]')
                diff = np.absolute(np.subtract(gauss_imgs[idx+1], gauss_imgs[idx]))
                diff_of_gauss.append(diff)

        octaves.append(diff_of_gauss)
        # cv.imshow('diff_of_gauss[0])', diff_of_gauss[0])
        # cv.imshow('diff_of_gauss[1])', diff_of_gauss[1])
        # cv.imshow('diff_of_gauss[2)', diff_of_gauss[2])
        # cv.imshow('diff_of_gauss[3)', diff_of_gauss[3])
        # cv.imshow('diff_of_gauss[4])', diff_of_gauss[4])
        # cv.imshow('diff_of_gauss[5])', diff_of_gauss[5])
        # cv.imshow('diff_of_gauss[6])', diff_of_gauss[6])
        # cv.imshow('diff_of_gauss[7])', diff_of_gauss[7])
        # cv.waitKey()

        # Down sample and repeat for new octave
        downsampled = self.downsample(gray)

    def downsample(self, image, factor=2):
        '''
        :param image: 2x2 numpy array GRAYSCALE
        :return:
        '''
        downsampled_img = image[::2,::2]
        return downsampled_img

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
    sift = Sift(img,1,5)
    sift.generate_diff_of_gaussians()
