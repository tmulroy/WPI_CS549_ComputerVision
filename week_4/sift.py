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
    a. Compute Difference of Gaussians
    b. Local Extrema Detection
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
        self.gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.num_of_octaves = 4

    def calculate_scale_space_extreme(self):
        '''
        This function calcualtes the scale space extrema
        :return:
        REFACTOR: make helper functions async
                Does a new sigma need to double for each downsampled image?
        '''

        img = self.gray
        diff_of_gauss_octaves = []
        local_sigma = self.sigma
        for i in range(0, self.num_of_octaves):
            octave = self.generate_octave(local_sigma, img)
            diff_of_gauss = self.generate_diff_of_gauss(octave)
            diff_of_gauss_octaves.append(diff_of_gauss)
            img = self.downsample(octave[-3])
            local_sigma *= 2

        # diff_of_gauss_octaves[octave index][image index]
        self.local_extrema_detection(diff_of_gauss_octaves)

    def local_extrema_detection(self, octaves):
        '''
        :param octaves: list of Difference of Gaussian Octaves
        :return:
        REFACTOR: there has to be a more efficient way to calculate this than O(n^4)...
        '''
        imgs = []
        pad = 1
        keypoints = []
        # Add padding to all images before iterating to find extrema O(n^2) (not accounting for cv.copyMakeBorder()
        for octave in octaves:
            for img_idx, img in enumerate(octave):
                img = cv.copyMakeBorder(img,pad,pad,pad,pad,cv.BORDER_REPLICATE)

        # Iterate through each DoG Image and determine if it is a local extrema
        for octave in octaves:
            for img_idx, img in enumerate(octave):
                neighbor_extrema_flag = False
                next_img_extrema_flag = False
                prev_img_extrema_flag = False
                rows,cols = img.shape
                for y in np.arange(pad, rows):
                    for x in np.arange(pad, cols):
                        neighbors = img[y-pad:y+pad+1, x-pad:x+pad+1]
                        pixel_val = neighbors[1,1]
                        if pixel_val > np.max(neighbors) or pixel_val < np.min(neighbors):
                            neighbor_extrema_flag = True

                        if img_idx == 0: # Special Case: First Image in an Octave
                            next_img = octave[1]
                            next_img_window = next_img[y-pad:y+pad+1, x-pad:x+pad+1]
                            if pixel_val > np.max(next_img_window) or pixel_val < np.min(next_img_window):
                                next_img_extrema_flag = True
                        elif  img_idx == len(octave) - 1: # Special Case: Last Image in an Octave
                            prev_img = octave[img_idx - 1]
                            prev_img_window = prev_img[y - pad:y + pad + 1, x - pad:x + pad + 1]
                            if pixel_val > np.max(prev_img_window) or pixel_val < np.min(prev_img_window):
                                prev_img_extrema_flag = True
                        else: # Every image besides first and last of an octave
                            next_img = octave[img_idx+1]
                            next_img_window = next_img[y - pad:y + pad + 1, x - pad:x + pad + 1]
                            prev_img = octave[img_idx-1]
                            prev_img_window = prev_img[y - pad:y + pad + 1, x - pad:x + pad + 1]
                            if pixel_val > np.max(next_img_window) or pixel_val < np.min(next_img_window):
                                next_img_extrema_flag = True
                            if pixel_val > np.max(prev_img_window) or pixel_val < np.min(prev_img_window):
                                prev_img_extrema_flag = True



                        if neighbor_extrema_flag or next_img_extrema_flag or prev_img_extrema_flag:
                            # print(f'keypoint found: image {img_idx}, {y},{x}')
                            keypoints.append((y,x))




    def generate_diff_of_gauss(self, octave):
        '''
        :param octave: list of numpy.ndarray((2,2))
        :return: diff_of_gauss: list of numpy.ndarray((2,2))
        This function calculates the difference of a list of adjacent gaussian-convolved images
        '''
        diff_of_gauss = []
        for idx,x in enumerate(octave):
            if idx != len(octave)-1:
                diff = np.absolute(np.subtract(octave[idx+1], octave[idx]))
                diff_of_gauss.append(diff)
        return diff_of_gauss

    def generate_octave(self, starting_sigma, gray_image):
        '''
        :param starting_sigma: int
        :param gray_image: numpy.ndarry((2,2))
        :return: octave: list of numpy.ndarray((2,2))
        Generate an Octave of Original Gray Image Convolved with Increasing Sigmas (k*sigma)
        Refactor: confirm that sigma is getting calculated correctly
        '''
        octave = []
        sigma_incr = starting_sigma
        for i in range(0, self.s+3):
            octave.append(gaussian_filter(gray_image, sigma=sigma_incr))
            sigma_incr = sigma_incr*self.k
        return octave

    def downsample(self, image, factor=2):
        '''
        :param image: 2x2 numpy array GRAYSCALE
        :param factor: every nth pixel to be taken from original image
        :return: downsampled_img: numpy.ndarray((0.5*original,0.5*original))
        '''
        downsampled_img = image[::factor,::factor]
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
    sift.calculate_scale_space_extreme()
