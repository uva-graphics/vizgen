"""blur_two_stage.py

A two-stage Gaussian blur same as the Halide image processing language as in
https://github.com/halide/Halide/tree/master/apps/blur

Author: Yuting Yang
"""

import numpy as np
import sys; sys.path += ['../../compiler']
import util

def two_stage_blur(input_img):
    """Blurs the image in matrix input_img and writes the values to output_img.
    
    The first stae uses a horizontal convolution
    
    Kernel1 = 1/3 1/3 1/3
    
    And the second stage uses a vertical convolution
    
    Kernel2 = 1/3 1/3 1/3
    
    For dealing with convolving along the edges of the image, we use the same way as Halide
    to compare with their results. We simply discard the two horizontal edges, and the left vertical
    edge, and 7 columns on the very left. The reason for this still need to be explored.
    
    Args:
        input_img, np.ndarray, reference to input image to blur
        output_img, np.ndarray, reference to array to save output image in
    """
    
    temp_img = np.zeros(input_img.shape)
    
    output_img = np.zeros(input_img.shape)
    
    #first stage blur
    for r in range(input_img.shape[0]-8):
        for c in range(input_img.shape[1]):
            
            temp_img[r, c] = (input_img[r, c] + input_img[r + 1, c] + input_img[r + 2, c]) / 3.0
    
    #second stage blur
    for r in range(input_img.shape[0]-8):
        for c in range(input_img.shape[1]-2):
        
            output_img[r, c] = (temp_img[r, c] + temp_img[r, c + 1] + temp_img[r, c + 2]) / 3.0
    
    return output_img

input_img_rgb = util.image_filename('temple_rgb.png')
input_img_gray = util.image_filename('temple_gray.png')

def test(n=None):
    """Default unit tests which compare any optimized output with the original Python output.
    """
    L = []
    #if do_rgb:
    if False:
        L.append(util.test_image_pipeline_filename(two_stage_blur, (input_img_rgb,), n, name = 'blur_two_stage.gaussian_blur rgb'))
    #if do_gray:
    if True:
        L.append(util.test_image_pipeline_filename(two_stage_blur, (input_img_gray,), n, grayscale = True, name = 'blur_two_stage.gaussian_blur gray'))
    return util.combine_tests(L)
    
if __name__ == '__main__':
    test()
    