'blur_two_stage.py\n\nA two-stage Gaussian blur same as the Halide image processing language as in\nhttps://github.com/halide/Halide/tree/master/apps/blur\n\nAuthor: Yuting Yang\n'
import numpy as np
import sys
sys.path += ['../../compiler']
import util
from blur_config import *

def two_stage_blur(input_img):
    'Blurs the image in matrix input_img and writes the values to output_img.\n    \n    The first stae uses a horizontal convolution\n    \n    Kernel1 = 1/3 1/3 1/3\n    \n    And the second stage uses a vertical convolution\n    \n    Kernel2 = 1/3 1/3 1/3\n    \n    For dealing with convolving along the edges of the image, we use the same way as Halide\n    to compare with their results. We simply discard the two horizontal edges, and the left vertical\n    edge, and 7 columns on the very left. The reason for this still need to be explored.\n    \n    Args:\n        input_img, np.ndarray, reference to input image to blur\n        output_img, np.ndarray, reference to array to save output image in\n    '
    pass
    output_img = np.zeros(input_img.shape)
    pass
    for r in range((input_img.shape[0] - 8)):
        for c in range((input_img.shape[1] - 2)):
            if ((r >= 0) and (r < (input_img.shape[0] - 8)) and ((c + 2) >= 1) and ((c + 2) < (input_img.shape[1] - 1)) and (r >= 0) and (r < (input_img.shape[0] - 8)) and (c >= 1) and (c < (input_img.shape[1] - 1))):
                output_img[(r, c)] = ((((((input_img[(r, c)] + input_img[((r + 1), c)]) + input_img[((r + 2), c)]) / 3.0) + (((input_img[(r, (c + 1))] + input_img[((r + 1), (c + 1))]) + input_img[((r + 2), (c + 1))]) / 3.0)) + (((input_img[(r, (c + 2))] + input_img[((r + 1), (c + 2))]) + input_img[((r + 2), (c + 2))]) / 3.0)) / 3.0)
            elif ((r >= 0) and (r < (input_img.shape[0] - 8)) and ((c + 2) == (input_img.shape[1] - 1)) and (r >= 0) and (r < (input_img.shape[0] - 8)) and (c >= 1) and (c < (input_img.shape[1] - 1))):
                output_img[(r, c)] = ((((((input_img[(r, c)] + input_img[((r + 1), c)]) + input_img[((r + 2), c)]) / 3.0) + (((input_img[(r, (c + 1))] + input_img[((r + 1), (c + 1))]) + input_img[((r + 2), (c + 1))]) / 3.0)) + (((input_img[(r, (input_img.shape[1] - 1))] + input_img[((r + 1), (input_img.shape[1] - 1))]) + input_img[((r + 2), (input_img.shape[1] - 1))]) / 3.0)) / 3.0)
            elif ((r >= 0) and (r < (input_img.shape[0] - 8)) and ((c + 2) >= 1) and ((c + 2) < (input_img.shape[1] - 1)) and (r >= 0) and (r < (input_img.shape[0] - 8)) and (c == 0)):
                output_img[(r, c)] = ((((((input_img[(r, 0)] + input_img[((r + 1), 0)]) + input_img[((r + 2), 0)]) / 3.0) + (((input_img[(r, (c + 1))] + input_img[((r + 1), (c + 1))]) + input_img[((r + 2), (c + 1))]) / 3.0)) + (((input_img[(r, (c + 2))] + input_img[((r + 1), (c + 2))]) + input_img[((r + 2), (c + 2))]) / 3.0)) / 3.0)
            elif ((r >= 0) and (r < (input_img.shape[0] - 8)) and ((c + 2) == (input_img.shape[1] - 1)) and (r >= 0) and (r < (input_img.shape[0] - 8)) and (c == 0)):
                output_img[(r, c)] = ((((((input_img[(r, 0)] + input_img[((r + 1), 0)]) + input_img[((r + 2), 0)]) / 3.0) + (((input_img[(r, (c + 1))] + input_img[((r + 1), (c + 1))]) + input_img[((r + 2), (c + 1))]) / 3.0)) + (((input_img[(r, (input_img.shape[1] - 1))] + input_img[((r + 1), (input_img.shape[1] - 1))]) + input_img[((r + 2), (input_img.shape[1] - 1))]) / 3.0)) / 3.0)
    return output_img
input_img_rgb = util.image_filename('temple_rgb.png')
input_img_gray = util.image_filename('temple_gray.png')

def test(n=None):
    'Default unit tests which compare any optimized output with the original Python output.\n    '
    L = []
    if do_rgb:
        L.append(util.test_image_pipeline_filename(two_stage_blur, (input_img_rgb,), n, name='blur_two_stage.gaussian_blur rgb'))
    if do_gray:
        L.append(util.test_image_pipeline_filename(two_stage_blur, (input_img_gray,), n, grayscale=True, name='blur_two_stage.gaussian_blur gray'))
    return util.combine_tests(L)
if (__name__ == '__main__'):
    test()