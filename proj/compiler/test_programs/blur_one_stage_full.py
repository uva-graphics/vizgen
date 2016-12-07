"""blur_one_stage.py

An intentionally slow implementation of a Gaussian blur using a 3x3 kernel.

Author: Sam Prestwood
"""

import numpy as np
import sys; sys.path += ['../../compiler']
import util

def gaussian_blur(input_img):
    """Blurs the image in matrix input_img and writes the values to output_img.

    This uses a 3x3 Gaussian kernel to convolve with the image matrix.

                1/16 2/16 1/16
    Kernel =    2/16 4/16 2/16
                1/16 2/16 1/16

    For dealing with convolving along the edges of the image, we renormalize the
    kernel based on which coordinates from the kernel are in-bounds.

    Args:
        input_img, np.ndarray, reference to input image to blur
        output_img, np.ndarray, reference to array to save output image in
    """

    output_img = np.zeros(input_img.shape)

    for r in range(input_img.shape[0]):
        for c in range(input_img.shape[1]):
            # center
            kernel_accum = 4.0 * input_img[r, c]
            kernel_norm = 4.0

            # top left
            if r > 0 and c > 0:
                kernel_accum += 1.0 * input_img[r - 1, c - 1]
                kernel_norm += 1.0

            # top middle
            if r > 0:
                kernel_accum += 2.0 * input_img[r - 1, c    ]
                kernel_norm += 2.0

            # top right
            if r > 0 and c < input_img.shape[1] - 1: 
                kernel_accum += 1.0 * input_img[r - 1, c + 1]
                kernel_norm += 1.0

            # left
            if c > 0:
                kernel_accum += 2.0 * input_img[r    , c - 1]
                kernel_norm += 2.0

            # right
            if c < input_img.shape[1] - 1:
                kernel_accum += 2.0 * input_img[r    , c + 1]
                kernel_norm += 2.0

            # bottom left
            if r < input_img.shape[0] - 1 and c > 0:
                kernel_accum += 1.0 * input_img[r + 1, c - 1]
                kernel_norm += 1.0

            # bottom middle
            if r < input_img.shape[0] - 1:
                kernel_accum += 2.0 * input_img[r + 1, c    ]
                kernel_norm += 2.0

            # bottom right
            if r < input_img.shape[0] - 1 and c < input_img.shape[1] - 1:
                kernel_accum += 1.0 * input_img[r + 1, c + 1]
                kernel_norm += 1.0

            output_img[r, c] = kernel_accum / kernel_norm

    return output_img

input_img_rgb = util.image_filename('small_temple_rgb.png')
input_img_gray =util.image_filename('small_temple_gray.png')

def test(n=1):
    """Default unit tests which compare any optimized output with the original Python output.
    """
    gaussian_blur(np.zeros((100,100)))
    gaussian_blur(np.zeros((100,100,3)))
    ans1 = util.test_image_pipeline_filename(gaussian_blur, (input_img_rgb,), n, name='blur_one_stage.gaussian_blur rgb')
    ans2 = util.test_image_pipeline_filename(gaussian_blur, (input_img_gray,), n, grayscale=True, name='blur_one_stage.gaussian_blur gray')
    return util.combine_tests((ans1, ans2))

if __name__ == '__main__':
    test()
