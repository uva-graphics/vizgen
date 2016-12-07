"""blur_one_stage.py

An intentionally slow implementation of a Gaussian blur using a 3x3 kernel.

Author: Sam Prestwood
"""

import numpy as np
import sys; sys.path += ['../../compiler']
import util

#transform(TypeSpecialize(checks=False))
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

#transform(LoopRemoveConditionals())
#transform(Parallel())
    for r in range(input_img.shape[0]):
        for c in range(input_img.shape[1]):
            # center
#transform(VectorizeInnermost())
            kernel_accum = 4.0 * input_img[r, c]
            kernel_norm = 4.0

            # top left
            if r > 0 and c > 0:
#transform(VectorizeInnermost())
                kernel_accum += 1.0 * input_img[r - 1, c - 1]
                kernel_norm += 1.0

#transform(VectorizeInnermost())
            output_img[r, c] = kernel_accum / kernel_norm
    
    return output_img

input_img_rgb = util.image_filename('rgba_small.png')
input_img_gray = util.image_filename('temple_gray.png')

def test(n=None):
    """Default unit tests which compare any optimized output with the original Python output.
    """
    L = []
    L.append(util.test_image_pipeline_filename(gaussian_blur, (input_img_rgb,), n, name='blur_one_stage.gaussian_blur rgb'))
    return util.combine_tests(L)

if __name__ == '__main__':
    test()
