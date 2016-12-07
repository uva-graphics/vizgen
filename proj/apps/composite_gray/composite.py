"""composite.py: Image compositing
"""

import numpy
import sys; sys.path += ['../../compiler']
import util
try:
    import numexpr
    has_numexpr = True
except ImportError:
    has_numexpr = False             # For PyPy comparisons

def composite(background, foreground, foreground_alpha, output_img):
    """Composites foreground on background using given alpha using loops."""

    for r in range(background.shape[0]):
        for c in range(background.shape[1]):
            alpha = foreground_alpha[r,c]
            output_img[r,c] = (1-alpha)*background[r,c] + alpha*foreground[r,c]

    return output_img

def composite_numpy(background, foreground, foreground_alpha):
    """Composites foreground on background using given alpha using numpy."""

    if foreground_alpha.shape != foreground.shape:
        foreground_alpha = numpy.dstack((foreground_alpha,)*foreground.shape[2])

    return (1-foreground_alpha)*background + foreground_alpha*foreground

def composite_numexpr(background, foreground, foreground_alpha):
    """Composites foreground on background using given alpha using numpy."""

    if foreground_alpha.shape != foreground.shape:
        foreground_alpha = numpy.dstack((foreground_alpha,)*foreground.shape[2])

    return numexpr.evaluate('(1-foreground_alpha)*background + foreground_alpha*foreground')


input_img_rgb = '../images/temple_rgb.png'
input_img_gray = '../images/temple_gray.png'
input_img2_rgb = '../images/house_rgb.png'
input_img2_gray = '../images/house_gray.png'
input_matte = input_img_gray

color_args = (input_img_rgb, input_img2_rgb, input_matte)
gray_args = (input_img_gray, input_img2_gray, input_matte)

def test(n=None):
    """
    Unit test for performance optimization.
    """
    
    L = []

    #if do_rgb:
    if False:
        L.append(util.test_image_pipeline_filename(composite, color_args, n, name='composite rgb', use_output_img=True))

    #if do_gray:
    if True:
        L.append(util.test_image_pipeline_filename(composite, gray_args, n, name='composite gray', use_output_img=True))
    
    return util.combine_tests(L)

def run_test_all(n=None):
    """
    Unit tests which compare various optimized programs with the original Python program output.
    """
    assert has_numexpr
    numexpr.set_num_threads(1)

    return util.combine_tests([
    util.test_image_pipeline_filename(composite, color_args, n, name='composite rgb'),
    util.test_image_pipeline_filename(composite, gray_args, n, grayscale=True, name='composite gray'),
    util.test_image_pipeline_filename(composite_numpy, color_args, n, name='composite_numpy rgb'),
    util.test_image_pipeline_filename(composite_numpy, gray_args, n, grayscale=True, name='composite_numpy gray'),
    util.test_image_pipeline_filename(composite_numexpr, color_args, n, name='composite_numexpr (1 thread) rgb'),
    util.test_image_pipeline_filename(composite_numexpr, gray_args, n, grayscale=True, name='composite_numexpr (1 thread) gray')])

if __name__ == '__main__':
    test()
