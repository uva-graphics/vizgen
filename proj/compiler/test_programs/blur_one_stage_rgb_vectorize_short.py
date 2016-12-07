#transform(ArrayStorage(True,True))     # applied to original line 1
import numpy
'blur_one_stage.py\n\nAn intentionally slow implementation of a Gaussian blur using a 3x3 kernel.\n\nAuthor: Sam Prestwood\n'
import numpy as np
import sys
sys.path += ['../../compiler']
import util

#transform(TypeSpecialize(({'_nonvar_range(input_img.shape[0])':('int',),'_nonvar_range(input_img.shape[1])':('int',),'_nonvar_return':('numpy.ndarray[numpy.float32_t, ndim=3](shape=(405, 640, 4),shape_list=[])',),'_return_value':'numpy.ndarray[numpy.float32_t, ndim=3](shape=(405, 640, 4),shape_list=[])','c':'int','input_img':'numpy.ndarray[numpy.float32_t, ndim=3](shape=(405, 640, 4),shape_list=[])','kernel_accum':'numpy.ndarray[numpy.float32_t, ndim=1](shape=(4,),shape_list=[])','kernel_norm':'double','output_img':'numpy.ndarray[numpy.float32_t, ndim=3](shape=(405, 640, 4),shape_list=[])','r':'int'},),False))     # applied to original line 8
def gaussian_blur(input_img, output_img):
    'Blurs the image in matrix input_img and writes the values to output_img.\n\n    This uses a 3x3 Gaussian kernel to convolve with the image matrix.\n\n                1/16 2/16 1/16\n    Kernel =    2/16 4/16 2/16\n                1/16 2/16 1/16\n\n    For dealing with convolving along the edges of the image, we renormalize the\n    kernel based on which coordinates from the kernel are in-bounds.\n\n    Args:\n        input_img, np.ndarray, reference to input image to blur\n        output_img, np.ndarray, reference to array to save output image in\n    '
#transform(LoopRemoveConditionals())     # applied to original line 10
    for r in range(input_img.shape[0]):
        for c in range(input_img.shape[1]):
#transform(VectorizeInnermost())     # applied to original line 12
            kernel_accum = (4.0 * input_img[(r, c)])
            kernel_norm = 4.0
            if ((r > 0) and (c > 0)):
#transform(VectorizeInnermost())     # applied to original line 15
                kernel_accum += (1.0 * input_img[((r - 1), (c - 1))])
                kernel_norm += 1.0
#transform(VectorizeInnermost())     # applied to original line 38
            output_img[(r, c)] = (kernel_accum / kernel_norm)
    return output_img
input_img_rgb = util.image_filename('temple_rgb.png')
input_img_gray = util.image_filename('temple_gray.png')

def test(n=None):
    'Default unit tests which compare any optimized output with the original Python output.\n    '
    L = []
    if True:
        L.append(util.test_image_pipeline_filename(gaussian_blur, (input_img_rgb,), n, name='blur_one_stage.gaussian_blur rgb', use_output_img=True))
    if False:
        L.append(util.test_image_pipeline_filename(gaussian_blur, (input_img_gray,), n, grayscale=True, name='blur_one_stage.gaussian_blur gray', use_output_img=True))
    return util.combine_tests(L)
if (__name__ == '__main__'):
    test()