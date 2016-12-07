
import numpy as np
import sys; sys.path += ['../../compiler']
import util

def two_stage_blur(input_img):

    temp_img = np.zeros(input_img.shape, 'float32')
    output_img = np.zeros(input_img.shape, 'float32')
    
    three = np.float32(3.0)
    v_vec = np.array([0.0, 0.0], 'float32')             # Test np.array() constructor also.
    
    #first stage blur
    for r in range(input_img.shape[0]-8):
        for c in range(input_img.shape[1]):
            
            temp_img[r, c] = (input_img[r, c] + input_img[r + 1, c] + input_img[r + 2, c]) / three
    
    #second stage blur
    for r in range(input_img.shape[0]-8):
        for c in range(input_img.shape[1]-2):
        
            output_img[r, c] = (temp_img[r, c] + temp_img[r, c + 1] + temp_img[r, c + 2]) / 3.0
    
    return output_img

input_img_rgb = util.image_filename('small_temple_rgb.png')
input_img_gray = util.image_filename('small_temple_gray.png')

def test(n=None):
    """Default unit tests which compare any optimized output with the original Python output.
    """
    return util.test_image_pipeline_filename(two_stage_blur, (input_img_gray,), n, grayscale = True, name = 'blur_two_stage.gaussian_blur gray', dtype='float32')
    
if __name__ == '__main__':
    test()
