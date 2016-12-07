# Slightly simplified version of bilateral_grid which works with compiler.

import numpy as np
#import skimage
#import skimage.io
#import skimage.color
import sys; sys.path += ['../../compiler']
import util

def bilateral_grid(input_img):

    r_sigma = 0.1
    s_sigma = 8
    
    histogram = np.zeros([input_img.shape[0] / s_sigma + 1, input_img.shape[1] / s_sigma + 1, int(1.0 / r_sigma) + 1, 2])
    blurx = np.zeros(histogram.shape)
    blury = np.zeros(histogram.shape)
    blurz = np.zeros(histogram.shape)
    
    interpolated = np.zeros([input_img.shape[0], input_img.shape[1], 2])
    output_img = np.zeros(input_img.shape)
    
    w = histogram.shape[0]
    h = histogram.shape[1]
    d = histogram.shape[2]
    
    for x in range(1, input_img.shape[0] - 1):
        for y in range(1, input_img.shape[1] - 1):
        
            val = np.clip(input_img[x, y], 0, 1)
            zi = int(val * (1.0 / r_sigma) + 0.5)
            
            xi = int((x + s_sigma / 2) / s_sigma)
            yi = int((y + s_sigma / 2) / s_sigma)
            
            histogram[xi, yi, zi, 0] += val;
            histogram[xi, yi, zi, 1] += 1;
        
        #y = 0
        val = np.clip(input_img[x, 0], 0, 1)
        zi = int(val * (1.0 / r_sigma) + 0.5)
        
        xi = int((x + s_sigma / 2) / s_sigma)
        
        histogram[xi, 0, zi, 0] += val * 5.0
        histogram[xi, 0, zi, 1] += 5
        
        #y = end
        val = np.clip(input_img[x, input_img.shape[1] - 1], 0, 1)
        zi = int(val * (1.0 / r_sigma) + 0.5)
        
        histogram[xi, h - 1, zi, 0] += val * 5.0
        histogram[xi, h - 1, zi, 1] += 5
    
    for y in range(1, input_img.shape[1] - 1):
    
        #x = 0
        val = np.clip(input_img[0, y], 0, 1)
        zi = int(val * (1.0 / r_sigma) + 0.5)
    
        yi = int((y + s_sigma / 2) / s_sigma)
        
        histogram[0, yi, zi, 0] += val * 5.0
        histogram[0, yi, zi, 1] += 5
        
        #x = end
        val = np.clip(input_img[input_img.shape[0] - 1, y], 0, 1)
        zi = int(val * (1.0 / r_sigma) + 0.5)
        
        histogram[w - 1, yi, zi, 0] += val * 5.0
        histogram[w - 1, yi, zi, 1] += 5
        
    #x = 0, y = 0
    val = np.clip(input_img[0, 0], 0, 1)
    zi = int(val * (1.0 / r_sigma) + 0.5)
    
    histogram[0, 0, zi, 0] += val * 17
    histogram[0, 0, zi, 1] += 17
    
    #x = 0, y = end
    val = np.clip(input_img[0, input_img.shape[1] - 1], 0, 1)
    zi = int(val * (1.0 / r_sigma) + 0.5)
    
    histogram[0, h - 1, zi, 0] += val * 17
    histogram[0, h - 1, zi, 1] += 17
    
    #x = end, y = 0
    val = np.clip(input_img[input_img.shape[0] - 1, 0], 0, 1)
    zi = int(val * (1.0 / r_sigma) + 0.5)
    
    histogram[w - 1, 0, zi, 0] += val * 17
    histogram[w - 1, 0, zi, 1] += 17
    
    #x = end, y = end
    val = np.clip(input_img[input_img.shape[0] - 1, input_img.shape[1] - 1], 0, 1)
    zi = int(val * (1.0 / r_sigma) + 0.5)
    
    histogram[w - 1, h - 1, zi, 0] += val * 17
    histogram[w - 1, h - 1, zi, 1] += 17
    
    #blur the grid
    for z in range(2, d - 2):
        
        blurz[:, :, z, :] = histogram[:, :, z - 2, :] + 4.0 * histogram[:, :, z - 1, :] + 6.0 * histogram[:, :, z, :] + 4.0 * histogram[:, :, z + 1, :] + histogram[:, :, z + 2, :]
    
    blurz[:, :, 0, :] = 6.0 * histogram[:, :, 0, :] + 4.0 * histogram[:, :, 1, :] + histogram[:, :, 2, :]
    blurz[:, :, 1, :] = 4.0 * histogram[:, :, 0, :] + 6.0 * histogram[:, :, 1, :] + 4.0 * histogram[:, :, 2, :] + histogram[:, :, 3, :]
    blurz[:, :, d - 2, :] = histogram[:, :, d - 4, :] + 4.0 * histogram[:, :, d - 3, :] + 6.0 * histogram[:, :, d - 2, :] + 4.0 * histogram[:, :, d - 1, :]
    blurz[:, :, d - 1, :] = histogram[:, :, d - 3, :] + 4.0 * histogram[:, :, d - 2, :] + 6.0 * histogram[:, :, d - 1, :]
    
    for y in range(2, h - 2):
    
        blury[:, y, :, :] = blurz[:, y - 2, :, :] + 4.0 * blurz[:, y - 1, :, :] + 6.0 * blurz[:, y, :, :] + 4.0 * blurz[:, y + 1, :, :] + blurz[:, y + 2, :, :]
    
    blury[:, 0, :, :] = 6.0 * blurz[:, 0, :, :] + 4.0 * blurz[:, 1, :, :] + blurz[:, 2, :, :]
    blury[:, 1, :, :] = 4.0 * blurz[:, 0, :, :] + 6.0 * blurz[:, 1, :, :] + 4.0 * blurz[:, 2, :, :] + blurz[:, 3, :, :]
    blury[:, h - 2, :, :] = blurz[:, h - 4, :, :] + 4.0 * blurz[:, h - 3, :, :] + 6.0 * blurz[:, h - 2, :, :] + 4.0 * blurz[:, h - 1, :, :]
    blury[:, h - 1, :, :] = blurz[:, h - 3, :, :] + 4.0 * blurz[:, h - 2, :, :] + 6.0 * blurz[:, h - 1, :, :]
        
    for x in range(2, w - 2):
    
        blurx[x, :, :, :] = blury[x - 2, :, :, :] + 4.0 * blury[x - 1, :, :, :] + 6.0 * blury[x, :, :, :] + 4.0 * blury[x + 1, :, :, :] + blury[x + 2, :, :, :]
        
    blurx[0, :, :, :] = 6.0 * blury[0, :, :, :] + 4.0 * blury[1, :, :, :] + blury[2, :, :, :]
    blurx[1, :, :, :] = 4.0 * blury[0, :, :, :] + 6.0 * blury[1, :, :, :] + 4.0 * blury[2, :, :, :] + blury[3, :, :, :]
    blurx[w - 2, :, :, :] = blury[w - 4, :, :, :] + 4.0 * blury[w - 3, :, :, :] + 6.0 * blury[w - 2, :, :, :] + 4.0 * blury[w - 1, :, :, :]
    blurx[w - 1, :, :, :] = blury[w - 3, :, :, :] + 4.0 * blury[w - 2, :, :, :] + 6.0 * blury[w - 1, :, :, :]
    
    #trilinear sample
    for x in range(output_img.shape[0]):
        for y in range(output_img.shape[1]):

            val = np.clip(input_img[x, y], 0, 1)
            zv = val * (1.0 / r_sigma)
            zi = int(zv)
            zf = zv - zi
            xf = float(x % s_sigma) / s_sigma
            yf = float(y % s_sigma) / s_sigma
            xi = int(x / s_sigma)
            yi = int(y / s_sigma)
            
            if xi == blurx.shape[0]-1:
                xi -= 1
                xf = 1.0
            if yi == blurx.shape[1]-1:
                yi -= 1
                yf = 1.0
            if zi == blurx.shape[2]-1:
                zi -= 1
                zf = 1.0
            
            lerp1 = blurx[xi, yi, zi, :] * (1.0 - yf) + blurx[xi, yi + 1, zi, :] * yf
            lerp2 = blurx[xi + 1, yi, zi, :] * (1.0 - yf) + blurx[xi + 1, yi + 1, zi, :] * yf
            lerp3 = lerp1 * (1.0 - xf) + lerp2 * xf
            
            lerp4 = blurx[xi, yi, zi + 1, :] * (1.0 - yf) + blurx[xi, yi + 1, zi + 1, :] * yf
            lerp5 = blurx[xi + 1, yi, zi + 1, :] * (1.0 - yf) + blurx[xi + 1, yi + 1, zi + 1, :] * yf
            lerp6 = lerp4 * (1.0 - xf) + lerp5 * xf
            
            interpolated[x, y, :] = lerp3 * (1.0 - zf) + lerp6 * zf
            output_img[x, y] = interpolated[x, y, 0] / interpolated[x, y, 1]
    
    return output_img

input_img = '../images/gray.png'
#input_img = '../images/small_temple_gray.png'
#input = skimage.io.imread(input_img)
#input = skimage.img_as_float(input)

#output = bilateral_grid(input)

def test(n = None):
    ans = util.test_image_pipeline_filename(bilateral_grid, (input_img,), n, name = 'bilateral_grid')
    return util.combine_tests([ans])

if __name__ == '__main__':
    test()