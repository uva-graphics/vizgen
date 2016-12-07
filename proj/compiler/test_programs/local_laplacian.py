import numpy as np
#import skimage
#import skimage.io
#import skimage.color
import sys; sys.path += ['../../compiler']
import util

#transform(TypeSpecialize(checks=False))
def local_laplacian(input_img):

    levels = 8
    alpha = 1.0 / 7.0
    beta = 1
    J = 8
    eps = 0.01
    
    gray = 0.299 * input_img[:, :, 0] + 0.587 * input_img[:, :, 1] + 0.114 * input_img[:, :, 2]
    
    w = input_img.shape[0]
    h = input_img.shape[1]
    
    gPyramid = np.zeros([w, h, levels, J])
    lPyramid = np.zeros([w, h, levels, J])
    inGPyramid = np.zeros([w, h, J])
    outLPyramid = np.zeros([w, h, J])
    outGPyramid = np.zeros([w, h, J])
    
    output_img = np.zeros([w, h, 3])
    
    downx = np.zeros([w, h, levels])
    gdownx = np.zeros([w, h])
    upx = np.zeros([w, h, levels])
    upy = np.zeros([w, h, levels])
    gupx = np.zeros([w, h])
    gupy = np.zeros([w, h])
    
#transform(Parallel())
    for k in range(levels):
        for x in range(w):
            for y in range(h):
            
                if x == 2000 and y == 800:
                    x = 2000
                level = k * (1.0 / (levels - 1))
                idx = int(gray[x, y] * (levels - 1) * 256.0)
                idx = np.clip(idx, 0, 256 * (levels - 1))
                fx = (idx - 256.0 * k) / 256.0
                gPyramid[x, y, k, 0] = beta * (gray[x, y] - level) + level + alpha * fx * np.exp(-fx * fx / 2.0)
    
    for j in range(1, J):
    
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
        
#transform(Parallel())
        for x in range(w_j * 2):
            for y in range(1, h_j - 1):
                
                downx[x, y, :] = (gPyramid[x, 2 * y - 1, :, j - 1] + 3.0 * (gPyramid[x, 2 * y, :, j - 1] + gPyramid[x, 2 * y + 1, :, j - 1]) + gPyramid[x, 2 * y + 2, :, j - 1]) / 8.0
            
            downx[x, 0, :] = (4.0 * gPyramid[x, 0, :, j - 1] + 3.0 * gPyramid[x, 1, :, j - 1] + gPyramid[x, 2, :, j - 1]) / 8.0
            downx[x, h_j - 1, :] = (gPyramid[x, h_j * 2 - 3, :, j - 1] + 3.0 * gPyramid[x, h_j * 2 - 2, :, j - 1] + 4.0 * gPyramid[x, h_j * 2 - 1, :, j - 1]) / 8.0
        
#transform(Parallel())
        for y in range(h_j):
            for x in range(1, w_j - 1):
            
                gPyramid[x, y, :, j] = (downx[2 * x - 1, y, :] + 3.0 * (downx[2 * x, y, :] + downx[2 * x + 1, y, :]) + downx[2 * x + 2, y, :]) / 8.0
            
            gPyramid[0, y, :, j] = (4.0 * downx[0, y, :] + 3.0 * downx[1, y, :] + downx[2, y, :]) / 8.0
            gPyramid[w_j - 1, y, :, j] = (downx[w_j * 2 - 3, y, :] + 3.0 * downx[w_j * 2 - 2, y, :] + 4.0 * downx[w_j * 2 - 1, y, :]) / 8.0
    
    lPyramid[:, :, :, J - 1] = gPyramid[:, :, :, J - 1]
    
    for j in range(J - 2, -1, -1):
        
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
        
#transform(Parallel())
        for x in range(int(w_j / 2)):
            for y in range(1, h_j):
            
                upx[x, y, :] = 0.25 * gPyramid[x, int((y + 1) / 2) - 1, :, j + 1] + 0.75 * gPyramid[x, int(y / 2), :, j + 1]
            
            upx[x, 0, :] = gPyramid[x, 0, :, j + 1]
        
#transform(Parallel())
        for y in range(h_j):
            for x in range(1, w_j):
            
                upy[x, y, :] = 0.25 * upx[int((x + 1) / 2) - 1, y, :] + 0.75 * upx[int(x / 2), y, :]
                lPyramid[x, y, :, j] = gPyramid[x, y, :, j] - upy[x, y, :]
            
            upy[0, y, :] = upx[0, y, :]
            lPyramid[0, y, :, j] = gPyramid[0, y, :, j] - upy[0, y, :]
    
    inGPyramid[:, :, 0] = gray
    
    for j in range(1, J):
    
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
        
#transform(Parallel())
        for x in range(w_j * 2):
            for y in range(1, h_j - 1):
            
                gdownx[x, y] = (inGPyramid[x, 2 * y - 1, j - 1] + 3.0 * (inGPyramid[x, 2 * y, j - 1] + inGPyramid[x, 2 * y + 1, j - 1]) + inGPyramid[x, 2 * y + 2, j - 1]) / 8.0
            
            gdownx[x, 0] = (4.0 * inGPyramid[x, 0, j - 1] + 3.0 * inGPyramid[x, 1, j - 1] + inGPyramid[x, 2, j - 1]) / 8.0
            gdownx[x, h_j - 1] = (inGPyramid[x, h_j * 2 - 3, j - 1] + 3.0 * inGPyramid[x, h_j * 2 - 2, j - 1] + 4.0 * inGPyramid[x, h_j * 2 - 1, j - 1]) / 8.0
        
#transform(Parallel())
        for y in range(h_j):
            for x in range(1, w_j - 1):
            
                inGPyramid[x, y, j] = (gdownx[2 * x - 1, y] + 3.0 * (gdownx[2 * x, y] + gdownx[2 * x + 1, y]) + gdownx[2 * x + 2, y]) / 8.0
            
            inGPyramid[0, y, j] = (4.0 * gdownx[0, y] + 3.0 * gdownx[1, y] + gdownx[2, y]) / 8.0
            inGPyramid[w_j - 1, y, j] = (gdownx[w_j * 2 - 3, y] + 3.0 * gdownx[w_j * 2 - 2, y] + 4.0 * gdownx[w_j * 2 - 1, y]) / 8.0
    
    for j in range(J):
        
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
        
#transform(Parallel())
        for x in range(w_j):
            for y in range(h_j):
            
                level = inGPyramid[x, y, j] * (levels - 1.0)
                li = np.clip(int(level), 0, levels - 2)
                lf = level - li
                outLPyramid[x, y, j] = (1.0 - lf) * lPyramid[x, y, li, j] + lf * lPyramid[x, y, li + 1, j]
    
    outGPyramid[:, :, J - 1] = outLPyramid[:, :, J - 1]
    
    for j in range(J - 2, -1, -1):
    
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
        
#transform(Parallel())
        for x in range(int(w_j / 2)):
            for y in range(1, h_j):
            
                gupx[x, y] = 0.25 * outGPyramid[x, int((y + 1) / 2) - 1, j + 1] + 0.75 * outGPyramid[x, int(y / 2), j + 1]
    
            gupx[x, 0] = outGPyramid[x, 0, j + 1]
            
#transform(Parallel())
        for y in range(h_j):
            for x in range(1, w_j):
                
                gupy[x, y] = 0.25 * gupx[int((x + 1) / 2) - 1, y] + 0.75 * gupx[int(x / 2), y]
                outGPyramid[x, y, j] = gupy[x, y] + outLPyramid[x, y, j]
                    
            gupy[0, y] = gupx[0, y]
            outGPyramid[0, y, j] = gupy[0, y] + outLPyramid[0, y, j]
    
#transform(Parallel())
    for x in range(w):
        for y in range(h):
            
            output_img[x, y, :] = outGPyramid[x, y, 0] * (input_img[x, y, :] + eps) / (gray[x, y] + eps)
    
    output_img = np.clip(output_img, 0, 1)
    
    return output_img

input_img = util.image_filename('small_temple_rgb.png')
#input = skimage.io.imread(input_img)
#input = skimage.img_as_float(input)

#output = local_laplacian(input)

#skimage.io.imsave('result.png', np.clip(output, 0, 1))

def test(n = None):
    ans = util.test_image_pipeline_filename(local_laplacian, (input_img,), n, name = 'local_laplacian')
    return util.combine_tests([ans])

if __name__ == '__main__':
    test()

