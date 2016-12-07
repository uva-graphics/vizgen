import numpy as np
#import skimage
#import skimage.io
#import skimage.color
import sys; sys.path += ['../../compiler']
import util

def local_laplacian(input_img):

    levels = 8
    alpha = 1.0 / 7.0
    beta = 1
    J = 8
    eps = 0.01
    
    gray = 0.299 * input_img[:, :, 0] + 0.587 * input_img[:, :, 1] + 0.114 * input_img[:, :, 2]
    
    w = input_img.shape[0]
    h = input_img.shape[1]
    
    gPyramid = np.empty([w, h, levels, J])
    lPyramid = np.empty([w, h, levels, J])
    inGPyramid = np.empty([w, h, J])
    outLPyramid = np.empty([w, h, J])
    outGPyramid = np.empty([w, h, J])
    
    output_img = np.empty([w, h, 3])

    for k in range(levels):
        for r in range(w):
            for c in range(h):
            
                level = k * (1.0 / (levels - 1))
                idx = int(gray[r, c] * (levels - 1) * 256.0)
                idx = np.clip(int(idx), 0, 256 * (levels - 1))
                fx = (idx - 256.0 * k) / 256.0
                gPyramid[r, c, k, 0] = beta * (gray[r, c] - level) + level + alpha * fx * np.exp(-fx * fx / 2.0)
    
    inGPyramid[:, :, 0] = gray
    
    for j in range(1, J):
    
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
        
        for r in range(w_j):
            for c in range(h_j):
            
                idx_r1 = 2 * r - 1
                idx_r2 = 2 * r + 2
                idx_c1 = 2 * c - 1
                idx_c2 = 2 * c + 2
                
                if r == 0:
                    idx_r1 = 0
                if r == w_j - 1:
                    idx_r2 = w_j * 2 - 1
                if c == 0:
                    idx_c1 = 0
                if c == h_j - 1:
                    idx_c2 = h_j * 2 - 1
                
                gPyramid[r, c, :, j] = (gPyramid[idx_r1, idx_c1, :, j - 1] + 3.0 * gPyramid[idx_r1, 2 * c, :, j - 1] + 3.0 * gPyramid[idx_r1, 2 * c + 1, :, j - 1] + gPyramid[idx_r1, idx_c2, :, j - 1] + 3.0 * gPyramid[2 * r, idx_c1, :, j - 1] + 9.0 * gPyramid[2 * r, 2 * c, :, j - 1] + 9.0 * gPyramid[2 * r, 2 * c + 1, :, j - 1] + 3.0 * gPyramid[2 * r, idx_c2, :, j - 1] + 3.0 * gPyramid[2 * r + 1, idx_c1, :, j - 1] + 9.0 * gPyramid[2 * r + 1, 2 * c, :, j - 1] + 9.0 * gPyramid[2 * r + 1, 2 * c + 1, :, j - 1] + 3.0 * gPyramid[2 * r + 1, idx_c2, :, j - 1] + gPyramid[idx_r2, idx_c1, :, j - 1] + 3.0 * gPyramid[idx_r2, 2 * c, :, j - 1] + 3.0 * gPyramid[idx_r2, 2 * c + 1, :, j - 1] + gPyramid[idx_r2, idx_c2, :, j - 1]) / 64.0
                inGPyramid[r, c, j] = (inGPyramid[idx_r1, idx_c1, j - 1] + 3.0 * inGPyramid[idx_r1, 2 * c, j - 1] + 3.0 * inGPyramid[idx_r1, 2 * c + 1, j - 1] + inGPyramid[idx_r1, idx_c2, j - 1] + 3.0 * inGPyramid[2 * r, idx_c1, j - 1] + 9.0 * inGPyramid[2 * r, 2 * c, j - 1] + 9.0 * inGPyramid[2 * r, 2 * c + 1, j - 1] + 3.0 * inGPyramid[2 * r, idx_c2, j - 1] + 3.0 * inGPyramid[2 * r + 1, idx_c1, j - 1] + 9.0 * inGPyramid[2 * r + 1, 2 * c, j - 1] + 9.0 * inGPyramid[2 * r + 1, 2 * c + 1, j - 1] + 3.0 * inGPyramid[2 * r + 1, idx_c2, j - 1] + inGPyramid[idx_r2, idx_c1, j - 1] + 3.0 * inGPyramid[idx_r2, 2 * c, j - 1] + 3.0 * inGPyramid[idx_r2, 2 * c + 1, j - 1] + inGPyramid[idx_r2, idx_c2, j - 1]) / 64.0
                
    lPyramid[:, :, :, J - 1] = gPyramid[:, :, :, J - 1]
    
    j = J - 1
    
    w_j = int(w * 2 ** (-j))
    h_j = int(h * 2 ** (-j))
        
    for r in range(w_j):
        for c in range(h_j):
            
            level = inGPyramid[r, c, j] * (levels - 1.0)
            li = np.clip(int(level), 0, levels - 2)
            lf = level - li
            outLPyramid[r, c, j] = (1.0 - lf) * lPyramid[r, c, li, j] + lf * lPyramid[r, c, li + 1, j]
    
    outGPyramid[:, :, J - 1] = outLPyramid[:, :, J - 1]
    
    for j in range(J - 2, -1, -1):
        
        w_j = int(w * 2 ** (-j))
        h_j = int(h * 2 ** (-j))
            
        for r in range(w_j):
            for c in range(h_j):
                
                idx_r1 = int((r + 1) / 2) - 1
                idx_c1 = int((c + 1) / 2) - 1
                
                if r == 0:
                    idx_r1 = 0
                if c == 0:
                    idx_c1 = 0
                
                lPyramid[r, c, :, j] = gPyramid[r, c, :, j] - (gPyramid[idx_r1, idx_c1, :, j + 1] + 3.0 * gPyramid[idx_r1, int(c / 2), :, j + 1] + 3.0 * gPyramid[int(r / 2), idx_c1, :, j + 1] + 9.0 * gPyramid[int(r / 2), int(c / 2), :, j + 1]) / 16.0
                
                level = inGPyramid[r, c, j] * (levels - 1.0)
                li = np.clip(int(level), 0, levels - 2)
                lf = level - li
                outLPyramid[r, c, j] = (1.0 - lf) * lPyramid[r, c, li, j] + lf * lPyramid[r, c, li + 1, j]
                
                outGPyramid[r, c, j] = outLPyramid[r, c, j] + (outGPyramid[idx_r1, idx_c1, j + 1] + 3.0 * outGPyramid[idx_r1, int(c / 2), j + 1] + 3.0 * outGPyramid[int(r / 2), idx_c1, j + 1] + 9.0 * outGPyramid[int(r / 2), int(c / 2), j + 1]) / 16.0
    
    for r in range(w):
        for c in range(h):
            
            output_img[r, c, :] = outGPyramid[r, c, 0] * (input_img[r, c, :] + eps) / (gray[r, c] + eps)
    
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