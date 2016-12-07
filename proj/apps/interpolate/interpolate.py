import numpy as np
import sys; sys.path += ['../../compiler']
import util

def interpolate(input_img):
    levels = 10
    w = input_img.shape[0]
    h = input_img.shape[1]
    downsampled = np.empty([levels, w, h, 4])
    downx = np.empty([levels, w, h, 4])
    interpolated = np.empty([levels, w, h, 4])
    upsampled = np.empty([levels, w, h, 4])
    upsampledx = np.empty([levels, w, h, 4])
    normalize = np.empty([w, h, 3])
    
    for x in range(w):
        for y in range(h):
            downsampled[0, x, y, :] = float(input_img[x, y, 3]) * input_img[x, y, :]
    
    for l in range(1,levels):
        prev = downsampled[l-1, :, :, :]
        
        for x in range(int(w * 2 ** (-l + 1))):
            for y in range(1, int(h * 2 ** (-l))):
                downx[l, x, y, :] = (prev[x, 2 * y - 1, :] + 2.0 * prev[x, 2 * y, :] + prev[x, 2 * y + 1, :]) * 0.25
            downx[l, x, 0, :] = (3.0 * prev[x, 0, :] + prev[x, 1, :]) * 0.25
        
        for y in range(int(h * 2 ** (-l))):
            for x in range(1, int(w * 2 ** (-l))):
                downsampled[l, x, y, :] = (downx[l, 2 * x - 1, y, :] + 2.0 * downx[l, 2 * x, y, :] + downx[l, 2 * x + 1, y, :]) * 0.25
            downsampled[l, 0, y, :] = (downx[l, 0, y, :] * 3.0 + downx[l, 1, y, :]) * 0.25
    
    interpolated[levels-1, :, :, :] = downsampled[levels-1, :, :, :]
    
    for l in range(levels - 2, -1, -1):
        for x in range(int(w * 2 ** (-l - 1))):
            for y in range(int(h * 2 ** (-l)) - 1):
                upsampledx[l, x, y, :] = (interpolated[l+1, x, int(y / 2), :] + interpolated[l+1, x, int((y + 1) / 2), :]) / 2.0
            upsampledx[l, x, int(h * 2 ** (-l)) - 1, :] = interpolated[l+1, x, int(h * 2 ** (-l - 1)) - 1, :]
    
        for y in range(int(h * 2 ** (-l))):
            for x in range(int(w * 2 ** (-l)) - 1):
                upsampled[l, x, y, :] = (upsampledx[l, int(x / 2), y, :] + upsampledx[l, int((x + 1) / 2), y, :]) / 2.0
                interpolated[l, x, y, :] = downsampled[l, x, y, :] + (1.0 - downsampled[l, x, y, 3]) * upsampled[l, x, y, :]
            upsampled[l, int(w * 2 ** (-l)) - 1, y, :] = upsampledx[l, int(w * 2 ** (-l - 1)) - 1, y, :]
            interpolated[l, int(w * 2 ** (-l)) - 1, y, :] = downsampled[l, int(w * 2 ** (-l)) - 1, y, :] + (1.0 - downsampled[l, int(w * 2 ** (-l)) - 1, y, 3]) * upsampled[l, int(w * 2 ** (-l)) - 1, y, :]
    
    for x in range(w):
        for y in range(h):
            normalize[x, y, :] = np.clip(interpolated[0, x, y, 0:3] / interpolated[0, x, y, 3], 0.0, 1.0)

    return normalize

input_img = util.image_filename('rgba_small.png')

def test(n = None):
    ans = util.test_image_pipeline_filename(interpolate, (input_img,), n, name = 'interpolate')
    return util.combine_tests([ans])

if __name__ == '__main__':
    test()
    
