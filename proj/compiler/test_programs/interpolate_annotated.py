import numpy as np
#import skimage
#import skimage.io
#import skimage.color
import sys; sys.path += ['../../compiler']
import util

#transform(TypeSpecialize(checks=False))
def interpolate(input_img):
    levels = 10
    w = input_img.shape[0]
    h = input_img.shape[1]
    downsampled = np.zeros([w, h, 4, levels])
    downx = np.zeros([w, h, 4, levels])
    interpolated = np.zeros([w, h, 4, levels])
    upsampled = np.zeros([w, h, 4, levels])
    upsampledx = np.zeros([w, h, 4, levels])
    normalize = np.zeros([w, h, 3])
    
#transform(Parallel())
    for x in range(w):
        for y in range(h):
            downsampled[x, y, :, 0] = float(input_img[x, y, 3]) * input_img[x, y, :]
    
    for l in range(1,levels):
        prev = downsampled[:, :, :, l - 1]
        
#transform(Parallel())
        for x in range(int(w * 2 ** (-l + 1))):
            for y in range(1, int(h * 2 ** (-l))):
                downx[x, y, :, l] = (prev[x, 2 * y - 1, :] + 2.0 * prev[x, 2 * y, :] + prev[x, 2 * y + 1, :]) * 0.25
            downx[x, 0, :, l] = (3.0 * prev[x, 0, :] + prev[x, 1, :]) * 0.25
        
#transform(Parallel())
        for y in range(int(h * 2 ** (-l))):
            for x in range(1, int(w * 2 ** (-l))):
                downsampled[x, y, :, l] = (downx[2 * x - 1, y, :, l] + 2.0 * downx[2 * x, y, :, l] + downx[2 * x + 1, y, :, l]) * 0.25
            downsampled[0, y, :, l] = (downx[0, y, :, l] * 3.0 + downx[1, y, :, l]) * 0.25

    interpolated[:, :, :, levels - 1] = downsampled[:, :, :, levels - 1]
    
    for l in range(levels - 2, -1, -1):
#transform(Parallel())
        for x in range(int(w * 2 ** (-l - 1))):
            for y in range(int(h * 2 ** (-l)) - 1):
                upsampledx[x, y, :, l] = (interpolated[x, int(y / 2), :, l + 1] + interpolated[x, int((y + 1) / 2), :, l + 1]) / 2.0
            upsampledx[x, int(h * 2 ** (-l)) - 1, :, l] = interpolated[x, int(h * 2 ** (-l - 1)) - 1, :, l + 1]
    
#transform(Parallel())
        for y in range(int(h * 2 ** (-l))):
            for x in range(int(w * 2 ** (-l)) - 1):
                upsampled[x, y, :, l] = (upsampledx[int(x / 2), y, :, l] + upsampledx[int((x + 1) / 2), y, :, l]) / 2.0
                interpolated[x, y, :, l] = downsampled[x, y, :, l] + (1.0 - downsampled[x, y, 3, l]) * upsampled[x, y, :, l]
            upsampled[int(w * 2 ** (-l)) - 1, y, :, l] = upsampledx[int(w * 2 ** (-l - 1)) - 1, y, :, l]
            interpolated[int(w * 2 ** (-l)) - 1, y, :, l] = downsampled[int(w * 2 ** (-l)) - 1, y, :, l] + (1.0 - downsampled[int(w * 2 ** (-l)) - 1, y, 3, l]) * upsampled[int(w * 2 ** (-l)) - 1, y, :, l]
    
#transform(Parallel())
    for x in range(w):
        for y in range(h):
            normalize[x, y, :] = np.clip(interpolated[x, y, 0:3, 0] / interpolated[x, y, 3, 0], 0.0, 1.0)

    return normalize

input_img = util.image_filename('rgba_small.png')
#input = skimage.io.imread(input_img)
#input = skimage.img_as_float(input)

#output = interpolate(input)

#skimage.io.imsave('result.png', np.clip(output, 0, 1))

def test(n = None):
    ans = util.test_image_pipeline_filename(interpolate, (input_img,), n, name = 'interpolate')
    return util.combine_tests([ans])

if __name__ == '__main__':
    test()
    
