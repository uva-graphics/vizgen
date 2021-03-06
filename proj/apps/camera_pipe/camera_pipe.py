import numpy as np
import sys; sys.path += ['../../compiler']
import util

def makeLUT(contrast, blackLevel, gamma, lut):
    minRaw = 0 + blackLevel
    maxRaw = 1023
    
    for i in range(minRaw + 1):
        lut[i] = 0
    
    invRange = 1.0 / (maxRaw - minRaw)
    b = 2 - 2.0 ** (contrast / 100.0)
    a = 2 - 2 * b
    
    for i in range(minRaw + 1, maxRaw + 1):
        y = (i - minRaw) * invRange
        y = y ** (1.0 / gamma)
        
        if y > 0.5:
            y = 1 - y
            y = a * y * y + b * y
            y = 1 - y
        else:
            y = a * y * y + b * y
            
        y = np.floor(y * 255 + 0.5)
        
        if y < 0:
            y = 0
        if y > 255:
            y = 255
        
        lut[i] = y
        
    for i in range(maxRaw + 1, 1024):
        lut[i] = 255
    
    return lut    

def makeColorMatrix(colorMatrix, colorTemp):
    alpha = (1.0 / colorTemp - 1.0 / 3200.0) / (1.0 / 7000.0 - 1.0 / 3200.0)
    
    colorMatrix[0] = alpha*1.6697 + (1-alpha)*2.2997;
    colorMatrix[1] = alpha*-0.2693 + (1-alpha)*-0.4478;
    colorMatrix[2] = alpha*-0.4004 + (1-alpha)*0.1706;
    colorMatrix[3] = alpha*-42.4346 + (1-alpha)*-39.0923;

    colorMatrix[4] = alpha*-0.3576 + (1-alpha)*-0.3826;
    colorMatrix[5] = alpha*1.0615 + (1-alpha)*1.5906;
    colorMatrix[6] = alpha*1.5949 + (1-alpha)*-0.2080;
    colorMatrix[7] = alpha*-37.1158 + (1-alpha)*-25.4311;

    colorMatrix[8] = alpha*-0.2175 + (1-alpha)*-0.0888;
    colorMatrix[9] = alpha*-1.8751 + (1-alpha)*-0.7344;
    colorMatrix[10]= alpha*6.9640 + (1-alpha)*2.2832;
    colorMatrix[11]= alpha*-26.6970 + (1-alpha)*-20.0826;
    
    return colorMatrix
    
def demosaic(input, out, colorTemp, contrast, denoise, blackLevel, gamma):

    BLOCK_WIDTH = 40
    BLOCK_HEIGHT = 24
    G = 0
    GR = 0
    R = 1
    B = 2
    GB = 3
    
    rawWidth = input.shape[1]
    rawHeight = input.shape[0]
    outWidth = rawWidth - 32
    outHeight = rawHeight - 48
    outWidth = min(outWidth, out.shape[1])
    outHeight = min(outHeight, out.shape[0])
    outWidth = outWidth // BLOCK_WIDTH
    outWidth *= BLOCK_WIDTH
    outHeight = outHeight // BLOCK_HEIGHT
    outHeight *= BLOCK_HEIGHT
    
    WIDTH = outWidth
    HEIGHT = outHeight
    
    lut = np.zeros(1024)
    lut = makeLUT(contrast, blackLevel, gamma, lut)
    
    colorMatrix = np.zeros(12)
    colorMatrix = makeColorMatrix(colorMatrix, colorTemp)
    
    for by in range(0, outHeight, BLOCK_HEIGHT):
        for bx in range(0, outWidth, BLOCK_WIDTH):
        
            inBlock = np.zeros([4, BLOCK_HEIGHT // 2 + 4, BLOCK_WIDTH // 2 + 4])
            
            for y in range(0, BLOCK_HEIGHT // 2 + 4):
                for x in range(0, BLOCK_WIDTH // 2 + 4):
                
                    inBlock[GR, y, x] = input[by + 2 * y, bx + 2 * x]
                    inBlock[R, y, x] = input[by + 2 * y, bx + 2 * x + 1]
                    inBlock[B, y, x] = input[by + 2 * y + 1, bx + 2 * x]
                    inBlock[GB, y, x] = input[by + 2 * y + 1, bx + 2 * x + 1]
                    
            linear = np.zeros([3, 4, BLOCK_HEIGHT // 2 + 4, BLOCK_WIDTH // 2 + 4])
            
            if denoise:
                for y in range(1, BLOCK_HEIGHT // 2 + 3):
                    for x in range(1, BLOCK_WIDTH // 2 + 3):
                        linear[G, GR, y, x] = min(inBlock[GR, y, x], max(max(inBlock[GR, y - 1, x], inBlock[GR, y + 1, x]), max(inBlock[GR, y, x + 1], inBlock[GR, y, x - 1])))
                        linear[R, R, y, x] = min(inBlock[R, y, x], max(max(inBlock[R, y - 1, x], inBlock[R, y + 1, x]), max(inBlock[R, y, x + 1], inBlock[R, y, x - 1])))
                        linear[B, B, y, x] = min(inBlock[B, y, x], max(max(inBlock[B, y - 1, x], inBlock[B, y + 1, x]), max(inBlock[B, y, x + 1], inBlock[B, y, x - 1])))
                        linear[G, GB, y, x] = min(inBlock[GB, y, x], max(max(inBlock[GB, y - 1, x], inBlock[GB, y + 1, x]), max(inBlock[GB, y, x + 1], inBlock[GB, y, x - 1])))
            else:
                for y in range(1, BLOCK_HEIGHT // 2 + 3):
                    for x in range(1, BLOCK_WIDTH // 2 + 3):
                        linear[G, GR, y, x] = inBlock[GR, y, x]
                        linear[R, R, y, x] = inBlock[R, y, x]
                        linear[B, B, y, x] = inBlock[B, y, x]
                        linear[G, GB, y, x] = inBlock[GB, y, x]
            
            for y in range(1, BLOCK_HEIGHT // 2 + 3):
                for x in range(1, BLOCK_WIDTH // 2 + 3):
                    gv_r = (linear[G, GB, y - 1, x] + linear[G, GB, y, x]) / 2.0
                    gvd_r = abs(linear[G, GB, y - 1, x] - linear[G, GB, y, x])
                    gh_r = (linear[G, GR, y, x] + linear[G, GR, y, x + 1]) / 2.0
                    ghd_r = abs(linear[G, GR, y, x] - linear[G, GR, y, x + 1]) / 2.0
                    
                    if ghd_r < gvd_r:
                        linear[G, R, y, x] = gh_r
                    else:
                        linear[G, R, y, x] = gv_r
                        
                    gv_b = (linear[G, GR, y + 1, x] + linear[G, GR, y, x]) / 2.0
                    gvd_b = abs(linear[G, GR, y + 1, x] - linear[G, GR, y, x])
                    gh_b = (linear[G, GB, y, x] + linear[G, GB, y, x - 1]) / 2.0
                    ghd_b = abs(linear[G, GB, y, x] - linear[G, GB, y, x - 1])
                    
                    if ghd_b < gvd_b:
                        linear[G, B, y, x] = gh_b
                    else:
                        linear[G, B, y, x] = gv_b
                        
            for y in range(1, BLOCK_HEIGHT // 2 + 3):
                for x in range(1, BLOCK_WIDTH // 2 + 3):
                    linear[R, GR, y, x] = ((linear[R, R, y, x - 1] + linear[R, R, y, x]) / 2.0 + linear[G, GR, y, x] - (linear[G, R, y, x - 1] + linear[G, R, y, x]) / 2.0)
                    linear[B, GR, y, x] = ((linear[B, B, y - 1, x] + linear[B, B, y, x]) / 2.0 + linear[G, GR, y, x] - (linear[G, B, y - 1, x] + linear[G, B, y, x]) / 2.0)
                    linear[R, GB, y, x] = ((linear[R, R, y, x] + linear[R, R, y + 1, x]) / 2.0 + linear[G, GB, y, x] - (linear[G, R, y, x] + linear[G, R, y + 1, x]) / 2.0)
                    linear[B, GB, y, x] = ((linear[B, B, y, x] + linear[B, B, y, x + 1]) / 2.0 + linear[G, GB, y, x] - (linear[G, B, y, x] + linear[G, B, y, x + 1]) / 2.0)
            
            for y in range(1, BLOCK_HEIGHT // 2 + 3):
                for x in range(1, BLOCK_WIDTH // 2 + 3):
                    rp_b = ((linear[R, R, y + 1, x - 1] + linear[R, R, y, x]) / 2.0 + linear[G, B, y, x] - (linear[G, R, y + 1, x - 1] + linear[G, R, y, x]) / 2.0)
                    rpd_b = abs(linear[R, R, y + 1, x - 1] - linear[R, R, y, x])
                    rn_b = ((linear[R, R, y, x - 1] + linear[R, R, y + 1, x]) / 2.0 + linear[G, B, y, x] - (linear[G, R, y, x - 1] + linear[G, R, y + 1, x]) / 2.0)
                    rnd_b = abs(linear[R, R, y, x - 1] - linear[R, R, y + 1, x])
                    
                    if rpd_b < rnd_b:
                        linear[R, B, y, x] = rp_b
                    else:
                        linear[R, B, y, x] = rn_b
                    
                    bp_r = ((linear[B, B, y - 1, x + 1] + linear[B, B, y, x]) / 2.0 + linear[G, R, y, x] - (linear[G, B, y - 1, x + 1] + linear[G, B, y, x]) / 2.0)
                    bpd_r = abs(linear[B, B, y - 1, x + 1] - linear[B, B, y, x])
                    bn_r = ((linear[B, B, y, x + 1] + linear[B, B, y - 1, x]) / 2.0 + linear[G, R, y, x] - (linear[G, B, y, x + 1] + linear[G, B, y - 1, x]) / 2.0)
                    bnd_r = abs(linear[B, B, y, x + 1] - linear[B, B, y - 1, x])
                    
                    if bpd_r < bnd_r:
                        linear[B, R, y, x] = bp_r
                    else:
                        linear[B, R, y, x] = bn_r
                        
            for y in range(2, BLOCK_HEIGHT // 2 + 2):
                for x in range(2, BLOCK_WIDTH // 2 + 2):
                    r = colorMatrix[0] * linear[R, GR, y, x] + colorMatrix[1] * linear[G, GR, y, x] + colorMatrix[2] * linear[B, GR, y, x] + colorMatrix[3]
                    g = colorMatrix[4] * linear[R, GR, y, x] + colorMatrix[5] * linear[G, GR, y, x] + colorMatrix[6] * linear[B, GR, y, x] + colorMatrix[7]
                    b = colorMatrix[8] * linear[R, GR, y, x] + colorMatrix[9] * linear[G, GR, y, x] + colorMatrix[10] * linear[B, GR, y, x] + colorMatrix[11]
                    
                    r = np.clip(r, 0, 1023)
                    ri = int(r + 0.5)
                    g = np.clip(g, 0, 1023)
                    gi = int(g + 0.5)
                    b = np.clip(b, 0, 1023)
                    bi = int(b + 0.5)
                    
                    out[by + (y - 2) * 2, bx + (x - 2) * 2, 0] = lut[ri]
                    out[by + (y - 2) * 2, bx + (x - 2) * 2, 1] = lut[gi]
                    out[by + (y - 2) * 2, bx + (x - 2) * 2, 2] = lut[bi]
                    
                    r = colorMatrix[0] * linear[R, R, y, x] + colorMatrix[1] * linear[G, R, y, x] + colorMatrix[2] * linear[B, R, y, x] + colorMatrix[3]
                    g = colorMatrix[4] * linear[R, R, y, x] + colorMatrix[5] * linear[G, R, y, x] + colorMatrix[6] * linear[B, R, y, x] + colorMatrix[7]
                    b = colorMatrix[8] * linear[R, R, y, x] + colorMatrix[9] * linear[G, R, y, x] + colorMatrix[10] * linear[B, R, y, x] + colorMatrix[11]
                    
                    r = np.clip(r, 0, 1023)
                    ri = int(r + 0.5)
                    g = np.clip(g, 0, 1023)
                    gi = int(g + 0.5)
                    b = np.clip(b, 0, 1023)
                    bi = int(b + 0.5)
                    
                    out[by + (y - 2) * 2, bx + (x - 2) * 2 + 1, 0] =lut[ri] 
                    out[by + (y - 2) * 2, bx + (x - 2) * 2 + 1, 1] =lut[gi] 
                    out[by + (y - 2) * 2, bx + (x - 2) * 2 + 1, 2] =lut[bi] 
                    
                    r = colorMatrix[0] * linear[R, B, y, x] + colorMatrix[1] * linear[G, B, y, x] + colorMatrix[2] * linear[B, B, y, x] + colorMatrix[3]
                    g = colorMatrix[4] * linear[R, B, y, x] + colorMatrix[5] * linear[G, B, y, x] + colorMatrix[6] * linear[B, B, y, x] + colorMatrix[7]
                    b = colorMatrix[8] * linear[R, B, y, x] + colorMatrix[9] * linear[G, B, y, x] + colorMatrix[10] * linear[B, B, y, x] + colorMatrix[11]
                    
                    r = np.clip(r, 0, 1023)
                    ri = int(r + 0.5)
                    g = np.clip(g, 0, 1023)
                    gi = int(g + 0.5)
                    b = np.clip(b, 0, 1023)
                    bi = int(b + 0.5)
                    
                    out[by + (y - 2) * 2 + 1, bx + (x - 2) * 2, 0] = lut[ri]
                    out[by + (y - 2) * 2 + 1, bx + (x - 2) * 2, 1] = lut[gi]
                    out[by + (y - 2) * 2 + 1, bx + (x - 2) * 2, 2] = lut[bi]
                    
                    r = colorMatrix[0] * linear[R, GB, y, x] + colorMatrix[1] * linear[G, GB, y, x] + colorMatrix[2] * linear[B, GB, y, x] + colorMatrix[3]
                    g = colorMatrix[4] * linear[R, GB, y, x] + colorMatrix[5] * linear[G, GB, y, x] + colorMatrix[6] * linear[B, GB, y, x] + colorMatrix[7]
                    b = colorMatrix[8] * linear[R, GB, y, x] + colorMatrix[9] * linear[G, GB, y, x] + colorMatrix[10] * linear[B, GB, y, x] + colorMatrix[11]
                    
                    r = np.clip(r, 0, 1023)
                    ri = int(r + 0.5)
                    g = np.clip(g, 0, 1023)
                    gi = int(g + 0.5)
                    b = np.clip(b, 0, 1023)
                    bi = int(b + 0.5)
                    
                    out[by + (y - 2) * 2 + 1, bx + (x - 2) * 2 + 1, 0] = lut[ri]
                    out[by + (y - 2) * 2 + 1, bx + (x - 2) * 2 + 1, 1] = lut[gi]
                    out[by + (y - 2) * 2 + 1, bx + (x - 2) * 2 + 1, 2] = lut[bi]
    
    return out

def camera_pipe(input_img):
    input_img = input_img * 256.0 * 256.0
    output_img = np.zeros([((input_img.shape[0] - 24) // 32) * 32, ((input_img.shape[1] - 32) // 32) * 32, 3])
    output_img = demosaic(input_img, output_img, 3700, 50, True, 25, 2.0)
    output_img /= 256.0
    return output_img
    
input_img = util.image_filename('bayer_small.png')

def test(n = None):
    ans = util.test_image_pipeline_filename(camera_pipe, (input_img,), n, name = 'camera_pipe_fcam')
    return util.combine_tests([ans])

if __name__ == '__main__':
    test()