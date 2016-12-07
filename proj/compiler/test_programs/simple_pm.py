import numpy as np

import sys; sys.path += ['../../compiler']
import util

def optical_flow(a, b, output_img):
    patch_w = 16
    dist_best = 1.0
    dist = patch_dist(a, b, 0, 0, 0, 0, patch_w, dist_best)

def patch_dist(a, b, ax, ay, bx, by, patch_w, dmax):
    ans = 0.0
    for dy in range(patch_w):
        ay_current = ay + dy
        by_current = by + dy
        for dx in range(patch_w):
            ax_current = ax + dx
            bx_current = bx + dx
            for c in range(a.shape[2]):
                acolor = a[ay_current, ax_current, c]
                bcolor = b[by_current, bx_current, c]
                ans += (bcolor - acolor) ** 2
            if ans > dmax:
                return ans
    return ans

def test(n = None, input_img1=util.image_filename('opt_flow1_smaller.png'), input_img2=util.image_filename('opt_flow2_smaller.png')):
    return util.test_image_pipeline_filename(optical_flow, (input_img1, input_img2,), n, grayscale = False, name = 'optical_flow', use_output_img=True)

if __name__ == '__main__':
    test()

