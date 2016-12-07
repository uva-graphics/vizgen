import numpy as np
import math
import random
import skimage.io

import sys; sys.path += ['../../compiler']
import util

def draw_line(x0, y0, x1, y1, input_img, color):

    #x0, y0, x1, y1 can be double
    #p0 = (x0, y0), p1 = (x1, y1) are start and ending point of a line

    w = input_img.shape[0]
    h = input_img.shape[1]
    
    isvalid_slope = abs(y1 - y0) < abs(x1 - x0)
    
    if not isvalid_slope:
        
        #switch position of x, y
        temp = x0
        x0 = y0
        y0 = temp
        temp = x1
        x1 = y1
        y1 = temp
    
    isvalid_x = x1 > x0
    
    if not isvalid_x:
        
        #switch position of p1, p2
        temp = x0
        x0 = x1
        x1 = temp
        temp = y0
        y0 = y1
        y1 = temp
        
    dx = x1 - x0
    dy = abs(y1 - y0)
    err = dx / 2.0
    
    if y0 < y1:
        y_sign = 1
    else:
        y_sign = -1
    
    y = int(y0)
   
    count = 0
    
    for x in range(int(x0), int(x1) + 1):
    
        if isvalid_slope:
            if x > -1 and y > -1 and x < w and y < h:
                input_img[x, y, 0:3] = color
        else:
            if x > -1 and y > -1 and x < h and y < w:
                input_img[y, x, 0:3] = color
        count += 1
        err -= dy
        if err < 0:
            y += y_sign
            err += dx

def draw_arrow(x0, y0, x1, y1, input_img, color):

    draw_line(x0, y0, x1, y1, input_img, color)
    
    dx = x1 - x0
    dy = y1 - y0
    
    #endpoint for one edge of arrow
    x2 = x0 + 0.75 * dx + 0.25 * (3 ** -0.5) * dy
    y2 = y0 + 0.75 * dy - 0.25 * (3 ** -0.5) * dx
    
    x3 = x0 + 0.75 * dx - 0.25 * (3 ** -0.5) * dy
    y3 = y0 + 0.75 * dy + 0.25 * (3 ** -0.5) * dx
    
    draw_line(x2, y2, x1, y1, input_img, color)
    draw_line(x3, y3, x1, y1, input_img, color)

def find_color_value(dist, r):

    #dist is a 2D vector with magnitude smaller than 1, r is its manitude 
    #representing a position inside the color wheel
    
    angle = np.arctan2(dist[0], dist[1])
    color = np.zeros(3)
    
    if angle >= 0 and angle <= 2.0 * np.pi / 3.0:
        
        scale = angle * 3.0 / (2.0 * np.pi)
        color[0] = 1.0 - scale
        color[1] = scale
        
    if angle > 2.0 * np.pi / 3.0:
    
        scale = (angle - 2.0 * np.pi / 3.0) * 3.0 / (2.0 * np.pi)
        color[1] = 1.0 - scale
        color[2] = scale
    
    if angle < -2.0 * np.pi / 3.0:
    
        real_angle = angle + 2.0 * np.pi
        scale = (real_angle - 2.0 * np.pi / 3.0) * 3.0 / (2.0 * np.pi)
        color[1] = 1.0 - scale
        color[2] = scale
        
    if angle < 0 and angle >= -2.0 * np.pi / 3.0:
    
        real_angle = angle + 2.0 * np.pi
        scale = (real_angle - 4.0 * np.pi / 3.0) * 3.0 / (2.0 * np.pi)
        color[2] = 1.0 - scale
        color[0] = scale

    color *= r ** 0.25
        
    return color

def optical_flow(a, b, output_img):
    patch_w = 16
    max_offset = 11
    nn_iters = 2

    h = a.shape[0]
    w = a.shape[1]
    ew = w-patch_w+1
    eh = h-patch_w+1
    assert b.shape[0] == a.shape[0] and b.shape[1] == a.shape[1]
    
    nnf = np.empty([eh, ew, 3])
    
    rs_iters = int(math.ceil(math.log2(max_offset*2)))
    
    # Initialization
    for ay in range(eh):
        for ax in range(ew):
            xmin = max(ax - max_offset, 0)
            xmax = min(ax + max_offset + 1, ew)
            ymin = max(ay - max_offset, 0)
            ymax = min(ay + max_offset + 1, eh)
            
            seed = ay << 16 | ax
            bx = util.randrange(seed, xmin, xmax)
            by = util.randrange(seed+1, ymin, ymax)
            
            nnf[ay, ax, 0] = bx
            nnf[ay, ax, 1] = by

            dcurrent = 0.0
            for dy in range(patch_w):
                ay_current = ay + dy
                by_current = by + dy
                for dx in range(patch_w):
                    ax_current = ax + dx
                    bx_current = bx + dx
                    for c in range(a.shape[2]):
                        acolor = a[ay_current, ax_current, c]
                        bcolor = b[by_current, bx_current, c]
                        dcurrent += (bcolor - acolor) ** 2

            nnf[ay, ax, 2] = dcurrent

    
    # Iterations
    for nn_iter in range(nn_iters):
        ystart = 0
        yend   = eh
        ystep  = 1
        xstart = 0
        xend   = ew
        xstep  = 1
        if nn_iter % 2 == 1:
            ystart = eh-1
            yend   = -1
            ystep  = -1
            xstart = ew-1
            xend   = -1
            xstep  = -1
        for ay in range(ystart, yend, ystep):
            for ax in range(xstart, xend, xstep):
                bx_best = int(nnf[ay, ax, 0])
                by_best = int(nnf[ay, ax, 1])
                dist_best = nnf[ay, ax, 2]
                
                # Propagation (x), incremental algorithm that takes O(patch_w) time
                ax_p = ax-xstep
                if ax_p >= 0 and ax_p < ew:
                    bx = int(nnf[ay, ax_p, 0])+xstep
                    by = int(nnf[ay, ax_p, 1])
                    if bx >= 0 and bx < ew and abs(ax-bx) <= max_offset and abs(ay-by) <= max_offset:
                        dist = nnf[(ay, ax_p, 2)]
                        delta_add = patch_w-1
                        delta_remove = -1
                        if xstep < 0:
                            delta_add = 0
                            delta_remove = patch_w
                        for dy in range(patch_w):
                            ay_current = (ay + dy)
                            by_current = (by + dy)

                            ax_current = (ax + delta_remove)
                            bx_current = (bx + delta_remove)
                            for c in range(3):
                                acolor = a[(ay_current, ax_current, c)]
                                bcolor = b[(by_current, bx_current, c)]
                                dist -= (bcolor - acolor)**2

                            ax_current = (ax + delta_add)
                            bx_current = (bx + delta_add)
                            for c in range(3):
                                acolor = a[(ay_current, ax_current, c)]
                                bcolor = b[(by_current, bx_current, c)]
                                dist += (bcolor - acolor)**2

                        if dist < dist_best:
                            bx_best = bx
                            by_best = by
                            dist_best = dist
                
                # Propagation (y), incremental algorithm that takes O(patch_w) time
                ay_p = ay-ystep
                if ay_p >= 0 and ay_p < eh:
                    bx = int(nnf[ay_p, ax, 0])
                    by = int(nnf[ay_p, ax, 1])+ystep
                    if by >= 0 and by < eh and abs(ax-bx) <= max_offset and abs(ay-by) <= max_offset:
                        dist = nnf[(ay_p, ax, 2)]
                        delta_add = patch_w-1
                        delta_remove = -1
                        if ystep < 0:
                            delta_add = 0
                            delta_remove = patch_w
                        ay_current = (ay + delta_remove)
                        by_current = (by + delta_remove)
                        for dx in range(patch_w):
                            ax_current = (ax + dx)
                            bx_current = (bx + dx)

                            for c in range(3):
                                acolor = a[(ay_current, ax_current, c)]
                                bcolor = b[(by_current, bx_current, c)]
                                dist -= (bcolor - acolor)**2

                        ay_current = (ay + delta_add)
                        by_current = (by + delta_add)
                        for dx in range(patch_w):
                            ax_current = (ax + dx)
                            bx_current = (bx + dx)
                            for c in range(3):
                                acolor = a[(ay_current, ax_current, c)]
                                bcolor = b[(by_current, bx_current, c)]
                                dist += (bcolor - acolor)**2

                        if dist < dist_best:
                            bx_best = bx
                            by_best = by
                            dist_best = dist

                # Random search
                rs_max = max_offset * 2
                for rs_iter in range(rs_iters):
                    xmin = max(max(bx_best - rs_max,     ax - max_offset), 0)
                    xmax = min(min(bx_best + rs_max + 1, ax + max_offset + 1), ew)
                    ymin = max(max(by_best - rs_max,     ay - max_offset), 0)
                    ymax = min(min(by_best + rs_max + 1, ay + max_offset + 1), eh)
                    
                    seed = ((ay << 16) | ax)
                    bx = util.randrange(seed, xmin, xmax)
                    by = util.randrange(seed+1, ymin, ymax)
                    
                    dist = 0.0
                    for dy in range(patch_w):
                        ay_current = ay + dy
                        by_current = by + dy
                        for dx in range(patch_w):
                            ax_current = ax + dx
                            bx_current = bx + dx
                            for c in range(a.shape[2]):
                                acolor = a[ay_current, ax_current, c]
                                bcolor = b[by_current, bx_current, c]
                                dist += (bcolor - acolor) ** 2
                            if dist > dist_best:
                                break
                        if dist > dist_best:
                            break

                    if dist < dist_best:
                        bx_best = bx
                        by_best = by
                        dist_best = dist

                nnf[ay,ax,0] = bx_best
                nnf[ay,ax,1] = by_best
                nnf[ay,ax,2] = dist_best
                
    #for y in range(eh):
    #    for x in range(ew):
    #        nnf[y,x,0] -= x
    #        nnf[y,x,1] -= y
    #        nnf[y,x,2] = 0.0
    #
    #return nnf

    for ay in range(max_offset, eh - max_offset, 10):
        for ax in range(max_offset, ew - max_offset, 10):
            u = nnf[ay,ax,0]-ax
            v = nnf[ay,ax,1]-ay
            scale = ((u ** 2 + v ** 2) ** 0.5) / max_offset
            dist_vis = np.array([u/max_offset, v/max_offset])
            color = find_color_value(dist_vis, scale)
            draw_arrow(float(ay), float(ax), float(ay + v), float(ax + u), output_img, color)

#def test(n = None, input_img1=util.image_filename('opt_flow1_smaller.png'), input_img2=util.image_filename('opt_flow2_smaller.png')):
def test(n = None, input_img1=util.image_filename('opt_flow1_small.png'), input_img2=util.image_filename('opt_flow2_small.png')):
#    util.is_initial_run = True
#    ans = util.test_image_pipeline_filename(optical_flow, (input_img1, input_img2,), n, grayscale = False, name = 'optical_flow', output_gain=1.0/22.0, output_bias=0.5)
    ans = util.test_image_pipeline_filename(optical_flow, (input_img1, input_img2,), n, grayscale = False, name = 'optical_flow', use_output_img=True)
    return util.combine_tests([ans])

def main():
    I=test()['output']
    skimage.io.imsave('output.png', I)
    print('Wrote to output.png')

if __name__ == '__main__':
    main()
