#import matplotlib.pyplot as plot
import numpy as np
#import skimage
#import skimage.io
#import skimage.color
import sys; sys.path += ['../../compiler']
import util

def find_corners(input_img):

    k = 0.05
    threshold = 0.5

    Ix = np.zeros(input_img.shape)
    Iy = np.zeros(input_img.shape)
    Ix2 = np.zeros(input_img.shape)
    Iy2 = np.zeros(input_img.shape)
    Ixy = np.zeros(input_img.shape)
    R = np.zeros(input_img.shape)

    w = input_img.shape[0]
    h = input_img.shape[1]

    #use sobel operator to calculate Ix, Iy, Ix2, Iy2, Ixy
    for r in range(w):
        for c in range(h):

            idx_r1 = r - 1
            idx_r2 = r + 1
            idx_c1 = c - 1
            idx_c2 = c + 1

            if r == 0:
                idx_r1 = 0
            if r == w - 1:
                idx_r2 = w - 1
            if c == 0:
                idx_c1 = 0
            if c == h - 1:
                idx_c2 = h - 1

            Ix[r, c] =    -input_img[idx_r1, idx_c1] - 2 * input_img[idx_r1, c     ] - input_img[idx_r1, idx_c2] + input_img[idx_r2, idx_c1] +  2 * input_img[idx_r2, c     ] + input_img[idx_r2, idx_c2]

            Iy[r, c] =    -input_img[idx_r1, idx_c1] - 2 * input_img[r     , idx_c1] - input_img[idx_r2, idx_c1] + input_img[idx_r1, idx_c2] + 2 * input_img[r     , idx_c2] + input_img[idx_r2, idx_c2]

            Ix2[r, c] = Ix[r, c] ** 2
            Iy2[r, c] = Iy[r, c] ** 2
            Ixy[r, c] = Ix[r, c] * Iy[r, c]

    # calculate the response of detector R using a Gaussian kernel described in
    # blur_one_stage:

    count = 0

    for r in range(w):
        for c in range(h):

            idx_r1 = r - 1
            idx_r2 = r + 1
            idx_c1 = c - 1
            idx_c2 = c + 1

            if r == 0:
                idx_r1 = 0
            if r == w - 1:
                idx_r2 = w - 1
            if c == 0:
                idx_c1 = 0
            if c == h - 1:
                idx_c2 = h - 1

            Sx2 =       Ix2[idx_r1, idx_c1] + 2.0 * Ix2[idx_r1, c     ] + Ix2[idx_r1, idx_c2] + 2.0 * Ix2[r     , idx_c1] + 4.0 * Ix2[r     , c     ] + 2.0 * Ix2[r     , idx_c2] + Ix2[idx_r2, idx_c1] + 2.0 * Ix2[idx_r2, c     ] + Ix2[idx_r2, idx_c2]

            Sy2 =       Iy2[idx_r1, idx_c1] + 2.0 * Iy2[idx_r1, c     ] + Iy2[idx_r1, idx_c2] + 2.0 * Iy2[r     , idx_c1] + 4.0 * Iy2[r     , c     ] + 2.0 * Iy2[r     , idx_c2] + Iy2[idx_r2, idx_c1] + 2.0 * Iy2[idx_r2, c     ] + Iy2[idx_r2, idx_c2]

            Sxy =       Ixy[idx_r1, idx_c1] + 2.0 * Ixy[idx_r1, c     ] + Ixy[idx_r1, idx_c2] + 2.0 * Ixy[r     , idx_c1] + 4.0 * Ixy[r     , c     ] + 2.0 * Ixy[r     , idx_c2] + Ixy[idx_r2, idx_c1] + 2.0 * Ixy[idx_r2, c     ] + Ixy[idx_r2, idx_c2]

            Sx2 /= 16.0
            Sy2 /= 16.0
            Sxy /= 16.0

            det = Sx2 * Sy2 - Sxy ** 2
            trace = Sx2 + Sy2
            R[r, c] = det - k * trace

            if R[r, c] > threshold:
                count += 1

    #store the corner positions
    corner = np.zeros([count, 2])
    count = 0

    for r in range(w):
        for c in range(h):

            if R[r, c] > threshold:

                corner[count, 0] = r*1.0 / w
                corner[count, 1] = c*1.0 / h
                count += 1

    return corner

def plot_circles(input_img,
                 corner_list,
                 output_img,
                 circle_radius=5, 
                 circle_color_r=0,
                 circle_color_g=0,
                 circle_color_b=255,
                 grayscale=False):
    """Draws circles at the corners in corner_list.
    """
    r2 = circle_radius**2
    color = np.array([circle_color_r, circle_color_g, circle_color_b])

    if grayscale:
        color = np.max(color)

    if np.max(input_img) <= 1:
        color /= 255.0

    h = input_img.shape[0]
    w = input_img.shape[1]

    for i in range(corner_list.shape[0]):
        corner_r = corner_list[i, 0]
        corner_c = corner_list[i, 1]
        loop_width_start = int(max(0, corner_c * w - circle_radius))
        loop_width_stop = int(min(input_img.shape[1], corner_c * w + circle_radius))
        loop_height_start = int(max(0, corner_r * h - circle_radius))
        loop_height_stop = int(min(input_img.shape[0], corner_r * h + circle_radius))

        for r in range(loop_height_start, loop_height_stop):
            for c in range(loop_width_start, loop_width_stop):
                d2 = (r - loop_height_start - circle_radius)**2 + (c - loop_width_start - circle_radius)**2

                if d2 <= r2:
                    output_img[r, c] = color

    return output_img

def harris_corner(input_img, output_img):
    corners = find_corners(input_img)

    #input_img = skimage.img_as_float(skimage.io.imread('../images/window.png'))
    #output_img = np.copy(input_img)

    use_grayscale = True if len(input_img.shape) == 2 else False
    plotted_result = plot_circles(input_img, corners, output_img, 
        grayscale=use_grayscale)

    #plot.imshow(plotted_result, cmap="Greys_r")
    #plot.imshow(plotted_result)
    #plot.show()
    #exit()

    return plotted_result

input_img = '../images/window.png'
#input = skimage.io.imread(input_img)
#input = skimage.img_as_float(input)
#input = skimage.color.rgb2gray(input)

#corner = harris_corner(input)

def test(n=None):

    ans = util.test_image_pipeline_filename(
            harris_corner, 
            (input_img,), 
            n, 
            grayscale=True, 
            name='harris_corner',
            use_output_img=True)
    return util.combine_tests([ans])

if __name__ == '__main__':
    test()
