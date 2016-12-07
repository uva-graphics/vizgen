"""
pacman_clean.py -- A version of the pacman renderer that is extra-easy to compile.

Renders a 2D version of pacman (meant to be run with Python 3.x)
"""

import math
import numpy as np
import random
import sys; sys.path += ['../../compiler']
import util
try:
    import skimage
    import skimage.color
    import skimage.io
    has_skimage = True
except ImportError:
    has_skimage = False         # For PyPy comparisons

use_gui = False

if use_gui:
    import pygame
    import time

def draw_circle(x, y, r, img, color_r, color_g, color_b):
    """Draws a Pacman to img

    Args:
        x, int, x location in img
        y, int, y location in img
        r, int, radius of circle
        img, np.array, 3D color image matrix
        color_r, int, red channel of color
        color_g, int, green channel of color
        color_b, int, blue channel of color
    """

    y_start = int(max(0, y - r))
    y_stop = int(min(y + r, img.shape[0] - 1))

    for y_i in range(y_start, y_stop):
        x_start = int(x - math.sqrt(r**2 - (y - y_i)**2))
        x_stop = int(x + math.sqrt(r**2 - (y - y_i)**2))

        for x_i in range(x_start, x_stop):
            img[x_i, y_i, 0] = color_r
            img[x_i, y_i, 1] = color_g
            img[x_i, y_i, 2] = color_b

def draw_pacman_right(x, y, r, img, ma, color_r, color_g, color_b):
    """Draws a Pacman to img

    Args:
        x, int, x location in img (center of pacman)
        y, int, y location in img (center of pacman)
        r, int, radius of pacman
        img, np.array, 3D color image matrix
        ma, float, angle mouth is open at ("mouth angle")
        color_r, float, red channel of color
        color_g, float, green channel of color
        color_b, float, blue channel of color
    """

    y_start = int(max(0, y - r))
    y_stop = int(min(y + r, img.shape[0] - 1))

    for y_i in range(y_start, y_stop):
        x_start = int(x - math.sqrt(r**2 - (y - y_i)**2))
        x_stop = int(x + math.sqrt(r**2 - (y - y_i)**2))

        # top half of mouth:
        if y_i > y - float(r) * math.sin(ma) and y_i <= y:
            r_mouth = float(y - y_i) / math.sin(ma)
            x_stop = int(x + r_mouth * math.cos(ma))

        # bottom half of mouth:
        elif y_i < y + float(r) * math.sin(ma) and y_i > y:
            r_mouth = float(y_i - y) / math.sin(ma)
            x_stop = int(x + r_mouth * math.cos(ma))

        for x_i in range(x_start, x_stop):
            img[x_i, y_i, 0] = color_r
            img[x_i, y_i, 1] = color_g
            img[x_i, y_i, 2] = color_b

    # draw the eye:
    draw_circle(x, y - int(r / 2), int(r / 10.), img, 0, 0, 0)

def draw_rect(x, y, w, h, img, color_r, color_g, color_b):
    """Draws a rectangle to img
    
    Args:
        x, int, x location in img
        y, int, y location in img
        w, int, width
        h, int, height
        img, np.array, 3D color image matrix
        color_r, int, red channel of color
        color_g, int, green channel of color
        color_b, int, blue channel of color
    """

    y_start = int(max(0, y))
    y_stop = int(min(y + h, img.shape[0] - 1))

    x_start = int(max(0, x))
    x_stop = int(min(x + w, img.shape[1] - 1))

    for y_i in range(y_start, y_stop):
        for x_i in range(x_start, x_stop):
            img[y_i, x_i, 0] = color_r
            img[y_i, x_i, 1] = color_g
            img[y_i, x_i, 2] = color_b

def draw_ghost(x, y, r, img, color_r, color_g, color_b, tf, blink):
    """Draws a ghost to img
    
    Args:
        x, int, x location in img
        y, int, y location in img
        r, int, radius of the ghosts's head
        img, np.array, 3D color image matrix
        color_r, int, red channel of color
        color_g, int, green channel of color
        color_b, int, blue channel of color
        tf, float, how much of ghost is not tentacles
        blink, boolean, whether or not the ghost should be blinking
    """

    y_start = int(max(0, y - r))
    y_stop = int(min(y + r, img.shape[1] - 1))

    for y_i in range(y_start, y_stop):
        x_start = int(x - math.sqrt(r**2 - (y - y_i)**2))
        x_stop = int(x + math.sqrt(r**2 - (y - y_i)**2))

        # bottom half of ghost:
        if y_i > y:
            x_start = int(max(0, x - r))
            x_stop = int(min(x + r, img.shape[0] - 1))

        # print(x_start, x_stop, y_start, y_stop)

        for x_i in range(x_start, x_stop):
            if y_i <= y + tf * r:
                img[x_i, y_i, 0] = color_r
                img[x_i, y_i, 1] = color_g
                img[x_i, y_i, 2] = color_b
            else:
                if x_i < x - r * 5/7. or (x_i > x - r * 3/7. and x_i < x - r * 1/7.) or (x_i > x + r * 1/7. and x_i < x + r * 3/7.) or (x_i > x + r * 5/7.):
                    img[x_i, y_i, 0] = color_r
                    img[x_i, y_i, 1] = color_g
                    img[x_i, y_i, 2] = color_b

    # draw the eye:
    if not blink:
        draw_circle(x - int(r / 4), y - int(r / 2), int(r / 5.), img, 1, 1, 1)
        draw_circle(x + int(r / 4), y - int(r / 2), int(r / 5.), img, 1, 1, 1)
        draw_circle(x - int(r / 8.), y - int(r / 2), int(r / 9.), img, 0, 0, 1)
        draw_circle(x + int(r * 3 / 8.), y - int(r / 2), int(r / 9.), img, 0, 0, 1)
    else:
        draw_rect(y - int(r / 2), x - int(r / 4), r / 8., r / 4., img, 0, 0, 0)
        draw_rect(y - int(r / 2), x + int(r / 4), r / 8., r / 4., img, 0, 0, 0)

def render_1_frame_no_write(input_img, output_img, time):
    """Renders a single frame, for testing purposes
    """

    ma = int(time) % 16

    if ma > 7:
        ma = 8 - (ma - 8)

    # clear screen
    draw_rect(0, 0, output_img.shape[0], output_img.shape[1], output_img, 0, 0, 0)
    
    # draw pacman:
    draw_pacman_right(100, 100, 50, output_img, ma * math.pi/32., 1, 1, 0)

    # draw a dot:
    draw_circle(200, 100, 15, output_img, 1, 1, 1)
    draw_circle(275, 100, 15, output_img, 1, 1, 1)
    draw_circle(350, 100, 15, output_img, 1, 1, 1)

    # draw a ghost:
    draw_ghost(500, 100, 50, output_img, 0, 1, 1, 0.75, ma == 7)

def animate(width, height):
    """Animates pacman and displays its framerate

    Draws a Pacman, multiple dots, and a ghost
    """

    num_frames_for_running_average = 25
    frame_times = np.zeros(num_frames_for_running_average)
    frame_index = 0
    img = np.zeros((width, height, 3))

    ma_index = 0.
    ma_diff = 1.

    pygame.init()

    screen = pygame.display.set_mode((width, height))

    current_time = time.time()
    old_time = current_time

    while True:
        # clear screen
        draw_rect(0, 0, width, height, img, 0, 0, 0)
        
        # draw pacman:
        draw_pacman_right(100, 100, 50, img, ma_index * math.pi/32., 1, 1, 0)
        ma_index += ma_diff

        # update angle differential
        if ma_index == 8:
            ma_diff = -1.
        elif ma_index == 0:
            ma_diff = 1.

        # draw a dot:
        draw_circle(200, 100, 15, img, 1, 1, 1)
        draw_circle(275, 100, 15, img, 1, 1, 1)
        draw_circle(350, 100, 15, img, 1, 1, 1)

        # draw a ghost:
        draw_ghost(500, 100, 50, img, 0, 1, 1, 0.75, ma_index == 7)

        # write numpy array to screen
        pygame.surfarray.blit_array(screen, img * 255)
        pygame.display.flip()
        
        old_time = current_time
        current_time = time.time()
        frame_times[frame_index] = current_time - old_time
        frame_index += 1

        if frame_index == num_frames_for_running_average:
            ave = np.average(frame_times)
            frame_index = 0

            print("Average drawtime per frame: %.7f seconds (%.3f fps)" % (
                ave, 1 / ave))

            # update clock to skip the computation done in the if-block
            current_time = time.time()

def test(time=0.0, n=None):
    """Default unit tests which compare any optimized output with the original Python output.
    """
    img = np.zeros((600, 200, 3))
    return util.test_image_pipeline(
        image_func=render_1_frame_no_write,
        input_imgL=(img,),
        n=None,
        ground_truth_output=None,
        use_output_img=True,
        name="pacman.render_1_frame_no_write python",
        imdiff_ignore_last_element=True,
        additional_kw_args={'time': time})

if __name__ == "__main__":
    if use_gui:
        animate(600, 200)
    else:
        test()