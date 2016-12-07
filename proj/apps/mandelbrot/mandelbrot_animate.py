"""mandelbrot.py

Generates a visual representation of the Mandelbrot Set (meant to be run with
Python 3.x)

Author: Sam Prestwood
"""

import numpy as np
import sys; sys.path += ['../../compiler']
import util
try:
    import skimage
    import skimage.color
    import skimage.io
    has_skimage = True
except ImportError:
    has_skimage = False         # For PyPy comparisons

def mandelbrot(x_start, x_stop, y_start, y_stop, max_iter, pixels_per_unit, time=0.0):
    """Returns a 2D array that represents the Mandelbrot Set

    Args:
        x_start: int, the lower bound of the x range
        x_stop: int, the upper bound of the x range
        y_start: int, the lower bound of the y range
        y_stop: int, the upper bound of the y range
        max_iter: int, the maximum number of iterations performed at a pixels
        pixels_per_unit: int, how many pixels to render per unit in the x and y
            range

    Returns:
        matrix: numpy array that represents the Mandelbrot Set
    """

    img_num_rows = int((y_stop - y_start) * pixels_per_unit)
    img_num_cols = int((x_stop - x_start) * pixels_per_unit)
    escape_radius_squared = 2.0**2
    
    scale = 1.0 + 2.0 * time / 10.0
    
    y_start += time / 6.0
    y_stop += time / 6.0
    
    x_start /= scale
    x_stop /= scale
    y_start /= scale
    y_stop /= scale
    pixels_per_unit *= scale
    
    matrix = np.zeros((img_num_rows, img_num_cols))

    for r in range(img_num_rows):
        for c in range(img_num_cols):
            z_real = 0.0
            z_imag = 0.0

            comp_real = c / pixels_per_unit + x_start
            comp_imag = (img_num_rows - r) / pixels_per_unit + y_start
            
            num_iter = 0

            while num_iter < max_iter:
                if z_real * z_real + z_imag * z_imag > escape_radius_squared:
                    break
                new_z_real = z_real * z_real - z_imag * z_imag + comp_real
                z_imag = 2 * z_real * z_imag + comp_imag
                z_real = new_z_real
                num_iter += 1


            matrix[r, c] = float(num_iter) / max_iter

        # overwrite_line("{:.1f}% done".format(float(r * 100 / img_num_rows)))

    return matrix

def mandelbrot_gray(matrix, output_file_name):
    """Generates a grayscale version of the Mandelbrot Set

    Writes its output file to output_file_name

    Args:
        matrix: np.array, 2D array representing the mandelbrot set
        output_file_name: string, filename to write image to
    """

    print("\nWriting image to:", output_file_name)
    skimage.io.imsave(output_file_name, matrix)

def mandelbrot_color(matrix, output_file_name):
    """Generates a color version of the Mandelbrot Set

    Writes its output file to output_file_name

     Args:
        matrix: np.array, 2D array representing the mandelbrot set
        output_file_name: string, filename to write image to
    """

    # I wasn't quite sure on how to do the coloring, so I just interpolated
    # between two colors:
    color1 = np.array([[.2], [.2], [.8]])
    color2 = np.array([[1], [.2], [.5]])

    color_img = np.zeros((matrix.shape[0], matrix.shape[1], 3))

    color_img[:, :, 0] = color1[0] + matrix[:, :] * (color2[0] - color1[0])
    color_img[:, :, 1] = color1[1] + matrix[:, :] * (color2[1] - color1[1])
    color_img[:, :, 2] = color1[2] + matrix[:, :] * (color2[2] - color1[2])

    print("\nWriting image to:", output_file_name)
    skimage.io.imsave(output_file_name, color_img)

MAX_ITER = 255
X_START = -2.5
X_STOP = 1.5
Y_START = -1.5
Y_STOP = 1.5
PIXELS_PER_UNIT = 200.0

IMG_NUM_ROWS = int((Y_STOP - Y_START) * PIXELS_PER_UNIT)
IMG_NUM_COLS = int((X_STOP - X_START) * PIXELS_PER_UNIT)

MANDELBROT_ARGS = (X_START, X_STOP, Y_START, Y_STOP, MAX_ITER, PIXELS_PER_UNIT, 0.0)

def test(time=0.0, n=None):
    """Default unit tests which compare any optimized output with the original Python output.
    """
    mandelbrot(X_START, X_STOP, Y_START, Y_STOP, MAX_ITER, PIXELS_PER_UNIT/10, time)
    util.is_initial_run = True
    MANDELBROT_ARGS = (X_START, X_STOP, Y_START, Y_STOP, MAX_ITER, PIXELS_PER_UNIT, time)
    
    return util.combine_tests((util.test_image_pipeline_filename(
        image_func=mandelbrot, 
        in_filenameL=tuple(), 
        n=n, 
        name="mandelbrot.mandelbrot gray python", 
        additional_args=MANDELBROT_ARGS),))

if __name__ == "__main__":
    test(time=10.0)

#for i in range(0, 11):
#    ans = test(i)
#    skimage.io.imsave('output/' + str(i) + '.png', np.clip(ans['output'], 0, 1))
    #matrix = mandelbrot(X_START,
    #                    X_STOP,
    #                    Y_START,
    #                    Y_STOP,
    #                    MAX_ITER,
    #                    PIXELS_PER_UNIT)
    #mandelbrot_gray(matrix, "mandelbrot_gray.png")
    #mandelbrot_color(matrix, "mandelbrot_color.png")