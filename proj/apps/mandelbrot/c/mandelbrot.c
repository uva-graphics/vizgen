#include "c_util.h"
#include <omp.h>
#include <stdio.h>

// Globals:
float MAX_ITER = 255;
float X_START = -2.5;
float X_STOP = 1.5;
float Y_START = -1.5;
float Y_STOP = 1.5;
float PIXELS_PER_UNIT = 200;

/*
Generates a visualization of the Mandelbrot set; this is for grayscale images
*/
void mandelbrot(FLOAT_IMG *output_img, float t)
{
    int r, c, num_iter;
    float escape_radius_squared = 2.0 * 2.0;
    float z_real, z_imag, c_real, c_imag, new_z_real;
    float scale = 1.0 + 2.0 * t / 10.0;

    Y_START += t / 6.0;
    Y_STOP += t / 6.0;
    
    X_START /= scale;
    X_STOP /= scale;
    Y_START /= scale;
    Y_STOP /= scale;
    PIXELS_PER_UNIT *= scale;

    #pragma omp parallel for private(c, z_real, z_imag, c_real, c_imag, new_z_real, num_iter)
    for(r = 0; r < output_img->rows; r++)
    {
        for(c = 0; c < output_img->cols; c++)
        {
            z_real = 0.0;
            z_imag = 0.0;

            c_real = ((float)c) / PIXELS_PER_UNIT + X_START;
            c_imag = ((float)(output_img->rows - r)) / PIXELS_PER_UNIT + 
                Y_START;

            num_iter = 0;

            while(num_iter < MAX_ITER)
            {
                if(z_real * z_real + z_imag * z_imag > escape_radius_squared)
                {
                    break;
                }

                new_z_real = z_real * z_real - z_imag * z_imag + c_real;
                z_imag = 2.0 * z_real * z_imag + c_imag;
                z_real = new_z_real;
                num_iter += 1;
            }

            output_img->data[r][c] = ((float)num_iter) / MAX_ITER;
        }
    }
}

int main(int argc, char **argv)
{
    FLOAT_IMG *output_img = zeros_gray(600, 800);
    double t1, t2;

    if(argc == 3)
    {
        t1 = wall_time();
        mandelbrot(output_img, atoi(argv[2]));
        t2 = wall_time();
        write_img_gray(output_img, argv[1]);
    }

    else if(argc == 2)
    {
        t1 = wall_time();
        mandelbrot(output_img, 10.0);
        t2 = wall_time();
        write_img_gray(output_img, argv[1]);
    }

    else if(argc == 1)
    {
        t1 = wall_time();
        mandelbrot(output_img, 10.0);
        t2 = wall_time();
        write_img_gray(output_img, "test.png");
    }

    else
    {
        printf("Usages:\n");
        printf("\t./a.out out.png <time>\n");
        printf("\t./a.out out.png\n");
        printf("\t./a.out\n");
        return 0;
    }

    printf("%f\n", t2 - t1);

    return 0;
}