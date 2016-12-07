#include "c_util.h"
#include <omp.h>
#include <stdio.h>

/*
Blurs the image in matrix input_img and writes the values to output_img.

This uses a 3x3 Gaussian kernel to convolve with the image matrix.

            1/16 2/16 1/16
Kernel =    2/16 4/16 2/16
            1/16 2/16 1/16

For dealing with convolving along the edges of the image, we renormalize the
kernel based on which coordinates from the kernel are in-bounds.
*/
void blur_one_stage(FLOAT_IMG *input_img, FLOAT_IMG *output_img)
{
    int r, c;
    float kernel_norm;
    float kernel_accum;

    #pragma omp parallel for private(c, kernel_norm, kernel_accum)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            // center
            kernel_accum = 4.0 * input_img->data[r][c];
            kernel_norm = 4.0;

            // top left
            if(r > 0 && c > 0)
            {
                kernel_accum += 1.0 * input_img->data[r - 1][c - 1];
                kernel_norm += 1.0;
            }

            // top middle
            if(r > 0)
            {
                kernel_accum += 2.0 * input_img->data[r - 1][c    ];
                kernel_norm += 2.0;
            }

            // top right
            if(r > 0 && c < input_img->cols - 1)
            {
                kernel_accum += 1.0 * input_img->data[r - 1][c + 1];
                kernel_norm += 1.0;
            }

            // left
            if(c > 0)
            {
                kernel_accum += 2.0 * input_img->data[r    ][c - 1];
                kernel_norm += 2.0;
            }

            // right
            if(c < input_img->cols - 1)
            {
                kernel_accum += 2.0 * input_img->data[r    ][c + 1];
                kernel_norm += 2.0;
            }
            
            // bottom left
            if(r < input_img->rows - 1 && c > 0)
            {
                kernel_accum += 1.0 * input_img->data[r + 1][c - 1];
                kernel_norm += 1.0;
            }

            // bottom middle
            if(r < input_img->rows - 1)
            {
                kernel_accum += 2.0 * input_img->data[r + 1][c    ];
                kernel_norm += 2.0;
            }

            // bottom right
            if(r < input_img->rows - 1 && c < input_img->cols - 1)
            {
                kernel_accum += 1.0 * input_img->data[r + 1][c + 1];
                kernel_norm += 1.0;
            }

            output_img->data[r][c] = kernel_accum / kernel_norm;
        }
    }
}

int main(int argc, char **argv)
{
    FLOAT_IMG *input_img;
    
    if(argc == 1)
    {
        input_img = read_img_gray("../../images/temple_gray.png");
    }
    else
    {
        if(argc == 3)
        {
            input_img = read_img_gray(argv[1]);
        }
        else
        {
            printf("Usage: ./a.out in.png out.png or ./a.out\n");
            return 0;
        }
    }
    FLOAT_IMG *output_img = zeros_gray(input_img->rows, input_img->cols);
   
    double d1 = wall_time();
    blur_one_stage(input_img, output_img);
    double d2 = wall_time();

    printf("%f\n", d2 - d1);
    
    if(argc == 1)
        write_img_gray(output_img, "test.png");
    else
        write_img_gray(output_img, argv[2]);

    return 0;
}