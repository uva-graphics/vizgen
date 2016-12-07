#include "c_util.h"
#include "vector_headers.h"
#include <pmmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>

void naive_2_stage_blur(FLOAT_IMG *img, FLOAT_IMG *output_img)
{
    int r, c, k;
    FLOAT_IMG *temp_img = zeros_gray(img->rows, img->cols);

    for(r = 1; r < img->rows - 1; r++)
    {
        for(c = 0; c < img->cols; c++)
        {
            temp_img->data[r][c] = (img->data[r - 1][c] + 
                                    img->data[r    ][c] + 
                                    img->data[r + 1][c]) / 3.0f;
        }
    }

    for(r = 0; r < img->rows; r++)
    {
        for(c = 1; c < img->cols - 1; c++)
        { 
            output_img->data[r][c] = (temp_img->data[r][c - 1] + 
                                      temp_img->data[r][c    ] + 
                                      temp_img->data[r][c + 1]) / 3.0f;
        }
    }

    free_img_gray(temp_img);
}

void vectorized_2_stage_blur(FLOAT_IMG *img, FLOAT_IMG *output_img)
{
    // used the following site as an example:
    // https://hgomersall.wordpress.com/2012/11/02/speedy-fast-1d-convolution-with-sse/

    FLOAT_IMG *temp_img = zeros_gray(img->rows, img->cols);
    int r, c, k, i;

    // we use a kernel of size 3, but our array needs to be length 4 because 
    // our vectors are of size 4)
    float kernel[4] = {0.3333, 0.3333, 0.3334, 0.0};
    float aligned_kernel[4] __attribute__ ((aligned (16)));

    // a list of vectors, one for each element of the kernel
    __m128 kernel_block[4] __attribute__((aligned (16)));

    // we want to copy the values from the kernel into vectors full of them.
    // example, if kernel = {0.1, 0.2, 0.3, 0.4}, then kernel_block would be: 
    //      {<0.1, 0.1, 0.1, 0.1>, 
    //       <0.2, 0.2, 0.2, 0.2>, 
    //       <0.3, 0.3, 0.3, 0.3>, 
    //       <0.4, 0.4, 0.4, 0.4>} 

    for(i = 0; i < 4; i++)
    {
        aligned_kernel[0] = kernel[i];
        aligned_kernel[1] = kernel[i];
        aligned_kernel[2] = kernel[i];
        aligned_kernel[3] = kernel[i];

        kernel_block[i] = _mm_load_ps(aligned_kernel);
    }

    // vector to temporarily store values from the input image
    __m128 data_block __attribute__ ((aligned (16)));
 
    // vector to temporarily store the product of a multiplication
    __m128 prod __attribute__ ((aligned (16)));

    // vector to accumulate the ultimate value for a pixel in the image
    __m128 acc __attribute__ ((aligned (16)));

    for(r = 0; r < img->rows; r++)
    {
        for(c = 0; c < img->cols; c += 4)
        {
            // set the accumulator to zero:
            acc = _mm_setzero_ps();

            for(k = 0; k < 4; k++)
            {
                // Load 4-float data block. These needs to be an unaliged
                // load (_mm_loadu_ps) as we step one sample at a time.
                data_block = _mm_loadu_ps(
                    img->flat_data + (r * img->rows) + c + k);
                prod = _mm_mul_ps(kernel_block[k], data_block);
                acc = _mm_add_ps(acc, prod);
            }

            _mm_storeu_ps(temp_img->flat_data + (r * img->rows) + c, acc);
        }
    }

    for(r = 0; r < img->rows - 2; r++)
    {
        for(c = 0; c < img->cols; c++)
        {
            output_img->data[r][c] = (temp_img->data[r    ][c] + 
                                      temp_img->data[r + 1][c] + 
                                      temp_img->data[r + 2][c]) / 3.0f;
        }
    }

    free_img_gray(temp_img);
}

void vectorized_2_stage_blur_2(FLOAT_IMG *img, FLOAT_IMG *output_img)
{
    FLOAT_IMG *temp_img = zeros_gray(img->rows, img->cols);
    int r, c, i;
    v4float v1, v2, v3; 

    // first do the pass over the Y axis
    for(r = 1; r < img->rows - 1; r++)
    {
        for(c = 0; c < img->cols; c++)
        {
            temp_img->data[r][c] = (img->data[r - 1][c] + 
                                    img->data[r    ][c] + 
                                    img->data[r + 1][c]) / 3.0f;
        }
    }

    for(r = 0; r < temp_img->rows; r++)
    {
        // offset by 4 on each edge so you can load 1 to the left and right
        // while still having aligned load for initial load
        for(c = 4; c < temp_img->cols - 4; c += 4)
        {
            v1 = _mm_load_ps(temp_img->flat_data + (r * temp_img->rows) + c);
            v2 = _mm_loadu_ps(temp_img->flat_data + (r * temp_img->rows) + c-1);
            v3 = _mm_loadu_ps(temp_img->flat_data + (r * temp_img->rows) + c+1);

            _mm_store_ps(output_img->flat_data + (r * output_img->rows) + c, 
                (v1 + v2 + v3) / 3.0f);
        }
    }

    free_img_gray(temp_img);
}

void vectorized_2_stage_blur_3(FLOAT_IMG *img, FLOAT_IMG *output_img)
{
    FLOAT_IMG *temp_img = zeros_gray(img->rows, img->cols);
    int r, c, i;
    v8float v1, v2, v3;

    // first do the pass over the Y axis
    for(r = 1; r < img->rows - 1; r++)
    {
        for(c = 0; c < img->cols; c++)
        {
            temp_img->data[r][c] = (img->data[r - 1][c] + 
                                    img->data[r    ][c] + 
                                    img->data[r + 1][c]) / 3.0f;
        }
    }

    for(r = 0; r < temp_img->rows; r++)
    {
        for(c = 8; c < temp_img->cols - 8; c += 8)
        {
            v1 = _mm256_load_ps(temp_img->flat_data + (r * temp_img->rows) + c);
            v2 = _mm256_loadu_ps(temp_img->flat_data + (r * temp_img->rows) + c - 1);
            v3 = _mm256_loadu_ps(temp_img->flat_data + (r * temp_img->rows) + c + 1);

            _mm256_store_ps(output_img->flat_data + (r * output_img->rows) + c,
                (v1 + v2 + v3) / 3.0f);
        }
    }

    free_img_gray(temp_img);
}

int main(int argc, char **argv)
{
    int dims[4] = {512, 1024, 2048, 4096};
    FLOAT_IMG *r, *o;
    int i, j, num_iter = 500, num_dims = 4;
    double t_total1 = 0, t_total2 = 0, t_total3 = 0, t1, t2;

    printf("Running tests %d iterations:\n-----\n", num_iter);

    for(j = 0; j < num_dims; j++)
    {
        printf("Using dimensions: %dx%d\n", dims[j], dims[j]);

        r = random_gray(dims[j], dims[j]);
        o = zeros_gray(dims[j], dims[j]);

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            naive_2_stage_blur(r, o);
            t2 = wall_time();
            t_total1 += (t2 - t1);
        }

        printf("Average time for naive 2-stage blur: %f seconds\n", 
            t_total1 / ((double)num_iter));
        // print_img_gray(o);

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            vectorized_2_stage_blur_2(r, o);
            t2 = wall_time();
            t_total2 += (t2 - t1);
        }

        printf("Average time for 4D vectorized 2-stage blur: %f seconds\n", 
            t_total2 / ((double)num_iter));
        // print_img_gray(o);

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            vectorized_2_stage_blur_3(r, o);
            t2 = wall_time();
            t_total3 += (t2 - t1);
        }

        printf("Average time for 8D vectorized 2-stage blur: %f seconds\n", 
            t_total3 / ((double)num_iter));

        printf("Speedup from 4D vectorized 2-stage blur: %f\n", t_total1 / t_total2);
        printf("Speedup from 8D vectorized 2-stage blur: %f\n", t_total1 / t_total3);
        printf("\n");

        free_img_gray(r);
        free_img_gray(o);
    }

    return 0;
}