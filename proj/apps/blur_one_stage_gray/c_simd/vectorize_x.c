#include "c_util.h"
#include "vector_headers.h"
#include <pmmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>

void naive_blur_one_stage(FLOAT_IMG *input_img, FLOAT_IMG *output_img)
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

void vectorized_blur_one_stage_4d(FLOAT_IMG *input_img, FLOAT_IMG *output_img)
{
    int r, c, rows = input_img->rows;
    v4float tmp, kernel_accum;
    float kernel_norm;

    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c += 4)
        {
            // center
            tmp = _mm_loadu_ps(input_img->flat_data + r * rows + c);
            kernel_accum = 4.0 * tmp;
            kernel_norm = 4.0;

            // top left
            if(r > 0 && c > 0)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + (r - 1) * rows + c - 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            // top middle
            if(r > 0)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + (r - 1) * rows + c);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }

            // top right
            if(r > 0 && c < input_img->cols - 1)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + (r - 1) * rows + c + 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            // left
            if(c > 0)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + r * rows + c - 1);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }

            // right
            if(c < input_img->cols - 1)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + r * rows + c + 1);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }
            
            // bottom left
            if(r < input_img->rows - 1 && c > 0)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + (r + 1) * rows + c - 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            // bottom middle
            if(r < input_img->rows - 1)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + (r + 1) * rows + c);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }

            // bottom right
            if(r < input_img->rows - 1 && c < input_img->cols - 1)
            {
                tmp = _mm_loadu_ps(input_img->flat_data + (r + 1) * rows + c + 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            _mm_storeu_ps(output_img->flat_data + r * rows + c,
                kernel_accum / kernel_norm);
        }
    }
}

void vectorized_blur_one_stage_8d(FLOAT_IMG *input_img, FLOAT_IMG *output_img)
{
    int r, c, rows = input_img->rows;
    v8float tmp, kernel_accum;
    float kernel_norm;

    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c += 8)
        {
            // center
            tmp = _mm256_loadu_ps(input_img->flat_data + r * rows + c);
            kernel_accum = 4.0 * tmp;
            kernel_norm = 4.0;

            // top left
            if(r > 0 && c > 0)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + (r - 1) * rows + c - 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            // top middle
            if(r > 0)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + (r - 1) * rows + c);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }

            // top right
            if(r > 0 && c < input_img->cols - 1)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + (r - 1) * rows + c + 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            // left
            if(c > 0)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + r * rows + c - 1);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }

            // right
            if(c < input_img->cols - 1)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + r * rows + c + 1);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }
            
            // bottom left
            if(r < input_img->rows - 1 && c > 0)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + (r + 1) * rows + c - 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            // bottom middle
            if(r < input_img->rows - 1)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + (r + 1) * rows + c);
                kernel_accum = kernel_accum + 2.0 * tmp;
                kernel_norm += 2.0;
            }

            // bottom right
            if(r < input_img->rows - 1 && c < input_img->cols - 1)
            {
                tmp = _mm256_loadu_ps(input_img->flat_data + (r + 1) * rows + c + 1);
                kernel_accum = kernel_accum + tmp;
                kernel_norm += 1.0;
            }

            _mm256_storeu_ps(output_img->flat_data + r * rows + c,
                kernel_accum / kernel_norm);
        }
    }
}

int main(int argc, char **argv)
{
    int dims[4] = {512, 1024, 2048, 4096};
    int i, j, num_iter = 500, num_dims = 4;
    FLOAT_IMG *r, *o;
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
            naive_blur_one_stage(r, o);
            t2 = wall_time();
            t_total1 += (t2 - t1);
        }

        printf("Average time for naive 1-stage blur: %f seconds\n", 
            t_total1 / ((double)num_iter));

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            vectorized_blur_one_stage_4d(r, o);
            t2 = wall_time();
            t_total2 += (t2 - t1);
        }

        printf("Average time for 4D vectorized 1-stage blur: %f seconds\n", 
            t_total2 / ((double)num_iter));

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            vectorized_blur_one_stage_8d(r, o);
            t2 = wall_time();
            t_total3 += (t2 - t1);
        }

        printf("Average time for 8D vectorized 1-stage blur: %f seconds\n", 
            t_total3 / ((double)num_iter));

        printf("Speedup from 4D vectorized 1-stage blur: %f\n",
            t_total1 / t_total2);
        printf("Speedup from 8D vectorized 1-stage blur: %f\n",
            t_total1 / t_total3);
        printf("\n");

        free_img_gray(r);
        free_img_gray(o);
    }

    return 0;
}