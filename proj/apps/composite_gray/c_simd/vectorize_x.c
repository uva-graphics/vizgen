#include "c_util.h"
#include "vector_headers.h"
#include <pmmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>

void naive_composite(FLOAT_IMG *background, 
                     FLOAT_IMG *foreground, 
                     FLOAT_IMG *foreground_alpha, 
                     FLOAT_IMG *output_img)
{
    int r, c;
    float alpha;
    
    #pragma omp parallel for private(c, alpha)
    for(r = 0; r < background->rows; r++)
    {
        for(c = 0; c < background->cols; c++)
        {
            alpha = foreground_alpha->data[r][c];
            output_img->data[r][c] = background->data[r][c] * (1.0f - alpha) + 
                                     foreground->data[r][c] * alpha;
        }
    }
}

void vectorized_composite_4d(FLOAT_IMG *background, 
                             FLOAT_IMG *foreground, 
                             FLOAT_IMG *foreground_alpha, 
                             FLOAT_IMG *output_img)
{
    int r, c, offset;
    v4float v_a, v_fg, v_bg;

    for(r = 0; r < background->rows; r++)
    {
        for(c = 0; c < background->cols; c += 4)
        {
            offset = (r * background->rows) + c;

            v_a = _mm_load_ps(foreground_alpha->flat_data + offset);
            v_fg = _mm_load_ps(foreground->flat_data + offset);
            v_bg = _mm_load_ps(background->flat_data + offset);

            _mm_store_ps(output_img->flat_data + offset, 
                v_bg * (1.0f - v_a) + v_fg * v_a);
        }
    }
}

void vectorized_composite_8d(FLOAT_IMG *background, 
                             FLOAT_IMG *foreground, 
                             FLOAT_IMG *foreground_alpha, 
                             FLOAT_IMG *output_img)
{
    int r, c, offset;
    v8float v_a, v_fg, v_bg;

    for(r = 0; r < background->rows; r++)
    {
        for(c = 0; c < background->cols; c += 8)
        {
            offset = (r * background->rows) + c;

            v_a = _mm256_load_ps(foreground_alpha->flat_data + offset);
            v_fg = _mm256_load_ps(foreground->flat_data + offset);
            v_bg = _mm256_load_ps(background->flat_data + offset);

            _mm256_store_ps(output_img->flat_data + offset, 
                v_bg * (1.0f - v_a) + v_fg * v_a);
        }
    }
}

int main(int argc, char **argv)
{
    int dims[4] = {512, 1024, 2048, 4096};
    int i, j, num_iter = 500, num_dims = 4;
    FLOAT_IMG *f, *fa, *b, *o;
    double t_total1 = 0, t_total2 = 0, t_total3 = 0, t1, t2;

    printf("Running tests %d iterations:\n-----\n", num_iter);

    for(j = 0; j < num_dims; j++)
    {
        printf("Using dimensions: %dx%d\n", dims[j], dims[j]);

        f = random_gray(dims[j], dims[j]);
        fa = random_alpha(dims[j], dims[j]);
        b = random_gray(dims[j], dims[j]);
        o = zeros_gray(dims[j], dims[j]);

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            naive_composite(b, f, fa, o);
            t2 = wall_time();
            t_total1 += (t2 - t1);
        }

        printf("Average time for naive composite: %f seconds\n", 
            t_total1 / ((double)num_iter));

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            vectorized_composite_4d(b, f, fa, o);
            t2 = wall_time();
            t_total2 += (t2 - t1);
        }

        printf("Average time for 4D vectorized composite: %f seconds\n", 
            t_total2 / ((double)num_iter));

        for(i = 0; i < num_iter; i++)
        {
            t1 = wall_time();
            vectorized_composite_8d(b, f, fa, o);
            t2 = wall_time();
            t_total3 += (t2 - t1);
        }

        printf("Average time for 8D vectorized composite: %f seconds\n", 
            t_total3 / ((double)num_iter));

        printf("Speedup from 4D vectorized 2-stage blur: %f\n",
            t_total1 / t_total2);
        printf("Speedup from 8D vectorized 2-stage blur: %f\n",
            t_total1 / t_total3);
        printf("\n");

        free_img_gray(f);
        free_img_gray(fa);
        free_img_gray(b);
        free_img_gray(o);
    }

    return 0;
}