#include "c_util.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>

float max(float a, float b)
{
    return a > b ? a : b;
}

float min(float a, float b)
{
    return a < b ? a : b;
}

real clip(real input, real lower_bound, real upper_bound)
{
    return min(max(input, lower_bound), upper_bound);
}

void interpolate(FLOAT4_IMG *input_img, FLOAT3_IMG *output_img, int levels)
{
    real ****downsampled;
    real ****downx;
    real ****interpolated;
    real ****upsampled;
    real ****upsampledx;
    real ***normalize;
    
    real *downsampled_data = malloc(levels * input_img->rows * input_img->cols * 4 * sizeof(real));
    real **downsampled_pointer = malloc(levels * sizeof(real ***) + levels * input_img->rows * sizeof(real **) + levels * input_img->rows * input_img->cols * sizeof(real *));
    downsampled = (real ****)downsampled_pointer;
    
    real *downx_data = malloc(levels * input_img->rows * input_img->cols * 4 * sizeof(real));
    real **downx_pointer = malloc(levels * sizeof(real ***) + levels * input_img->rows * sizeof(real **) + levels * input_img->rows * input_img->cols * sizeof(real *));
    downx = (real ****)downx_pointer;
    
    real *interpolated_data = malloc(levels * input_img->rows * input_img->cols * 4 * sizeof(real));
    real **interpolated_pointer = malloc(levels * sizeof(real ***) + levels * input_img->rows * sizeof(real **) + levels * input_img->rows * input_img->cols * sizeof(real *));
    interpolated = (real ****)interpolated_pointer;
    
    real *upsampled_data = malloc(levels * input_img->rows * input_img->cols * 4 * sizeof(real));
    real **upsampled_pointer = malloc(levels * sizeof(real ***) + levels * input_img->rows * sizeof(real **) + levels * input_img->rows * input_img->cols * sizeof(real *));
    upsampled = (real ****)upsampled_pointer;
    
    real *upsampledx_data = malloc(levels * input_img->rows * input_img->cols * 4 * sizeof(real));
    real **upsampledx_pointer = malloc(levels * sizeof(real ***) + levels * input_img->rows * sizeof(real **) + levels * input_img->rows * input_img->cols * sizeof(real *));
    upsampledx = (real ****)upsampledx_pointer;
    
    int r, c, l;
    
    #pragma omp parallel for private(r, c)
    for (l = 0; l < levels; l++)
    {
        downsampled[l] = (real ***)&downsampled_pointer[levels + l * input_img->rows];
        downx[l] = (real ***)&downx_pointer[levels + l * input_img->rows];
        interpolated[l] = (real ***)&interpolated_pointer[levels + l * input_img->rows];
        upsampled[l] = (real ***)&upsampled_pointer[levels + l * input_img->rows];
        upsampledx[l] = (real ***)&upsampledx_pointer[levels + l * input_img->rows];
        
        for (r = 0; r < input_img->rows; r++)
        {
            downsampled[l][r] = (real **)&downsampled_pointer[levels + levels * input_img->rows +  l * input_img->rows * input_img->cols + r * input_img->cols];
            downx[l][r] = (real **)&downx_pointer[levels + levels * input_img->rows +  l * input_img->rows * input_img->cols + r * input_img->cols];
            interpolated[l][r] = (real **)&interpolated_pointer[levels + levels * input_img->rows +  l * input_img->rows * input_img->cols + r * input_img->cols];
            upsampled[l][r] = (real **)&upsampled_pointer[levels + levels * input_img->rows +  l * input_img->rows * input_img->cols + r * input_img->cols];
            upsampledx[l][r] = (real **)&upsampledx_pointer[levels + levels * input_img->rows +  l * input_img->rows * input_img->cols + r * input_img->cols];
            
            for (c = 0; c < input_img->cols; c++)
            {
                downsampled[l][r][c] = &downsampled_data[4 * (l * input_img->rows * input_img->cols + r * input_img->cols + c)];
                downx[l][r][c] = &downx_data[4 * (l * input_img->rows * input_img->cols + r * input_img->cols + c)];
                interpolated[l][r][c] = &interpolated_data[4 * (l * input_img->rows * input_img->cols + r * input_img->cols + c)];
                upsampled[l][r][c] = &upsampled_data[4 * (l * input_img->rows * input_img->cols + r * input_img->cols + c)];
                upsampledx[l][r][c] = &upsampledx_data[4 * (l * input_img->rows * input_img->cols + r * input_img->cols + c)];
                
                downsampled[l][r][c][0] = 0.0f;
                downsampled[l][r][c][1] = 0.0f;
                downsampled[l][r][c][2] = 0.0f;
                downsampled[l][r][c][3] = 0.0f;
                
                downx[l][r][c][0] = 0.0f;
                downx[l][r][c][1] = 0.0f;
                downx[l][r][c][2] = 0.0f;
                downx[l][r][c][3] = 0.0f;
                
                interpolated[l][r][c][0] = 0.0f;
                interpolated[l][r][c][1] = 0.0f;
                interpolated[l][r][c][2] = 0.0f;
                interpolated[l][r][c][3] = 0.0f;
                
                upsampled[l][r][c][0] = 0.0f;
                upsampled[l][r][c][1] = 0.0f;
                upsampled[l][r][c][2] = 0.0f;
                upsampled[l][r][c][3] = 0.0f;
                
                upsampledx[l][r][c][0] = 0.0f;
                upsampledx[l][r][c][1] = 0.0f;
                upsampledx[l][r][c][2] = 0.0f;
                upsampledx[l][r][c][3] = 0.0f;
            }
        }
    }

    #pragma omp parallel for private(c)
    for (r = 0; r < input_img->rows; r++)
    {
        for (c = 0; c < input_img->cols; c++)
        {
            downsampled[0][r][c][0] = input_img->data[r][c][3] * input_img->data[r][c][0];
            downsampled[0][r][c][1] = input_img->data[r][c][3] * input_img->data[r][c][1];
            downsampled[0][r][c][2] = input_img->data[r][c][3] * input_img->data[r][c][2];
            downsampled[0][r][c][3] = input_img->data[r][c][3] * input_img->data[r][c][3];
        }
    }
    
    real ***prev;
    
    for (l = 1; l < levels; l++)
    {
        prev = downsampled[l - 1];
        
        #pragma omp parallel private(c)
        for (r = 0; r < input_img->rows >> (l - 1); r++)
        {
            for(c = 1; c < input_img->cols >> l; c++)
            {
                downx[l][r][c][0] = (prev[r][2 * c - 1][0] + 2.0f * prev[r][2 * c][0] + prev[r][2 * c + 1][0]) * 0.25f;
                downx[l][r][c][1] = (prev[r][2 * c - 1][1] + 2.0f * prev[r][2 * c][1] + prev[r][2 * c + 1][1]) * 0.25f;
                downx[l][r][c][2] = (prev[r][2 * c - 1][2] + 2.0f * prev[r][2 * c][2] + prev[r][2 * c + 1][2]) * 0.25f;
                downx[l][r][c][3] = (prev[r][2 * c - 1][3] + 2.0f * prev[r][2 * c][3] + prev[r][2 * c + 1][3]) * 0.25f;
            }
            downx[l][r][0][0] = (3.0f * prev[r][0][0] + prev[r][1][0]) * 0.25f;
            downx[l][r][0][1] = (3.0f * prev[r][0][1] + prev[r][1][1]) * 0.25f;
            downx[l][r][0][2] = (3.0f * prev[r][0][2] + prev[r][1][2]) * 0.25f;
            downx[l][r][0][3] = (3.0f * prev[r][0][3] + prev[r][1][3]) * 0.25f;
        }
        
        #pragma omp parallel private(r)
        for(c = 0; c < input_img->cols >> l; c++)
        {
            for(r = 1; r < input_img->rows >> l; r++)
            {
                downsampled[l][r][c][0] = (downx[l][2 * r - 1][c][0] + 2.0f * downx[l][2 * r][c][0] + downx[l][2 * r + 1][c][0]) * 0.25f;
                downsampled[l][r][c][1] = (downx[l][2 * r - 1][c][1] + 2.0f * downx[l][2 * r][c][1] + downx[l][2 * r + 1][c][1]) * 0.25f;
                downsampled[l][r][c][2] = (downx[l][2 * r - 1][c][2] + 2.0f * downx[l][2 * r][c][2] + downx[l][2 * r + 1][c][2]) * 0.25f;
                downsampled[l][r][c][3] = (downx[l][2 * r - 1][c][3] + 2.0f * downx[l][2 * r][c][3] + downx[l][2 * r + 1][c][3]) * 0.25f;
            }
            downsampled[l][0][c][0] = (3.0f * downx[l][0][c][0] + downx[l][1][c][0]) * 0.25f;
            downsampled[l][0][c][1] = (3.0f * downx[l][0][c][1] + downx[l][1][c][1]) * 0.25f;
            downsampled[l][0][c][2] = (3.0f * downx[l][0][c][2] + downx[l][1][c][2]) * 0.25f;
            downsampled[l][0][c][3] = (3.0f * downx[l][0][c][3] + downx[l][1][c][3]) * 0.25f;
        }
    }
    
    #pragma omp parallel private(c)
    for (r = 0; r < input_img->rows; r++)
    {
        for (c = 0; c < input_img->cols; c++)
        {
            interpolated[levels - 1][r][c][0] = downsampled[levels - 1][r][c][0];
            interpolated[levels - 1][r][c][1] = downsampled[levels - 1][r][c][1];
            interpolated[levels - 1][r][c][2] = downsampled[levels - 1][r][c][2];
            interpolated[levels - 1][r][c][3] = downsampled[levels - 1][r][c][3];
        }
    }
    
    for(l = levels - 2; l > -1; l--)
    {
        #pragma omp parallel private(c)
        for(r = 0; r < input_img->rows >> (l + 1); r++)
        {
            for(c = 0; c < (input_img->cols >> l) - 1; c++)
            {
                upsampledx[l][r][c][0] = (interpolated[l + 1][r][(int)(c / 2)][0] + interpolated[l + 1][r][(int)((c + 1) / 2)][0]) / 2.0f;
                upsampledx[l][r][c][1] = (interpolated[l + 1][r][(int)(c / 2)][1] + interpolated[l + 1][r][(int)((c + 1) / 2)][1]) / 2.0f;
                upsampledx[l][r][c][2] = (interpolated[l + 1][r][(int)(c / 2)][2] + interpolated[l + 1][r][(int)((c + 1) / 2)][2]) / 2.0f;
                upsampledx[l][r][c][3] = (interpolated[l + 1][r][(int)(c / 2)][3] + interpolated[l + 1][r][(int)((c + 1) / 2)][3]) / 2.0f;
            }
            upsampledx[l][r][(input_img->cols >> l) - 1][0] = interpolated[l + 1][r][(input_img->cols >> (l + 1)) - 1][0];
            upsampledx[l][r][(input_img->cols >> l) - 1][1] = interpolated[l + 1][r][(input_img->cols >> (l + 1)) - 1][1];
            upsampledx[l][r][(input_img->cols >> l) - 1][2] = interpolated[l + 1][r][(input_img->cols >> (l + 1)) - 1][2];
            upsampledx[l][r][(input_img->cols >> l) - 1][3] = interpolated[l + 1][r][(input_img->cols >> (l + 1)) - 1][3];
        }
        
        #pragma omp parallel private(r)
        for(c = 0; c < input_img->cols >> l; c++)
        {
            for(r = 0; r < (input_img->rows >> l) - 1; r++)
            {
                upsampled[l][r][c][0] = (upsampledx[l][(int)(r / 2)][c][0] + upsampledx[l][(int)((r + 1) / 2)][c][0]) / 2.0f;
                upsampled[l][r][c][1] = (upsampledx[l][(int)(r / 2)][c][1] + upsampledx[l][(int)((r + 1) / 2)][c][1]) / 2.0f;
                upsampled[l][r][c][2] = (upsampledx[l][(int)(r / 2)][c][2] + upsampledx[l][(int)((r + 1) / 2)][c][2]) / 2.0f;
                upsampled[l][r][c][3] = (upsampledx[l][(int)(r / 2)][c][3] + upsampledx[l][(int)((r + 1) / 2)][c][3]) / 2.0f;
                
                interpolated[l][r][c][0] = downsampled[l][r][c][0] + (1.0f - downsampled[l][r][c][3]) * upsampled[l][r][c][0];
                interpolated[l][r][c][1] = downsampled[l][r][c][1] + (1.0f - downsampled[l][r][c][3]) * upsampled[l][r][c][1];
                interpolated[l][r][c][2] = downsampled[l][r][c][2] + (1.0f - downsampled[l][r][c][3]) * upsampled[l][r][c][2];
                interpolated[l][r][c][3] = downsampled[l][r][c][3] + (1.0f - downsampled[l][r][c][3]) * upsampled[l][r][c][3];
            }
            upsampled[l][(input_img->rows >> l) - 1][c][0] = upsampledx[l][(input_img->rows >> (l + 1)) - 1][c][0];
            upsampled[l][(input_img->rows >> l) - 1][c][1] = upsampledx[l][(input_img->rows >> (l + 1)) - 1][c][1];
            upsampled[l][(input_img->rows >> l) - 1][c][2] = upsampledx[l][(input_img->rows >> (l + 1)) - 1][c][2];
            upsampled[l][(input_img->rows >> l) - 1][c][3] = upsampledx[l][(input_img->rows >> (l + 1)) - 1][c][3];
            
            interpolated[l][(input_img->rows >> l) - 1][c][0] = downsampled[l][(input_img->rows >> l) - 1][c][0] + (1.0f - downsampled[l][(input_img->rows >> l) - 1][c][3]) * upsampled[l][(input_img->rows >> l) - 1][c][0];
            interpolated[l][(input_img->rows >> l) - 1][c][1] = downsampled[l][(input_img->rows >> l) - 1][c][1] + (1.0f - downsampled[l][(input_img->rows >> l) - 1][c][3]) * upsampled[l][(input_img->rows >> l) - 1][c][1];
            interpolated[l][(input_img->rows >> l) - 1][c][2] = downsampled[l][(input_img->rows >> l) - 1][c][2] + (1.0f - downsampled[l][(input_img->rows >> l) - 1][c][3]) * upsampled[l][(input_img->rows >> l) - 1][c][2];
            interpolated[l][(input_img->rows >> l) - 1][c][3] = downsampled[l][(input_img->rows >> l) - 1][c][3] + (1.0f - downsampled[l][(input_img->rows >> l) - 1][c][3]) * upsampled[l][(input_img->rows >> l) - 1][c][3];
        }
    }
    
    #pragma omp parallel for private(c)
    for (r = 0; r < input_img->rows; r++)
    {
        for (c = 0; c < input_img->cols; c++)
        {
            output_img->data[r][c][0] = clip(interpolated[0][r][c][0] / interpolated[0][r][c][3], 0.0f, 1.0f);
            output_img->data[r][c][1] = clip(interpolated[0][r][c][1] / interpolated[0][r][c][3], 0.0f, 1.0f);
            output_img->data[r][c][2] = clip(interpolated[0][r][c][2] / interpolated[0][r][c][3], 0.0f, 1.0f);
        }
    }
    
    free(downsampled_data);
    free(downsampled_pointer);
    free(downx_data);
    free(downx_pointer);
    free(interpolated_data);
    free(interpolated_pointer);
    free(upsampled_data);
    free(upsampled_pointer);
    free(upsampledx_data);
    free(upsampledx_pointer);
}

int main(int argc, char **argv)
{
    FLOAT4_IMG *input_img;
    int levels = 10;
    
    if(argc == 1)
        input_img = read_img_rgba("../../images/rgba_small.png");
    else
    {
        if(argc == 3)
            input_img = read_img_rgba(argv[1]);
        else if (argc == 4)
        {
            input_img = read_img_rgba(argv[1]);
            levels = atoi(argv[3]);
        }
        else
        {
            printf("Usage: ./a.out in.png out.png or ./a.out \n");
            return 0;
        }
    }
  
    FLOAT3_IMG *output_img = zeros_rgb(input_img->rows, input_img->cols);
    
    if (input_img->rows >> (levels - 1) < 1 || input_img->cols >> (levels - 1) < 1)
        levels = min((int)(log2(input_img->rows + 1)), (int)(log2(input_img->cols + 1)));
    
    double d1 = wall_time();
    interpolate(input_img, output_img, levels);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
        write_img_rgb(output_img, "test.png");
    else
        write_img_rgb(output_img, argv[2]);
    
    free_img_rgba(input_img);
    free_img_rgb(output_img);
    
    return 0;
}