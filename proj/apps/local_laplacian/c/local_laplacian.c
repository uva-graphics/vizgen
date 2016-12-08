#include "c_util.h"
#include <stdio.h>
#include <omp.h>

float max(float a, float b)
{
    return a > b ? a : b;
}

float min(float a, float b)
{
    return a < b ? a : b;
}

float clip(float input, float lower_bound, float upper_bound)
{
    return min(max(input, lower_bound), upper_bound);
}

void local_laplacian(FLOAT3_IMG *input_img, FLOAT3_IMG *output_img)
{
    int levels = 8;
    float alpha = 1.0f / 7.0f;
    float beta = 1.0f;
    int J = 8;
    float eps = 0.01f;
    
    int r, c;
    int i, j;
    
    FLOAT_IMG *gray = zeros_gray(input_img->rows, input_img->cols);
    
    #pragma omp parallel for private(c)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            gray->data[r][c] = 0.299f * input_img->data[r][c][0] + 0.587f * input_img->data[r][c][1] + 0.114f * input_img->data[r][c][2];
        }
    }
    
    int w = input_img->rows;
    int h = input_img->cols;
    
    real ****gPyramid;
    real ****lPyramid;
    real ***inGPyramid;
    real ***outLPyramid;
    real ***outGPyramid;
    
    real *gPyramid_data = malloc(input_img->rows * input_img->cols * levels * J * sizeof(real));
    real **gPyramid_pointer = malloc(input_img->rows * sizeof(real ***) + input_img->rows * input_img->cols * sizeof(real **) + input_img->rows * input_img->cols * levels * sizeof(real *));
    gPyramid = (real ****)gPyramid_pointer;
    
    real *lPyramid_data = malloc(input_img->rows * input_img->cols * levels * J * sizeof(real));
    real **lPyramid_pointer = malloc(input_img->rows * sizeof(real ***) + input_img->rows * input_img->cols * sizeof(real **) + input_img->rows * input_img->cols * levels * sizeof(real *));
    lPyramid = (real ****)lPyramid_pointer;
    
    real *inGPyramid_data = malloc(input_img->rows * input_img->cols * J * sizeof(real));
    real **inGPyramid_pointer = malloc(input_img->rows * sizeof(real **) + input_img->rows * input_img->cols * sizeof(real *));
    inGPyramid = (real ***)inGPyramid_pointer;
    
    real *outLPyramid_data = malloc(input_img->rows * input_img->cols * J * sizeof(real));
    real **outLPyramid_pointer = malloc(input_img->rows * sizeof(real **) + input_img->rows * input_img->cols * sizeof(real *));
    outLPyramid = (real ***)outLPyramid_pointer;
    
    real *outGPyramid_data = malloc(input_img->rows * input_img->cols * J * sizeof(real));
    real **outGPyramid_pointer = malloc(input_img->rows * sizeof(real **) + input_img->rows * input_img->cols * sizeof(real *));
    outGPyramid = (real ***)outGPyramid_pointer;
    
    #pragma omp parallel for private(c, i, j)
    for(r = 0; r < input_img->rows; r++)
    {
        gPyramid[r] = (real ***)&gPyramid_pointer[input_img->rows + r * input_img->cols];
        lPyramid[r] = (real ***)&lPyramid_pointer[input_img->rows + r * input_img->cols];
        inGPyramid[r] = (real **)&inGPyramid_pointer[input_img->rows + r * input_img->cols];
        outLPyramid[r] = (real **)&outLPyramid_pointer[input_img->rows + r * input_img->cols];
        outGPyramid[r] = (real **)&outGPyramid_pointer[input_img->rows + r * input_img->cols];
    
        for(c = 0; c < input_img->cols; c++)
        {
            gPyramid[r][c] = (real **)&gPyramid_pointer[input_img->rows + input_img->rows * input_img->cols + r * input_img->cols * levels + c * levels];
            lPyramid[r][c] = (real **)&lPyramid_pointer[input_img->rows + input_img->rows * input_img->cols + r * input_img->cols * levels + c * levels];
            inGPyramid[r][c] = &inGPyramid_data[J * (r * input_img->cols + c)];
            outLPyramid[r][c] = &outLPyramid_data[J * (r * input_img->cols + c)];
            outGPyramid[r][c] = &outGPyramid_data[J * (r * input_img->cols + c)];
        
            for(i = 0; i < levels; i++)
            {
                gPyramid[r][c][i] = &gPyramid_data[J * (r * input_img->cols * levels + c * levels + i)];
                lPyramid[r][c][i] = &lPyramid_data[J * (r * input_img->cols * levels + c * levels + i)];
        
                for(j = 0; j < J; j++)
                {
                    gPyramid[r][c][i][j] = 0.0f;
                    lPyramid[r][c][i][j] = 0.0f;
                }
            }
        
            for(j = 0; j < J; j++)
            {
                inGPyramid[r][c][j] = 0.0f;
                outLPyramid[r][c][j] = 0.0f;
                outGPyramid[r][c][j] = 0.0f;
            }
        }
    }
    
    int k;
    float level;
    int idx;
    float fx;
    
    #pragma omp parallel for private(r, c, level, idx, fx)
    for(k = 0; k < levels; k++)
    {
        for(r = 0; r < input_img->rows; r++)
        {
            for(c = 0; c < input_img->cols; c++)
            {
                level = k * (1.0f / (levels - 1.0f));
                idx = gray->data[r][c] * (levels - 1) * 256;
                idx = clip(idx, 0, 256 * (levels - 1));
                fx = (idx - 256.0f * k) / 256.0f;
                gPyramid[r][c][k][0] = beta * (gray->data[r][c] - level) + level + alpha * fx * exp(-fx * fx / 2.0f);
            }
        }
    }
    
    #pragma omp parallel for private(c)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            inGPyramid[r][c][0] = gray->data[r][c];
        }
    }
    
    int idx_r1, idx_r2, idx_c1, idx_c2;
    int w_j, h_j;
    
    for(j = 1; j < J; j++)
    {
        w_j = input_img->rows >> j;
        h_j = input_img->cols >> j;
        
        #pragma omp parallel for private(c, i, idx_r1, idx_r2, idx_c1, idx_c2)
        for(r = 0; r < w_j; r++)
        {
            for(c = 1; c < h_j; c++)
            {
                idx_r1 = 2 * r - 1;
                idx_r2 = 2 * r + 2;
                idx_c1 = 2 * c - 1;
                idx_c2 = 2 * c + 2;
                
                if (r == 0)
                    idx_r1 = 0;
                if (r == w_j - 1)
                    idx_r2 = w_j * 2 - 1;
                if (c == 0)
                    idx_c1 = 0;
                if (c == h_j - 1)
                    idx_c2 = h_j * 2 - 1;
                
                for (i = 0; i < levels; i++)
                {
                    gPyramid[r][c][i][j] = (gPyramid[idx_r1][idx_c1][i][j - 1] + 3.0f * gPyramid[idx_r1][2 * c][i][j - 1] + 3.0f * gPyramid[idx_r1][2 * c + 1][i][j - 1] + gPyramid[idx_r1][idx_c2][i][j - 1] + 3.0f * gPyramid[2 * r][idx_c1][i][j - 1] + 9.0f * gPyramid[2 * r][2 * c][i][j - 1] + 9.0f * gPyramid[2 * r][2 * c + 1][i][j - 1] + 3.0f * gPyramid[2 * r][idx_c2][i][j - 1] + 3.0f * gPyramid[2 * r + 1][idx_c1][i][j - 1] + 9.0f * gPyramid[2 * r + 1][2 * c][i][j - 1] + 9.0f * gPyramid[2 * r + 1][2 * c + 1][i][j - 1] + 3.0f * gPyramid[2 * r + 1][idx_c2][i][j - 1] + gPyramid[idx_r2][idx_c1][i][j - 1] + 3.0f * gPyramid[idx_r2][2 * c][i][j - 1] + 3.0f * gPyramid[idx_r2][2 * c + 1][i][j - 1] + gPyramid[idx_r2][idx_c2][i][j - 1]) / 64.0f;
                    if (r == 49 && c == 78 && i == 7 && j == 1)
                        r = 49;
                }
                inGPyramid[r][c][j] = (inGPyramid[idx_r1][idx_c1][j - 1] + 3.0f * inGPyramid[idx_r1][2 * c][j - 1] + 3.0f * inGPyramid[idx_r1][2 * c + 1][j - 1] + inGPyramid[idx_r1][idx_c2][j - 1] + 3.0f * inGPyramid[2 * r][idx_c1][j - 1] + 9.0f * inGPyramid[2 * r][2 * c][j - 1] + 9.0f * inGPyramid[2 * r][2 * c + 1][j - 1] + 3.0f * inGPyramid[2 * r][idx_c2][j - 1] + 3.0f * inGPyramid[2 * r + 1][idx_c1][j - 1] + 9.0f * inGPyramid[2 * r + 1][2 * c][j - 1] + 9.0f * inGPyramid[2 * r + 1][2 * c + 1][j - 1] + 3.0f * inGPyramid[2 * r + 1][idx_c2][j - 1] + inGPyramid[idx_r2][idx_c1][j - 1] + 3.0f * inGPyramid[idx_r2][2 * c][j - 1] + 3.0f * inGPyramid[idx_r2][2 * c + 1][j - 1] + inGPyramid[idx_r2][idx_c2][j - 1]) / 64.0f;
            }
        }
    }
    
    #pragma omp parallel for private(r, c)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            for (k = 0; k < levels; k++)
            {
                lPyramid[r][c][k][J - 1] = gPyramid[r][c][k][J - 1];
            }
        }
    }
    
    j = J - 1;
    w_j = input_img->rows >> j;
    h_j = input_img->cols >> j;
    
    int li;
    float lf;
    
    #pragma omp parallel for private(c, level, li, lf)
    for (r = 0; r < w_j; r++)
    {
        for (c = 0; c < h_j; c++)
        {
            level = inGPyramid[r][c][j] * (levels - 1.0f);
            li = clip((int)level, 0, levels - 2);
            lf = level - li;
            outLPyramid[r][c][j] = (1.0f - lf) * lPyramid[r][c][li][j] + lf * lPyramid[r][c][li + 1][j];
        }
    }
    
    #pragma omp parallel for private(c)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            outGPyramid[r][c][J - 1] = outLPyramid[r][c][J - 1];
        }
    }
    
    for(j = J - 2; j > -1; j--)
    {
        w_j = input_img->rows >> j;
        h_j = input_img->cols >> j;
        
        #pragma omp parallel for private(c, i, idx_r1, idx_c1, level, li, lf)
        for (r = 0; r < w_j; r++)
        {
            for (c = 0; c < h_j; c++)
            {
                idx_r1 = (int)((r + 1) / 2) - 1;
                idx_c1 = (int)((c + 1) / 2) - 1;
                
                if (r == 0)
                    idx_r1 = 0;
                if (c == 0)
                    idx_c1 = 0;
                
                for (i = 0; i < levels; i++)
                {
                    lPyramid[r][c][i][j] = gPyramid[r][c][i][j] - (gPyramid[idx_r1][idx_c1][i][j + 1] + 3.0f * gPyramid[idx_r1][(int)(c / 2)][i][j + 1] + 3.0f * gPyramid[(int)(r / 2)][idx_c1][i][j + 1] + 9.0f * gPyramid[(int)(r / 2)][(int)(c / 2)][i][j + 1]) / 16.0f;
                }
                
                level = inGPyramid[r][c][j] * (levels - 1.0f);
                li = clip((int)level, 0, levels - 2);
                lf = level - li;
                outLPyramid[r][c][j] = (1.0f - lf) * lPyramid[r][c][li][j] + lf * lPyramid[r][c][li + 1][j];
                
                outGPyramid[r][c][j] = outLPyramid[r][c][j] + (outGPyramid[idx_r1][idx_c1][j + 1] + 3.0f * outGPyramid[idx_r1][(int)(c / 2)][j + 1] + 3.0f * outGPyramid[(int)(r / 2)][idx_c1][j + 1] + 9.0f * outGPyramid[(int)(r / 2)][(int)(c / 2)][j + 1]) / 16.0f;
            }
        }
    }
    
    #pragma omp parallel for private(c)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            output_img->data[r][c][0] = clip(outGPyramid[r][c][0] * (input_img->data[r][c][0] + eps) / (gray->data[r][c] + eps), 0, 1);
            output_img->data[r][c][1] = clip(outGPyramid[r][c][0] * (input_img->data[r][c][1] + eps) / (gray->data[r][c] + eps), 0, 1);
            output_img->data[r][c][2] = clip(outGPyramid[r][c][0] * (input_img->data[r][c][2] + eps) / (gray->data[r][c] + eps), 0, 1);
        }
    }
    
    free_img_gray(gray);
    
    free(gPyramid_data);
    free(gPyramid_pointer);
    free(lPyramid_data);
    free(lPyramid_pointer);
    free(inGPyramid_data);
    free(inGPyramid_pointer);
    free(outLPyramid_data);
    free(outLPyramid_pointer);
    free(outGPyramid_data);
    free(outGPyramid_pointer);
}

int main(int argc, char **argv)
{
    FLOAT3_IMG *input_img;
    
    if(argc == 1)
    input_img = read_img_rgb("../../images/small_temple_rgb.png");
    else
    {
    if(argc == 3)
        input_img = read_img_rgb(argv[1]);
    else
    {
        printf("Usage: ./a.out in.png out.png or ./a.out\n");
        return 0;
    }
    }
    
    FLOAT3_IMG *output_img = zeros_rgb(input_img->rows, input_img->cols);
    
    double d1 = wall_time();
    local_laplacian(input_img, output_img);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
    write_img_rgb(output_img, "test.png");
    else
    write_img_rgb(output_img, argv[2]);
    
    free_img_rgb(input_img);
    free_img_rgb(output_img);
    
    return 0;
}