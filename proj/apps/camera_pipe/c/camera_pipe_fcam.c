#define IS_CAMERA_PIPE True

#include "c_util.h"
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "pfm.h"

real max(real a, real b)
{
    return a > b ? a : b;
}

real min(real a, real b)
{
    return a < b ? a : b;
}

real clip(real input, real lower_bound, real upper_bound)
{
    return min(max(input, lower_bound), upper_bound);
}

void makeLUT(real contrast, int blackLevel, real gamma, real *lut)
{
    int minRaw, maxRaw;
    int i;
    real invRange, a, b, y;
    
    minRaw = 0 + blackLevel;
    maxRaw = 1023;
    
    #pragma omp parallel for
    for (i = 0; i < minRaw + 1; i++)
        lut[i] = 0.0f;
    
    invRange = 1.0f / (maxRaw - minRaw);
    b = 2.0f - pow(2.0f, contrast / 100.0f);
    a = 2.0f - 2.0f * b;
    
    #pragma omp parallel for private(y)
    for (i = minRaw + 1; i < maxRaw + 1; i++)
    {
        y = (float)(i - minRaw) * invRange;
        y = pow(y, 1.0f / gamma);
        
        if (y > 0.5f)
        {
            y = 1.0f - y;
            y = a * y * y + b * y;
            y = 1.0f - y;
        }
        else
            y = a * y * y + b * y;
        
        y = (real)(int)(y * 255.0f + 0.5f);
        
        if (y < 0.0f)
            y = 0.0f;
        if (y > 255.0f)
            y = 255.0f;
        
        lut[i] = y;
    }
    
    #pragma omp parallel for
    for (i = maxRaw + 1; i < 1024; i++)
        lut[i] = 255.0f;
}

void makeColorMatrix(real *colorMatrix, real colorTemp)
{
    real alpha;
    
    alpha = (1.0f / colorTemp - 1.0f / 3200.0f) / (1.0f / 7000.0f - 1.0f / 3200.0f);
    
    colorMatrix[0] = alpha*1.6697f + (1.0f-alpha)*2.2997f;
    colorMatrix[1] = alpha*-0.2693f + (1.0f-alpha)*-0.4478f;
    colorMatrix[2] = alpha*-0.4004f + (1.0f-alpha)*0.1706f;
    colorMatrix[3] = alpha*-42.4346f + (1.0f-alpha)*-39.0923f;

    colorMatrix[4] = alpha*-0.3576f + (1.0f-alpha)*-0.3826f;
    colorMatrix[5] = alpha*1.0615f + (1.0f-alpha)*1.5906f;
    colorMatrix[6] = alpha*1.5949f + (1.0f-alpha)*-0.2080f;
    colorMatrix[7] = alpha*-37.1158f + (1.0f-alpha)*-25.4311f;

    colorMatrix[8] = alpha*-0.2175f + (1.0f-alpha)*-0.0888f;
    colorMatrix[9] = alpha*-1.8751f + (1.0f-alpha)*-0.7344f;
    colorMatrix[10]= alpha*6.9640f + (1.0f-alpha)*2.2832f;
    colorMatrix[11]= alpha*-26.6970f + (1.0f-alpha)*-20.0826f;
}

void demosaic(FLOAT_IMG *input, FLOAT3_IMG *out, real colorTemp, real contrast, int denoise, int blackLevel, real gamma)
{
    const int BLOCK_WIDTH = 40;
    const int BLOCK_HEIGHT = 24;
    int G, GR, R, B, GB;
    int rawWidth, rawHeight, outWidth, outHeight, WIDTH, HEIGHT;
    real gv_r, gvd_r, gh_r, ghd_r;
    real gv_b, gvd_b, gh_b, ghd_b;
    real rp_b, rpd_b, rn_b, rnd_b;
    real bp_r, bpd_r, bn_r, bnd_r;
    real r, g, b;
    int ri, gi, bi;
    
    G = 0;
    GR = 0;
    R = 1;
    B = 2;
    GB = 3;
    
    rawWidth = (int)(input->cols);
    rawHeight = (int)(input->rows);
    outWidth = rawWidth - 32;
    outHeight = rawHeight - 48;
    outWidth = outWidth < (int)(out->cols) ? outWidth : (int)(out->cols);
    outHeight = outHeight < (int)(out->rows) ? outHeight : (int)(out->rows);
    outWidth /= BLOCK_WIDTH;
    outWidth *= BLOCK_WIDTH;
    outHeight /= BLOCK_HEIGHT;
    outHeight *= BLOCK_HEIGHT;
    
    WIDTH = outWidth;
    HEIGHT = outHeight;
    
    real lut[1024];
    makeLUT(contrast, blackLevel, gamma, lut);
    
    real colorMatrix[12];
    makeColorMatrix(colorMatrix, colorTemp);
    
    int bx, by, x, y, k, j;
    
    #pragma omp parallel for private(bx, y, x, gv_r, gvd_r, gh_r, ghd_r, gv_b, gvd_b, gh_b, ghd_b, rp_b, rpd_b, rn_b, rnd_b, bp_r, bpd_r, bn_r, bnd_r, r, g, b, ri, gi, bi)
    for (by = 0; by < outHeight; by += BLOCK_HEIGHT)
    {
        for (bx = 0; bx < outWidth; bx += BLOCK_WIDTH)
        {
            real inBlock[4][BLOCK_HEIGHT / 2 + 4][BLOCK_WIDTH / 2 + 4];
            for (k = 0; k < 4; k++)
                for (y = 0; y < BLOCK_HEIGHT / 2 + 4; y++)
                    for (x = 0; x < BLOCK_WIDTH / 2 + 4; x++)
                        inBlock[k][y][x] = 0.0f;
            
            for (y = 0; y < BLOCK_HEIGHT / 2 + 4; y++)
            {
                for (x = 0; x < BLOCK_WIDTH / 2 + 4; x++)
                {
                    inBlock[GR][y][x] = input->data[by + 2 * y][bx + 2 * x];
                    inBlock[R][y][x] = input->data[by + 2 * y][bx + 2 * x + 1];
                    inBlock[B][y][x] = input->data[by + 2 * y + 1][bx + 2 * x];
                    inBlock[GB][y][x] = input->data[by + 2 * y + 1][bx + 2 * x + 1];
                }
            }
            
            real linear[3][4][BLOCK_HEIGHT / 2 + 4][BLOCK_WIDTH / 2 + 4];
            for (k = 0; k < 3; k++)
                for (j = 0; j < 3; j++)
                    for (y = 0; y < BLOCK_HEIGHT / 2 + 4; y++)
                        for (x = 0; x < BLOCK_WIDTH / 2 + 4; x++)
                            linear[k][j][y][x] = 0.0f;
            
            if (denoise)
            {
                for (y = 1; y < BLOCK_HEIGHT / 2 + 3; y++)
                {
                    for (x = 1; x < BLOCK_WIDTH / 2 + 3; x++)
                    {
                        linear[G][GR][y][x] = min(inBlock[GR][y][x], max(max(inBlock[GR][y - 1][x], inBlock[GR][y + 1][x]), max(inBlock[GR][y][x + 1], inBlock[GR][y][x - 1])));
                        linear[R][R][y][x] = min(inBlock[R][y][x], max(max(inBlock[R][y - 1][x], inBlock[R][y + 1][x]), max(inBlock[R][y][x + 1], inBlock[R][y][x - 1])));
                        linear[B][B][y][x] = min(inBlock[B][y][x], max(max(inBlock[B][y - 1][x], inBlock[B][y + 1][x]), max(inBlock[B][y][x + 1], inBlock[B][y][x - 1])));
                        linear[G][GB][y][x] = min(inBlock[GB][y][x], max(max(inBlock[GB][y - 1][x], inBlock[GB][y + 1][x]), max(inBlock[GB][y][x + 1], inBlock[GB][y][x - 1])));
                    }
                }
            }
            else
            {
                for (y = 1; y < BLOCK_HEIGHT / 2 + 3; y++)
                {
                    for (x = 1; x < BLOCK_HEIGHT / 2 + 3; x++)
                    {
                        linear[G][GR][y][x] = inBlock[GR][y][x];
                        linear[R][R][y][x] = inBlock[R][y][x];
                        linear[B][B][y][x] = inBlock[B][y][x];
                        linear[G][GB][y][x] = inBlock[GB][y][x];
                    }
                }
            }
            
            for (y = 1; y < BLOCK_HEIGHT / 2 + 3; y++)
            {
                for (x = 1; x < BLOCK_WIDTH / 2 + 3; x++)
                {
                    gv_r = (linear[G][GB][y-1][x] + linear[G][GB][y][x])/2.0f;
                    gvd_r = linear[G][GB][y-1][x] - linear[G][GB][y][x];
                    gvd_r = abs(gvd_r);
                    gh_r = (linear[G][GR][y][x] + linear[G][GR][y][x+1])/2.0f;
                    ghd_r = linear[G][GR][y][x] - linear[G][GR][y][x+1];
                    ghd_r = abs(ghd_r);
                    linear[G][R][y][x] = ghd_r < gvd_r ? gh_r : gv_r;

                    gv_b = (linear[G][GR][y+1][x] + linear[G][GR][y][x])/2.0f;
                    gvd_b = linear[G][GR][y+1][x] - linear[G][GR][y][x];
                    gvd_b = abs(gvd_b);
                    gh_b = (linear[G][GB][y][x] + linear[G][GB][y][x-1])/2.0f;
                    ghd_b = linear[G][GB][y][x] - linear[G][GB][y][x-1];
                    ghd_b = abs(ghd_b);
                    linear[G][B][y][x] = ghd_b < gvd_b ? gh_b : gv_b;
                }
            }
            
            for (y = 1; y < BLOCK_HEIGHT / 2 + 3; y++)
            {
                for (x = 1; x < BLOCK_WIDTH / 2 + 3; x++)
                {
                    linear[R][GR][y][x] = ((linear[R][R][y][x-1] + linear[R][R][y][x])/2.0f + linear[G][GR][y][x] - (linear[G][R][y][x-1] + linear[G][R][y][x])/2.0);
                    linear[B][GR][y][x] = ((linear[B][B][y-1][x] + linear[B][B][y][x])/2.0f + linear[G][GR][y][x] - (linear[G][B][y-1][x] + linear[G][B][y][x])/2.0);
                    linear[R][GB][y][x] = ((linear[R][R][y][x] + linear[R][R][y+1][x])/2.0f + linear[G][GB][y][x] - (linear[G][R][y][x] + linear[G][R][y+1][x])/2.0);
                    linear[B][GB][y][x] = ((linear[B][B][y][x] + linear[B][B][y][x+1])/2.0f + linear[G][GB][y][x] - (linear[G][B][y][x] + linear[G][B][y][x+1])/2.0);
                }
            }
            
            for (y = 1; y < BLOCK_HEIGHT / 2 + 3; y++)
            {
                for (x = 1; x < BLOCK_WIDTH / 2 + 3; x++)
                {
                    rp_b = ((linear[R][R][y+1][x-1] + linear[R][R][y][x])/2.0f + linear[G][B][y][x] - (linear[G][R][y+1][x-1] + linear[G][R][y][x])/2.0f);
                    rpd_b = linear[R][R][y+1][x-1] - linear[R][R][y][x];
                    rpd_b = abs(rpd_b);
                    rn_b = ((linear[R][R][y][x-1] + linear[R][R][y+1][x])/2.0f + linear[G][B][y][x] - (linear[G][R][y][x-1] + linear[G][R][y+1][x])/2.0f);
                    rnd_b = linear[R][R][y][x-1] - linear[R][R][y+1][x];
                    rnd_b = abs(rnd_b);
                    linear[R][B][y][x] = rpd_b < rnd_b ? rp_b : rn_b;

                    bp_r = ((linear[B][B][y-1][x+1] + linear[B][B][y][x])/2.0f + linear[G][R][y][x] - (linear[G][B][y-1][x+1] + linear[G][B][y][x])/2.0f);
                    bpd_r = linear[B][B][y-1][x+1] - linear[B][B][y][x];
                    bpd_r = abs(bpd_r);
                    bn_r = ((linear[B][B][y][x+1] + linear[B][B][y-1][x])/2.0f + linear[G][R][y][x] - (linear[G][B][y][x+1] + linear[G][B][y-1][x])/2.0f);
                    bnd_r = linear[B][B][y][x+1] - linear[B][B][y-1][x];
                    bnd_r = abs(bnd_r);
                    linear[B][R][y][x] = bpd_r < bnd_r ? bp_r : bn_r;
                }
            }
            
            for (y = 2; y < BLOCK_HEIGHT / 2 + 2; y++)
            {
                for (x = 2; x < BLOCK_WIDTH / 2 + 2; x++)
                {
                    r = colorMatrix[0]*linear[R][GR][y][x] + colorMatrix[1]*linear[G][GR][y][x] + colorMatrix[2]*linear[B][GR][y][x] + colorMatrix[3];
                    g = colorMatrix[4]*linear[R][GR][y][x] + colorMatrix[5]*linear[G][GR][y][x] + colorMatrix[6]*linear[B][GR][y][x] + colorMatrix[7];
                    b = colorMatrix[8]*linear[R][GR][y][x] + colorMatrix[9]*linear[G][GR][y][x] + colorMatrix[10]*linear[B][GR][y][x] + colorMatrix[11];
                    
                    r = clip(r, 0.0f, 1023.0f);
                    ri = (int)(r + 0.5f);
                    g = clip(g, 0.0f, 1023.0f);
                    gi = (int)(g + 0.5f);
                    b = clip(b, 0.0f, 1023.0f);
                    bi = (int)(b + 0.5f);
                    
                    out->data[by + (y - 2) * 2][bx + (x - 2) * 2][0] = lut[ri];
                    out->data[by + (y - 2) * 2][bx + (x - 2) * 2][1] = lut[gi];
                    out->data[by + (y - 2) * 2][bx + (x - 2) * 2][2] = lut[bi];
                    
                    r = colorMatrix[0]*linear[R][R][y][x] + colorMatrix[1]*linear[G][R][y][x] + colorMatrix[2]*linear[B][R][y][x] + colorMatrix[3];
                    g = colorMatrix[4]*linear[R][R][y][x] + colorMatrix[5]*linear[G][R][y][x] + colorMatrix[6]*linear[B][R][y][x] + colorMatrix[7];
                    b = colorMatrix[8]*linear[R][R][y][x] + colorMatrix[9]*linear[G][R][y][x] + colorMatrix[10]*linear[B][R][y][x] + colorMatrix[11];
                    
                    r = clip(r, 0.0f, 1023.0f);
                    ri = (int)(r + 0.5f);
                    g = clip(g, 0.0f, 1023.0f);
                    gi = (int)(g + 0.5f);
                    b = clip(b, 0.0f, 1023.0f);
                    bi = (int)(b + 0.5f);
                    
                    out->data[by + (y - 2) * 2][bx + (x - 2) * 2 + 1][0] = lut[ri];
                    out->data[by + (y - 2) * 2][bx + (x - 2) * 2 + 1][1] = lut[gi];
                    out->data[by + (y - 2) * 2][bx + (x - 2) * 2 + 1][2] = lut[bi];
                    
                    r = colorMatrix[0]*linear[R][B][y][x] + colorMatrix[1]*linear[G][B][y][x] + colorMatrix[2]*linear[B][B][y][x] + colorMatrix[3];
                    g = colorMatrix[4]*linear[R][B][y][x] + colorMatrix[5]*linear[G][B][y][x] + colorMatrix[6]*linear[B][B][y][x] + colorMatrix[7];
                    b = colorMatrix[8]*linear[R][B][y][x] + colorMatrix[9]*linear[G][B][y][x] + colorMatrix[10]*linear[B][B][y][x] + colorMatrix[11];
                    
                    r = clip(r, 0.0f, 1023.0f);
                    ri = (int)(r + 0.5f);
                    g = clip(g, 0.0f, 1023.0f);
                    gi = (int)(g + 0.5f);
                    b = clip(b, 0.0f, 1023.0f);
                    bi = (int)(b + 0.5f);
                    
                    out->data[by + (y - 2) * 2 + 1][bx + (x - 2) * 2][0] = lut[ri];
                    out->data[by + (y - 2) * 2 + 1][bx + (x - 2) * 2][1] = lut[gi];
                    out->data[by + (y - 2) * 2 + 1][bx + (x - 2) * 2][2] = lut[bi];
                    
                    r = colorMatrix[0]*linear[R][GB][y][x] + colorMatrix[1]*linear[G][GB][y][x] + colorMatrix[2]*linear[B][GB][y][x] + colorMatrix[3];
                    g = colorMatrix[4]*linear[R][GB][y][x] + colorMatrix[5]*linear[G][GB][y][x] + colorMatrix[6]*linear[B][GB][y][x] + colorMatrix[7];
                    b = colorMatrix[8]*linear[R][GB][y][x] + colorMatrix[9]*linear[G][GB][y][x] + colorMatrix[10]*linear[B][GB][y][x] + colorMatrix[11];
                    
                    r = clip(r, 0.0f, 1023.0f);
                    ri = (int)(r + 0.5f);
                    g = clip(g, 0.0f, 1023.0f);
                    gi = (int)(g + 0.5f);
                    b = clip(b, 0.0f, 1023.0f);
                    bi = (int)(b + 0.5f);
                    
                    out->data[by + (y - 2) * 2 + 1][bx + (x - 2) * 2 + 1][0] = lut[ri];
                    out->data[by + (y - 2) * 2 + 1][bx + (x - 2) * 2 + 1][1] = lut[gi];
                    out->data[by + (y - 2) * 2 + 1][bx + (x - 2) * 2 + 1][2] = lut[bi];
                }
            }
        }
    }
}

void camera_pipe(FLOAT_IMG *input_img, FLOAT3_IMG *output_img)
{
    int r, c;
    
    demosaic(input_img, output_img, 3700.0f, 50.0f, 1, 25, 2.0f);
    #pragma omp parallel for private(c)
    for (r = 0; r < output_img->rows; r++)
    {
        for (c = 0; c < output_img->cols; c++)
        {
            output_img->data[r][c][0] /= 256.0f;
            output_img->data[r][c][1] /= 256.0f;
            output_img->data[r][c][2] /= 256.0f;
        }
    }
}

int main(int argc, char **argv)
{
    int w, h;
    float *depth;
    if (argc == 1)
        depth = read_pfm_file("../../images/bayer_small.pfm", &w, &h);
    else
        if (argc == 3)
            depth = read_pfm_file(argv[1], &w, &h);
        else
        {
            printf("Usage: ./a.out in.pfm out.png or ./a.out\n");
            return 0;
        }
    
    FLOAT_IMG *input_img = zeros_gray(h, w);
    int r, c;
    int index;
    #pragma omp parallel for private(c, index)
    for (r = 0; r < h; r++)
    {
        for (c = 0; c < w; c++)
        {
            index = w * h - w * (r + 1) + c;
            input_img->data[r][c] = (real)(depth[index]);
        }
    }
    
    free(depth);
    
    FLOAT3_IMG *output_img = zeros_rgb(((h - 24) / 32) * 32, ((w - 32) / 32) * 32);
    
    double d1 = wall_time();
    camera_pipe(input_img, output_img);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
    write_img_rgb(output_img, "test.png");
    else
    write_img_rgb(output_img, argv[2]);
    
    free_img_gray(input_img);
    free_img_rgb(output_img);

    return 0;
}
