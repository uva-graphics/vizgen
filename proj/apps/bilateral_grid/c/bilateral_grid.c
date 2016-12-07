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

void bilateral_grid(FLOAT_IMG *input_img, FLOAT_IMG *output_img)
{
    float r_sigma = 0.1f;
    int s_sigma = 8;
    
    real ****histogram;
    real ****blurx;
    real ****blury;
    real ****blurz;
    
    int w = (int)(input_img->rows / s_sigma) + 1;
    int h = (int)(input_img->cols / s_sigma) + 1;
    int d = (int)(1.0f / r_sigma) + 1;
    
    real *histogram_data = malloc(w * h * d * 2 * sizeof(real));
    real **histogram_pointer = malloc(w * sizeof(real ***) + w * h * sizeof(real **) + w * h * d * sizeof(real *));
    histogram = (real ****)histogram_pointer;
    
    real *blurx_data = malloc(w * h * d * 2 * sizeof(real));
    real **blurx_pointer = malloc(w * sizeof(real ***) + w * h * sizeof(real **) + w * h * d * sizeof(real *));
    blurx = (real ****)blurx_pointer;
    
    real *blury_data = malloc(w * h * d * 2 * sizeof(real));
    real **blury_pointer = malloc(w * sizeof(real ***) + w * h * sizeof(real **) + w * h * d * sizeof(real *));
    blury = (real ****)blury_pointer;
    
    real *blurz_data = malloc(w * h * d * 2 * sizeof(real));
    real **blurz_pointer = malloc(w * sizeof(real ***) + w * h * sizeof(real **) + w * h * d * sizeof(real *));
    blurz = (real ****)blurz_pointer;
    
    int r, c, k;
    
    #pragma omp parallel for private(c, k)
    for(r = 0; r < w; r++)
    {
	histogram[r] = (real ***)&histogram_pointer[w + r * h];
	blurx[r] = (real ***)&blurx_pointer[w + r * h];
	blury[r] = (real ***)&blury_pointer[w + r * h];
	blurz[r] = (real ***)&blurz_pointer[w + r * h];
	
	for(c = 0; c < h; c++)
	{
	    histogram[r][c] = (real **)&histogram_pointer[w + w * h + r * h * d + c * d];
	    blurx[r][c] = (real **)&blurx_pointer[w + w * h + r * h * d + c * d];
	    blury[r][c] = (real **)&blury_pointer[w + w * h + r * h * d + c * d];
	    blurz[r][c] = (real **)&blurz_pointer[w + w * h + r * h * d + c * d];
	    
	    for(k = 0; k < d; k++)
	    {
		histogram[r][c][k] = &histogram_data[2 * (r * h * d + c * d + k)];
		blurx[r][c][k] = &blurx_data[2 * (r * h * d + c * d + k)];
		blury[r][c][k] = &blury_data[2 * (r * h * d + c * d + k)];
		blurz[r][c][k] = &blurz_data[2 * (r * h * d + c * d + k)];
		
		histogram[r][c][k][0] = 0.0f;
		histogram[r][c][k][1] = 0.0f;
		
		blurx[r][c][k][0] = 0.0f;
		blurx[r][c][k][1] = 0.0f;
		
		blury[r][c][k][0] = 0.0f;
		blury[r][c][k][1] = 0.0f;
		
		blurz[r][c][k][0] = 0.0f;
		blurz[r][c][k][1] = 0.0f;
	    }
	}
    }
    
    real ***interpolated;
    real *interpolated_data = malloc(input_img->rows * input_img->cols * 2 * sizeof(real));
    real **interpolated_pointer = malloc(input_img->rows * sizeof(real **) + input_img->rows * input_img->cols * sizeof(real *));
    interpolated = (real ***)interpolated_pointer;
    
    #pragma omp parallel for private(c)
    for(r = 0; r < input_img->rows; r++)
    {
	interpolated[r] = (real **)&interpolated_pointer[input_img->rows + r * input_img->cols];
	
	for(c = 0; c < input_img->cols; c++)
	{
	    interpolated[r][c] = &interpolated_data[2 * (r * input_img->cols + c)];
	    interpolated[r][c][0] = 0.0f;
	    interpolated[r][c][1] = 0.0f;
	}
    }
    
    float val;
    int zi;
    int xi;
    int yi;
    
    #pragma omp parallel for private(c, val, xi, yi, zi)
    for(r = 1; r < input_img->rows - 1; r++)
    {
	for(c = 1; c < input_img->cols - 1; c++)
	{
	    val = clip(input_img->data[r][c], 0, 1);
	    zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	    
	    xi = (int)((float)(r + (float)s_sigma / 2.0f) / (float)s_sigma);
	    yi = (int)((float)(c + (float)s_sigma / 2.0f) / (float)s_sigma);
	    
	    histogram[xi][yi][zi][0] += val;
	    histogram[xi][yi][zi][1] += 1.0f;
	}
	
	val = clip(input_img->data[r][0], 0, 1);
	zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	    
	xi = (int)((float)(r + (float)s_sigma / 2.0f) / (float)s_sigma);
	    
	histogram[xi][0][zi][0] += val * 5.0f;
	histogram[xi][0][zi][1] += 5.0f;
	
	val = clip(input_img->data[r][input_img->cols - 1], 0, 1);
	zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	    
	histogram[xi][h - 1][zi][0] += val * 5.0f;
	histogram[xi][h - 1][zi][1] += 5.0f;
    }
    
    #pragma omp parallel for private(val, yi, zi)
    for(c = 1; c < input_img->cols - 1; c++)
    {
	val = clip(input_img->data[0][c], 0, 1);
	zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	    
	yi = (int)((float)(c + (float)s_sigma / 2.0f) / (float)s_sigma);
	    
	histogram[0][yi][zi][0] += val * 5.0f;
	histogram[0][yi][zi][1] += 5.0f;
	
	val = clip(input_img->data[input_img->rows - 1][c], 0, 1);
	zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	    
	histogram[w - 1][yi][zi][0] += val * 5.0f;
	histogram[w - 1][yi][zi][1] += 5.0f;
    }
    
    val = clip(input_img->data[0][0], 0, 1);
    zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	      
    histogram[0][0][zi][0] += val * 17.0f;
    histogram[0][0][zi][1] += 17.0f;
    
    val = clip(input_img->data[0][input_img->cols - 1], 0, 1);
    zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	      
    histogram[0][h - 1][zi][0] += val * 17.0f;
    histogram[0][h - 1][zi][1] += 17.0f;
    
    val = clip(input_img->data[input_img->rows - 1][0], 0, 1);
    zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	      
    histogram[w - 1][0][zi][0] += val * 17.0f;
    histogram[w - 1][0][zi][1] += 17.0f;
    
    val = clip(input_img->data[input_img->rows - 1][input_img->cols - 1], 0, 1);
    zi = (int)(val * (1.0f / r_sigma) + 0.5f);
	      
    histogram[w - 1][h - 1][zi][0] += val * 17.0f;
    histogram[w - 1][h - 1][zi][1] += 17.0f;

    #pragma omp parallel for private(r, c)
    for(k = 2; k < d - 2; k++)
    {
	for(r = 0; r < w; r++)
	{
	    for(c = 0; c < h; c++)
	    {
		blurz[r][c][k][0] = histogram[r][c][k - 2][0] + 4.0f * histogram[r][c][k - 1][0] + 6.0f * histogram[r][c][k][0] + 4.0f * histogram[r][c][k + 1][0] + histogram[r][c][k + 2][0];
		blurz[r][c][k][1] = histogram[r][c][k - 2][1] + 4.0f * histogram[r][c][k - 1][1] + 6.0f * histogram[r][c][k][1] + 4.0f * histogram[r][c][k + 1][1] + histogram[r][c][k + 2][1];
	    }
	}
    }
    
    #pragma omp parallel for private(c)
    for(r = 0; r < w; r++)
    {
	for(c = 0; c < h; c++)
	{
	    blurz[r][c][0][0] = 6.0f * histogram[r][c][0][0] + 4.0f * histogram[r][c][1][0] + histogram[r][c][2][0];
	    blurz[r][c][0][1] = 6.0f * histogram[r][c][0][1] + 4.0f * histogram[r][c][1][1] + histogram[r][c][2][1];
	    
	    blurz[r][c][1][0] = 4.0f * histogram[r][c][0][0] + 6.0f * histogram[r][c][1][0] + 4.0f * histogram[r][c][2][0] + histogram[r][c][3][0];
	    blurz[r][c][1][1] = 4.0f * histogram[r][c][0][1] + 6.0f * histogram[r][c][1][1] + 4.0f * histogram[r][c][2][1] + histogram[r][c][3][1];
	    
	    blurz[r][c][d - 2][0] = histogram[r][c][d - 4][0] + 4.0f * histogram[r][c][d - 3][0] + 6.0f * histogram[r][c][d - 2][0] + 4.0f * histogram[r][c][d - 1][0];
	    blurz[r][c][d - 2][1] = histogram[r][c][d - 4][1] + 4.0f * histogram[r][c][d - 3][1] + 6.0f * histogram[r][c][d - 2][1] + 4.0f * histogram[r][c][d - 1][1];
	    
	    blurz[r][c][d - 1][0] = histogram[r][c][d - 3][0] + 4.0f * histogram[r][c][d - 2][0] + 6.0f * histogram[r][c][d - 1][0];
	    blurz[r][c][d - 1][1] = histogram[r][c][d - 3][1] + 4.0f * histogram[r][c][d - 2][1] + 6.0f * histogram[r][c][d - 1][1];
	}
    }
    
    #pragma omp parallel for private(r, c)
    for(k = 0; k < d; k++)
    {
	for(c = 2; c < h - 2; c++)
	{
	    for(r = 0; r < w; r++)
	    {
		blury[r][c][k][0] = blurz[r][c - 2][k][0] + 4.0f * blurz[r][c - 1][k][0] + 6.0f * blurz[r][c][k][0] + 4.0f * blurz[r][c + 1][k][0] + blurz[r][c + 2][k][0];
		blury[r][c][k][1] = blurz[r][c - 2][k][1] + 4.0f * blurz[r][c - 1][k][1] + 6.0f * blurz[r][c][k][1] + 4.0f * blurz[r][c + 1][k][1] + blurz[r][c + 2][k][1];
	    }
	}
    }
    
    #pragma omp parallel for private(r)
    for(k = 0; k < d; k++)
    {
	for(r = 0; r < w; r++)
	{
	    blury[r][0][k][0] = 6.0f * blurz[r][0][k][0] + 4.0f * blurz[r][1][k][0] + blurz[r][2][k][0];
	    blury[r][0][k][1] = 6.0f * blurz[r][0][k][1] + 4.0f * blurz[r][1][k][1] + blurz[r][2][k][1];
	    
	    blury[r][1][k][0] = 4.0f * blurz[r][0][k][0] + 6.0f * blurz[r][1][k][0] + 4.0f * blurz[r][2][k][0] + blurz[r][3][k][0];
	    blury[r][1][k][1] = 4.0f * blurz[r][0][k][1] + 6.0f * blurz[r][1][k][1] + 4.0f * blurz[r][2][k][1] + blurz[r][3][k][1];

	    blury[r][h - 2][k][0] = blurz[r][h - 4][k][0] + 4.0f * blurz[r][h - 3][k][0] + 6.0f * blurz[r][h - 2][k][0] + 4.0f * blurz[r][h - 1][k][0];
	    blury[r][h - 2][k][1] = blurz[r][h - 4][k][1] + 4.0f * blurz[r][h - 3][k][1] + 6.0f * blurz[r][h - 2][k][1] + 4.0f * blurz[r][h - 1][k][1];

	    blury[r][h - 1][k][0] = blurz[r][h - 3][k][0] + 4.0f * blurz[r][h - 2][k][0] + 6.0f * blurz[r][h - 1][k][0];
	    blury[r][h - 1][k][1] = blurz[r][h - 3][k][1] + 4.0f * blurz[r][h - 2][k][1] + 6.0f * blurz[r][h - 1][k][1];
	}
    }
    
    #pragma omp parallel for private(r, c)
    for(k = 0; k < d; k++)
    {
	for(r = 2; r < w - 2; r++)
	{
	    for(c = 0; c < h; c++)
	    {
		blurx[r][c][k][0] = blury[r - 2][c][k][0] + 4.0f * blury[r - 1][c][k][0] + 6.0f * blury[r][c][k][0] + 4.0f * blury[r + 1][c][k][0] + blury[r + 2][c][k][0];
		blurx[r][c][k][1] = blury[r - 2][c][k][1] + 4.0f * blury[r - 1][c][k][1] + 6.0f * blury[r][c][k][1] + 4.0f * blury[r + 1][c][k][1] + blury[r + 2][c][k][1];
	    }
	}
    }
    
    #pragma omp parallel for private(c)
    for(k = 0; k < d; k++)
    {
	for(c = 0; c < h; c++)
	{
	    blurx[0][c][k][0] = 6.0f * blury[0][c][k][0] + 4.0f * blury[1][c][k][0] + blury[2][c][k][0];
	    blurx[0][c][k][1] = 6.0f * blury[0][c][k][1] + 4.0f * blury[1][c][k][1] + blury[2][c][k][1];
	    
	    blurx[1][c][k][0] = 4.0f * blury[0][c][k][0] + 6.0f * blury[1][c][k][0] + 4.0f * blury[2][c][k][0] + blury[3][c][k][0];
	    blurx[1][c][k][1] = 4.0f * blury[0][c][k][1] + 6.0f * blury[1][c][k][1] + 4.0f * blury[2][c][k][1] + blury[3][c][k][1];
	    
	    blurx[w - 2][c][k][0] = blury[w - 4][c][k][0] + 4.0f * blury[w - 3][c][k][0] + 6.0f * blury[w - 2][c][k][0] + 4.0f * blury[w - 1][c][k][0];
	    blurx[w - 2][c][k][1] = blury[w - 4][c][k][1] + 4.0f * blury[w - 3][c][k][1] + 6.0f * blury[w - 2][c][k][1] + 4.0f * blury[w - 1][c][k][1];
	    
	    blurx[w - 1][c][k][0] = blury[w - 3][c][k][0] + 4.0f * blury[w - 2][c][k][0] + 6.0f * blury[w - 1][c][k][0];
	    blurx[w - 1][c][k][1] = blury[w - 3][c][k][1] + 4.0f * blury[w - 2][c][k][1] + 6.0f * blury[w - 1][c][k][1];
	}
    }
    
    float zv;
    float xf, yf, zf;
    float lerp1, lerp2, lerp3, lerp4, lerp5, lerp6;
    
    #pragma omp parallel for private(c, val, zv, zi, zf, xf, yf, xi, yi, lerp1, lerp2, lerp3, lerp4, lerp5, lerp6)
    for(r = 0; r < input_img->rows; r++)
    {
	for(c = 0; c < input_img->cols; c++)
	{
	    val = clip(input_img->data[r][c], 0, 1);
	    zv = val * (1.0f / r_sigma);
	    zi = (int)zv;
	    zf = zv - zi;
	    xf = (float)(r % s_sigma) / (float)s_sigma;
	    yf = (float)(c % s_sigma) / (float)s_sigma;
	    xi = (int)(r / s_sigma);
	    yi = (int)(c / s_sigma);
	    
	    if(zi == 10)
		zi = 9;
	    
	    lerp1 = blurx[xi][yi][zi][0] * (1.0f - yf) + blurx[xi][yi + 1][zi][0] * yf;
	    lerp2 = blurx[xi + 1][yi][zi][0] * (1.0f - yf) + blurx[xi + 1][yi + 1][zi][0] * yf;
	    lerp3 = lerp1 * (1.0f - xf) + lerp2 * xf;
	    
	    lerp4 = blurx[xi][yi][zi + 1][0] * (1.0f - yf) + blurx[xi][yi + 1][zi + 1][0] * yf;
	    lerp5 = blurx[xi + 1][yi][zi + 1][0] * (1.0f - yf) + blurx[xi + 1][yi + 1][zi + 1][0] * yf;
	    lerp6 = lerp4 * (1.0f - xf) + lerp5 * xf;
	    
	    interpolated[r][c][0] = lerp3 * (1.0f - zf) + lerp6 * zf;
	    
	    lerp1 = blurx[xi][yi][zi][1] * (1.0f - yf) + blurx[xi][yi + 1][zi][1] * yf;
	    lerp2 = blurx[xi + 1][yi][zi][1] * (1.0f - yf) + blurx[xi + 1][yi + 1][zi][1] * yf;
	    lerp3 = lerp1 * (1.0f - xf) + lerp2 * xf;
	    
	    lerp4 = blurx[xi][yi][zi + 1][1] * (1.0f - yf) + blurx[xi][yi + 1][zi + 1][1] * yf;
	    lerp5 = blurx[xi + 1][yi][zi + 1][1] * (1.0f - yf) + blurx[xi + 1][yi + 1][zi + 1][1] * yf;
	    lerp6 = lerp4 * (1.0f - xf) + lerp5 * xf;
	    
	    interpolated[r][c][1] = lerp3 * (1.0f - zf) + lerp6 * zf;
	    
	    output_img->data[r][c] = interpolated[r][c][0] / interpolated[r][c][1];
	}
    }
    
    free(histogram_data);
    free(histogram_pointer);
    free(blurx_data);
    free(blurx_pointer);
    free(blury_data);
    free(blury_pointer);
    free(blurz_data);
    free(blurz_pointer);
    
    free(interpolated_data);
    free(interpolated_pointer);
}

int main(int argc, char **argv)
{
    FLOAT_IMG *input_img;
    
    if(argc == 1)
    {
	input_img = read_img_gray("../../images/gray.png");
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
    bilateral_grid(input_img, output_img);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
	write_img_gray(output_img, "test.png");
    else
	write_img_gray(output_img, argv[2]);
    
    free_img_gray(input_img);
    free_img_gray(output_img);
    
    return 0;
}
	    
