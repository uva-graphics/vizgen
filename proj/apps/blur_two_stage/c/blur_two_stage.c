#include "c_util.h"
#include <omp.h>
#include <stdio.h>

//supports only RGB

void blur_two_stage(FLOAT3_IMG *img, FLOAT3_IMG *output_img)
{
    int r, c, k;
    FLOAT3_IMG *temp_img = zeros_rgb(img->rows, img->cols);
    
    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows - 8; r++)
    {
        for(c = 0; c < img->cols - 2; c++)
	{  
	    temp_img->data[r][c][0] = (img->data[r][c][0] + img->data[r + 1][c][0] + img->data[r + 2][c][0]) / 3.0f;
	    temp_img->data[r][c][1] = (img->data[r][c][1] + img->data[r + 1][c][1] + img->data[r + 2][c][1]) / 3.0f;
	    temp_img->data[r][c][2] = (img->data[r][c][2] + img->data[r + 1][c][2] + img->data[r + 2][c][2]) / 3.0f;
	}
    }
    
    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows - 8; r++)
    {
        for(c = 0; c < img->cols - 2; c++)
	{ 
	    output_img->data[r][c][0] = (temp_img->data[r][c][0] + temp_img->data[r][c + 1][0] + temp_img->data[r][c + 2][0]) / 3.0f;
	    output_img->data[r][c][1] = (temp_img->data[r][c][1] + temp_img->data[r][c + 1][1] + temp_img->data[r][c + 2][1]) / 3.0f;
	    output_img->data[r][c][2] = (temp_img->data[r][c][2] + temp_img->data[r][c + 1][2] + temp_img->data[r][c + 2][2]) / 3.0f;
	}
    }
}

int main(int argc, char **argv)
{
    FLOAT3_IMG *img;
    
    if(argc == 1)
    {
	img = read_img_rgb("../../images/temple_rgb.png");
    }
    else
    {
	if(argc == 3)
	{
	    img = read_img_rgb(argv[1]);
	}
	else
	{
	    printf("Usage: ./a.out in.png out.png or ./a.out\n");
	    return 0;
	}
    }  
    
    FLOAT3_IMG *output_img = zeros_rgb(img->rows, img->cols);
    
    double d1 = wall_time();
    blur_two_stage(img, output_img);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
	write_img_rgb(output_img, "test.png");
    else
	write_img_rgb(output_img, argv[2]);
    
    free_img_rgb(img);
    free_img_rgb(output_img);
    
    return 0;
}