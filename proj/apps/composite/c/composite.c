#include "c_util.h"
#include <stdio.h>
#include <omp.h>

void composite(FLOAT3_IMG *background, FLOAT3_IMG *foreground, FLOAT_IMG *foreground_alpha, FLOAT3_IMG *output_img)
{
    int r, c;
    float alpha;
    
    #pragma omp parallel for private(c, alpha)
    for(r = 0; r < background->rows; r++)
    {
	for(c = 0; c < background->cols; c++)
	{
	    alpha = foreground_alpha->data[r][c];
	    output_img->data[r][c][0] = (1.0f - alpha) * background->data[r][c][0] + alpha * foreground->data[r][c][0];
	    output_img->data[r][c][1] = (1.0f - alpha) * background->data[r][c][1] + alpha * foreground->data[r][c][1];
	    output_img->data[r][c][2] = (1.0f - alpha) * background->data[r][c][2] + alpha * foreground->data[r][c][2];
	}
    }
}

int main(int argc, char **argv)
{
    FLOAT3_IMG *background;
    FLOAT3_IMG *foreground;
    FLOAT_IMG *foreground_alpha;
    
    if(argc == 1)
    {
	background = read_img_rgb("../../images/temple_rgb.png");
	foreground = read_img_rgb("../../images/house_rgb.png");
	foreground_alpha = read_img_gray("../../images/temple_gray.png");
    }
    else
    {
	if(argc == 5)
	{
	    background = read_img_rgb(argv[1]);
	    foreground = read_img_rgb(argv[2]);
	    foreground_alpha = read_img_gray(argv[3]);
	}
	else
	{
	    printf("Usage: ./a.out background.png foreground.png foreground_alpha.png out.png or ./a.out");
	    return 0;
	}
    }
    
    FLOAT3_IMG *output_img = zeros_rgb(background->rows, background->cols);
    
    double d1 = wall_time();
    composite(background, foreground, foreground_alpha, output_img);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
	write_img_rgb(output_img, "test.png");
    else
	write_img_rgb(output_img, argv[4]);
    
    return 0;
}
