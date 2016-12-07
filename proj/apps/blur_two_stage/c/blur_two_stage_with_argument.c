#include "c_util.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    if(argc < 2)
    {
	printf("Invalid argument.\n");
	printf("Usage: \"./a.out rgb\" or \"./a.out gray\"\n");
	return 0;
    }
  
    if(strcmp(argv[1], "rgb") == 0)
    {
	FLOAT3_IMG *img = read_img_rgb("../../images/temple_rgb.png");
    
	int i, j, r, c, k;
    
	FLOAT3_IMG *temp_img = zeros_rgb(img->rows, img->cols);
	FLOAT3_IMG *output_img = zeros_rgb(img->rows, img->cols);
    
	for(r = 0; r < img->rows - 8; r++)
	{
	    for(c = 0; c < img->cols - 2; c++)
	    {
		for(k = 0; k < 3; k++)
		{
		    temp_img->data[r][c][k] = (img->data[r][c][k] + img->data[r + 1][c][k] + img->data[r + 2][c][k]) / 3.0f;
		}
	    }
	}
    
	for(r = 0; r < img->rows - 8; r++)
	{
	    for(c = 0; c < img->cols - 2; c++)
	    {
		for(k = 0; k < 3; k++)
		{
		    output_img->data[r][c][k] = (temp_img->data[r][c][k] + temp_img->data[r][c + 1][k] + temp_img->data[r][c + 2][k]) / 3.0f;
		}
	    }
	}
    
	write_img_rgb(output_img, "test_rgb.png");
    }
    
    else if(strcmp(argv[1], "gray") == 0)
    {
	FLOAT_IMG *img = read_img_gray("../../images/temple_gray.png");
    
	int i, j, r, c, k;
    
	FLOAT_IMG *temp_img = zeros_gray(img->rows, img->cols);
	FLOAT_IMG *output_img = zeros_gray(img->rows, img->cols);
    
	for(r = 0; r < img->rows - 8; r++)
	{
	    for(c = 0; c < img->cols - 2; c++)
	    {
		temp_img->data[r][c] = (img->data[r][c] + img->data[r + 1][c] + img->data[r + 2][c]) / 3.0f;
	    }
	}
    
	for(r = 0; r < img->rows - 8; r++)
	{
	    for(c = 0; c < img->cols - 2; c++)
	    {
		output_img->data[r][c] = (temp_img->data[r][c] + temp_img->data[r][c + 1] + temp_img->data[r][c + 2]) / 3.0f;
	    }
	}
    
	write_img_gray(output_img, "test_gray.png");
    }
    
    else
    {
	printf("Invalid argument.\n");
	printf("Usage: \"./a.out rgb\" or \"./a.out gray\"\n");
    }

    return 0;
}
