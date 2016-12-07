#include "c_util.h"
#include <omp.h>
#include <stdio.h>

//supports only RGB

void blur_two_stage(FLOAT_IMG *img, FLOAT_IMG *output_img)
{
    int r, c, k;
    FLOAT_IMG *temp_img = zeros_gray(img->rows, img->cols);
    
    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows - 8; r++)
    {
        for(c = 0; c < img->cols - 2; c++)
	{  
	    temp_img->data[r][c] = (img->data[r][c] + img->data[r + 1][c] + img->data[r + 2][c]) / 3.0f;
	}
    }
    
    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows - 8; r++)
    {
        for(c = 0; c < img->cols - 2; c++)
	{ 
	    output_img->data[r][c] = (temp_img->data[r][c] + temp_img->data[r][c + 1] + temp_img->data[r][c + 2]) / 3.0f;
	}
    }
}

int main(int argc, char **argv)
{  
    FLOAT_IMG *img;
    
    if(argc == 1)
    {
	img = read_img_gray("../../images/temple_gray.png");
    }
    else
    {
	if(argc == 3)
	{
	    img = read_img_gray(argv[1]);
	}
	else
	{
	    printf("Usage: ./a.out in.png out.png or ./a.out\n");
	    return 0;
	}
    }
    
    FLOAT_IMG *output_img = zeros_gray(img->rows, img->cols);
    
    double d1 = wall_time();
    blur_two_stage(img, output_img);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
	write_img_gray(output_img, "test.png");
    else
	write_img_gray(output_img, argv[2]);
    
    free_img_gray(img);
    free_img_gray(output_img);
    
    return 0;
}