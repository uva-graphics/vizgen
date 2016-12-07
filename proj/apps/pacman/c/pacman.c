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

float clip(float input, float lower_bound, float upper_bound)
{
    return min(max(input, lower_bound), upper_bound);
}

void draw_circle(int x, int y, int r, FLOAT3_IMG *img, float color_r, float color_g, float color_b)
{
    int y_start = max(0, y - r);
    int y_stop = min(y + r, img->cols - 1);
    int x_i, y_i, x_start, x_stop;
    
    #pragma omp parallel for private(x_i, x_start, x_stop)
    for (y_i = y_start; y_i < y_stop; y_i++)
    {
	x_start = (int)(x - sqrt(r * r - (y - y_i) * (y - y_i)));
	x_stop = (int)(x + sqrt(r * r - (y - y_i) * (y - y_i)));
	
	for (x_i = x_start; x_i < x_stop; x_i++)
	{
	    img->data[y_i][x_i][0] = color_r;
	    img->data[y_i][x_i][1] = color_g;
	    img->data[y_i][x_i][2] = color_b;
	}
    }
}

void draw_pacman_right(int x, int y, int r, FLOAT3_IMG *img, float ma, float color_r, float color_g, float color_b)
{
    int y_start = max(0, y - r);
    int y_stop = min(y + r, img->cols - 1);
    int x_i, y_i, x_start, x_stop;
    float r_mouth;
    
    #pragma omp parallel for private(x_i, x_start, x_stop, r_mouth)
    for (y_i = y_start; y_i < y_stop; y_i++)
    {
	x_start = (int)(x - sqrt(r * r - (y - y_i) * (y - y_i)));
	x_stop = (int)(x + sqrt(r * r - (y - y_i) * (y - y_i)));
	
	if (y_i > (float)y - (float)r * sin(ma) && y_i <= y)
	{
	    r_mouth = (float)(y - y_i) / sin(ma);
	    x_stop = (int)(x + r_mouth * cos(ma));
	}
	else if (y_i < (float)y + (float)r * sin(ma) && y_i > y)
	{
	    r_mouth = (float)(y_i - y) / sin(ma);
	    x_stop = (int)(x + r_mouth * cos(ma));
	}
	
	for (x_i = x_start; x_i < x_stop; x_i++)
	{
	  img->data[y_i][x_i][0] = color_r;
	  img->data[y_i][x_i][1] = color_g;
	  img->data[y_i][x_i][2] = color_b;
	}
    }
    
    draw_circle(x, y - r / 2, r / 10, img, 0.0f, 0.0f, 0.0f);
}
    
void draw_rect(int x, int y, int w, int h, FLOAT3_IMG *img, float color_r, float color_g, float color_b)
{
    int y_start = max(0, y);
    int y_stop = min(y + h, img->cols - 1);
    
    int x_start = max(0, x);
    int x_stop = min(x + w, img->rows - 1);
    
    int x_i, y_i;
    
    #pragma omp parallel for private(x_i)
    for (y_i = y_start; y_i < y_stop; y_i++)
    {
	for (x_i = x_start; x_i < x_stop; x_i++)
	{
	  img->data[x_i][y_i][0] = color_r;
	  img->data[x_i][y_i][1] = color_g;
	  img->data[x_i][y_i][2] = color_b;
	}
    }
}

void draw_ghost(int x, int y, int r, FLOAT3_IMG *img, float color_r, float color_g, float color_b, float tf, int blink)
{
    int y_start = max(0, y - r);
    int y_stop = min(y + r, img->rows - 1);
    
    int x_i, y_i, x_start, x_stop;
    
    #pragma omp parallel for private(x_i, x_start, x_stop)
    for (y_i = y_start; y_i < y_stop; y_i++)
    {
	x_start = (int)(x - sqrt(r * r - (y - y_i) * (y - y_i)));
	x_stop = (int)(x + sqrt(r * r - (y - y_i) * (y - y_i)));
	
	if (y_i > y)
	{
	    x_start = max(0, x - r);
	    x_stop = min(x + r, img->cols - 1);
	}
	
	for (x_i = x_start; x_i < x_stop; x_i++)
	{
	    if (y_i <= y + tf * r)
	    {
		img->data[y_i][x_i][0] = color_r;
		img->data[y_i][x_i][1] = color_g;
		img->data[y_i][x_i][2] = color_b;
	    }
	    else
	    {
		if (x_i < x - r * (5.0f / 7.0f) || (x_i > x - r * (3.0f / 7.0f) && x_i < x - r * (1.0f / 7.0f)) || (x_i > x + r * 1.0f / 7.0f && x_i < x + r * (3.0f / 7.0f)) || x_i > x + r * (5.0f / 7.0f))
		{
		    img->data[y_i][x_i][0] = color_r;
		    img->data[y_i][x_i][1] = color_g;
		    img->data[y_i][x_i][2] = color_b;
		}
	    }
	}
    }
    
    if (!blink)
    {
	draw_circle(x - r / 4, y - r / 2, r / 5, img, 1.0f, 1.0f, 1.0f);
	draw_circle(x + r / 4, y - r / 2, r / 5, img, 1.0f, 1.0f, 1.0f);
	draw_circle(x - r / 8, y - r / 2, r / 9, img, 0.0f, 0.0f, 1.0f);
	draw_circle(x + 3 * r / 8, y - r / 2, r / 9, img, 0.0f, 0.0f, 1.0f);
    }
    else
    {
	draw_rect(y - r / 2, x - r / 4, r / 8, r / 4, img, 0.0f, 0.0f, 0.0f);
	draw_rect(y - r / 2, x + r / 4, r / 8, r / 4, img, 0.0f, 0.0f, 0.0f);
    }
}

void render_1_frame_no_write(FLOAT3_IMG *input_img, FLOAT3_IMG *output_img, float time)
{
    int ma = (int)time % 16;
    
    if (ma > 7)
	ma = 8 - (ma - 8);
    
    draw_rect(0, 0, output_img->cols, output_img->rows, output_img, 0.0f, 0.0f, 0.0f);
    
    draw_pacman_right(100, 100, 50, output_img, (float)ma * M_PI / 32.0, 1.0f, 1.0f, 0.0f);
    
    draw_circle(200, 100, 15, output_img, 1.0f, 1.0f, 1.0f);
    draw_circle(275, 100, 15, output_img, 1.0f, 1.0f, 1.0f);
    draw_circle(350, 100, 15, output_img, 1.0f, 1.0f, 1.0f);
    
    draw_ghost(500, 100, 50, output_img, 0.0f, 1.0f, 1.0f, 0.75f, ma == 7);
}

int main(int argc, char **argv)
{
    FLOAT3_IMG *input_img = zeros_rgb(200, 600);
    FLOAT3_IMG *output_img = zeros_rgb(200, 600);
    
    float time;
    double d1, d2;
    
    if (argc > 3)
    {
	printf("Usage: ./a.out or ./a.out out.png or ./a.out out.png time\n");
	return 0;
    }
    
    if (argc == 3)
    {
	time = atof(argv[2]);
	d1 = wall_time();
	render_1_frame_no_write(input_img, output_img, time);
	d2 = wall_time();
    }
    else
    {
	time = 3.0f;
	d1 = wall_time();
	render_1_frame_no_write(input_img, output_img, time);
	d2 = wall_time();
    }
    
    printf("%f\n", d2 - d1);
    
    if (argc == 1)
	write_img_rgb(output_img, "test.png");
    else
	write_img_rgb(output_img, argv[1]);
    
    return 0;
}
    
    
