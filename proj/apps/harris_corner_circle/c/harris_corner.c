#include "c_util.h"
#include <stdio.h>
#include <omp.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

//corner position stored in pointer corner

int count_harris_corner(FLOAT_IMG *input_img, float **R)
{
    float k = 0.05f;
    float threshold = 0.5f;

    FLOAT_IMG *Ix = zeros_gray(input_img->rows, input_img->cols);
    FLOAT_IMG *Iy = zeros_gray(input_img->rows, input_img->cols);
    FLOAT_IMG *Ix2 = zeros_gray(input_img->rows, input_img->cols);
    FLOAT_IMG *Iy2 = zeros_gray(input_img->rows, input_img->cols);
    FLOAT_IMG *Ixy = zeros_gray(input_img->rows, input_img->cols);

    int r, c;
    int idx_r1, idx_r2, idx_c1, idx_c2;

    #pragma omp parallel for private(c, idx_r1, idx_r2, idx_c1, idx_c2)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            idx_r1 = r - 1;
            idx_r2 = r + 1;
            idx_c1 = c - 1;
            idx_c2 = c + 1;

            if(r == 0)
                idx_r1 = 0;
            if(r == input_img->rows - 1)
                idx_r2 = input_img->rows - 1;
            if(c == 0)
                idx_c1 = 0;
            if(c == input_img->cols - 1)
                idx_c2 = input_img->cols - 1;

            Ix->data[r][c] = -input_img->data[idx_r1][idx_c1] - 2.0f * input_img->data[idx_r1][c] - input_img->data[idx_r1][idx_c2] + input_img->data[idx_r2][idx_c1] + 2.0f * input_img->data[idx_r2][c] + input_img->data[idx_r2][idx_c2];
            Iy->data[r][c] = -input_img->data[idx_r1][idx_c1] - 2.0f * input_img->data[r][idx_c1] - input_img->data[idx_r2][idx_c1] + input_img->data[idx_r1][idx_c2] + 2.0f * input_img->data[r][idx_c2] + input_img->data[idx_r2][idx_c2];

            Ix2->data[r][c] = Ix->data[r][c] * Ix->data[r][c];
            Iy2->data[r][c] = Iy->data[r][c] * Iy->data[r][c];
            Ixy->data[r][c] = Ix->data[r][c] * Iy->data[r][c];
        }
    }

    int count = 0;
    float Sx2, Sy2, Sxy;
    float det, trace;

    #pragma omp parallel for private(c, idx_r1, idx_r2, idx_c1, idx_c2, Sx2, Sy2, Sxy, det, trace) reduction(+: count)
    for(r = 0; r < input_img->rows; r++)
    {
        for(c = 0; c < input_img->cols; c++)
        {
            idx_r1 = r - 1;
            idx_r2 = r + 1;
            idx_c1 = c - 1;
            idx_c2 = c + 1;

            if(r == 0)
                idx_r1 = 0;
            if(r == input_img->rows - 1)
                idx_r2 = input_img->rows - 1;
            if(c == 0)
                idx_c1 = 0;
            if(c == input_img->cols - 1)
                idx_c2 = input_img->cols - 1;

            Sx2 = Ix2->data[idx_r1][idx_c1] + 2.0f * Ix2->data[idx_r1][c] + Ix2->data[idx_r1][idx_c2] + 2.0f * Ix2->data[r][idx_c1] + 4.0f * Ix2->data[r][c] + 2.0f * Ix2->data[r][idx_c2] + Ix2->data[idx_r2][idx_c1] + 2.0f * Ix2->data[idx_r2][c] + Ix2->data[idx_r2][idx_c2];
            Sy2 = Iy2->data[idx_r1][idx_c1] + 2.0f * Iy2->data[idx_r1][c] + Iy2->data[idx_r1][idx_c2] + 2.0f * Iy2->data[r][idx_c1] + 4.0f * Iy2->data[r][c] + 2.0f * Iy2->data[r][idx_c2] + Iy2->data[idx_r2][idx_c1] + 2.0f * Iy2->data[idx_r2][c] + Iy2->data[idx_r2][idx_c2];
            Sxy = Ixy->data[idx_r1][idx_c1] + 2.0f * Ixy->data[idx_r1][c] + Ixy->data[idx_r1][idx_c2] + 2.0f * Ixy->data[r][idx_c1] + 4.0f * Ixy->data[r][c] + 2.0f * Ixy->data[r][idx_c2] + Ixy->data[idx_r2][idx_c1] + 2.0f * Ixy->data[idx_r2][c] + Ixy->data[idx_r2][idx_c2];

            Sx2 /= 16.0f;
            Sy2 /= 16.0f;
            Sxy /= 16.0f;

            det = Sx2 * Sy2 - Sxy * Sxy;
            trace = Sx2 + Sy2;
            R[r][c] = det - k * trace;

            if(R[r][c] > threshold)
                count++;
        }
    }

    free_img_gray(Ix);
    free_img_gray(Iy);
    free_img_gray(Ix2);
    free_img_gray(Iy2);
    free_img_gray(Ixy);

    return count;
}

void draw_circles(FLOAT_IMG *input_img,
                  FLOAT_IMG *output_img, 
                  int **corners,
                  int num_corners)
{
    int circle_radius = 5, circle_color = 0.5f;
    int h = input_img->height, w = input_img->width;
    int i, r, c, corner_r, corner_c, w_start, w_stop, h_start, h_stop;
    float d2, r2 = (float)(circle_radius * circle_radius);

    //printf("num_corners = %d\n", num_corners);

    //#pragma omp parallel for private(i, r, c, corner_r, corner_c, w_start, w_stop, h_start, h_stop, d2)
    for(i = 0; i < num_corners; i++)
    {
        corner_r = corners[i][0];
        corner_c = corners[i][1];

        w_start = (int)(MAX(0              , corner_c - circle_radius));
        w_stop  = (int)(MIN(input_img->cols, corner_c + circle_radius));
        h_start = (int)(MAX(0              , corner_r - circle_radius));
        h_stop  = (int)(MIN(input_img->rows, corner_r + circle_radius));

        //printf("w_start=%d, w_stop=%d, h_start=%d, h_stop=%d\n", 
        //    w_start, w_stop, h_start, h_stop);


        for(r = h_start; r < h_stop; r++)
        {
            for(c = w_start; c < w_stop; c++)
            {
                d2 = (float)((r - h_start - circle_radius) * 
                             (r - h_start - circle_radius) + 
                             (c - w_start - circle_radius) *
                             (c - w_start - circle_radius));

                if(d2 <= r2)
                {
                    output_img->data[r][c] = circle_color;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    FLOAT3_IMG *img;

    if(argc == 1)
    {
        img = read_img_rgb("../../images/window.png");
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

    FLOAT_IMG *input_img = zeros_gray(img->rows, img->cols);
    float threshold = 0.5f;

    int r, c;

    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows; r++)
    {
        for(c = 0; c < img->cols; c++)
        {
            input_img->data[r][c] = 0.299f * img->data[r][c][0] + 0.587f * img->data[r][c][1] + 0.114 * img->data[r][c][2];
        }
    }

    float **R = malloc(img->rows * sizeof(float *));
    float *R_data = malloc(img->rows * img->cols * sizeof(float));

    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows; r++)
    {
        R[r] = &R_data[r * img->cols];
        for(c = 0; c < img->cols; c++)
        {
            R[r][c] = 0.0f;
        }
    }

    int count;

    double d1 = wall_time();

    count = count_harris_corner(input_img, R);

    int **corner = malloc(count * sizeof(int *));
    int *corner_data = malloc(count * 2 * sizeof(int));

    for(r = 0; r < count; r++)
    {
        corner[r] = &corner_data[r * 2];
        corner[r][0] = 0.0f;
        corner[r][1] = 0.0f;
    }

    count = 0;

    #pragma omp parallel for private(c)
    for(r = 0; r < img->rows; r++)
    {
        for(c = 0; c < img->cols; c++)
        {
            if(R[r][c] > threshold)
            {
                corner[count][0] = r;
                corner[count][1] = c;
                count++;
            }
        }
    }

    draw_circles(input_img, input_img, corner, count);

    double d2 = wall_time();
    printf("%f\n", d2 - d1);

    write_img_gray(input_img, argv[2]); 

    /*
    FILE *fp;

    if(argc == 1)
        fp = fopen("corner_position.csv", "w");
    else
        fp = fopen(argv[2], "w");

    fprintf(fp, "x_position, y_position");

    for(r = 0; r < count; r++)
    {
        fprintf(fp, "\n%i, %i", corner[r][0], corner[r][1]);
    }

    fclose(fp);
    */

    free(R);
    free(R_data);
    free(corner);
    free(corner_data);

    free_img_rgb(img);
    free_img_gray(input_img);

    return 0;
}




