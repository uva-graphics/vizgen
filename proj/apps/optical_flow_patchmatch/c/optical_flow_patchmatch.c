#include "c_util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define PATCH_W 16
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

void draw_line(float x0, float y0, float x1, float y1, FLOAT3_IMG *input_img, float *color)
{
    bool isvalid_slope = fabs(y1 - y0) < fabs(x1 - x0);
    float temp;
    
    if(!isvalid_slope)
    {
        temp = x0;
        x0 = y0;
        y0 = temp;
        temp = x1;
        x1 = y1;
        y1 = temp;
    }
    
    bool isvalid_x = x1 > x0;
    
    if(!isvalid_x)
    {
        temp = x0;
        x0 = x1;
        x1 = temp;
        temp = y0;
        y0 = y1;
        y1 = temp;
    }
    
    float dx, dy;
    dx = x1 - x0;
    dy = fabs(y1 - y0);
    
    float err = dx / 2.0f;
    int y_sign;
    
    if(y0 < y1)
        y_sign = 1;
    else
        y_sign = -1;
    
    int y = (int)y0;
    
    int count = 0;
    int x;
    
    for(x = (int)x0; x < (int)x1 + 1; x++)
    {
        if(isvalid_slope)
        {
            if(x > -1 && y > -1 && x < input_img->rows && y < input_img->cols)
            {
                input_img->data[x][y][0] = color[0];
                input_img->data[x][y][1] = color[1];
                input_img->data[x][y][2] = color[2];
            }
        }
        else
        {
            if(x > -1 && y > -1 && x < input_img->cols && y < input_img->rows)
            {
                input_img->data[y][x][0] = color[0];
                input_img->data[y][x][1] = color[1];
                input_img->data[y][x][2] = color[2];
            }
        }
        
        count++;
        err -= dy;
        
        if(err < 0)
        {
            y += y_sign;
            err += dx;
        }
    }
}

void draw_arrow(float x0, float y0, float x1, float y1, FLOAT3_IMG *input_img, float *color)
{
    draw_line(x0, y0, x1, y1, input_img, color);
    
    float dx, dy;
    dx = x1 - x0;
    dy = y1 - y0;
    
    float x2, y2, x3, y3;
    x2 = x0 + 0.75f * dx + 0.25f * dy / sqrt(3);
    y2 = y0 + 0.75f * dy - 0.25f * dx / sqrt(3);
    x3 = x0 + 0.75f * dx - 0.25f * dy / sqrt(3);
    y3 = y0 + 0.75f * dy + 0.25f * dx / sqrt(3);
    
    draw_line(x2, y2, x1, y1, input_img, color);
    draw_line(x3, y3, x1, y1, input_img, color);
}

void find_color_value(float *dist, float r, float *color)
{
    float angle = atan2(dist[0], dist[1]);
    float scale;
    float real_angle;
    
    color[0] = 0.0f;
    color[1] = 0.0f;
    color[2] = 0.0f;
    
    if(angle >= 0 && angle <= 2.0f * M_PI / 3.0f)
    {
        scale = angle * 3.0f / (2.0f * M_PI);
        color[0] = 1.0f - scale;
        color[1] = scale;
    }
    
    if(angle > 2.0f * M_PI / 3.0f)
    {
        scale = (angle - 2.0f * M_PI / 3.0f) * 3.0f / (2.0f * M_PI);
        color[1] = 1.0f - scale;
        color[2] = scale;
    }
    
    if(angle < -2.0f * M_PI / 3.0f)
    {
        real_angle = angle + 2.0f * M_PI;
        scale = (real_angle - 2.0f * M_PI / 3.0f) * 3.0f / (2.0f * M_PI);
        color[1] = 1.0f - scale;
        color[2] = scale;
    }
    
    if(angle < 0 && angle >= -2.0f * M_PI / 3.0f)
    {
        real_angle = angle + 2.0f * M_PI;
        scale = (real_angle - 4.0f * M_PI / 3.0f) * 3.0f / (2.0f * M_PI);
        color[2] = 1.0f - scale;
        color[0] = scale;
    }
    
    color[0] = color[0] * sqrt(sqrt(r));
    color[1] = color[1] * sqrt(sqrt(r));
    color[2] = color[2] * sqrt(sqrt(r));
}

/*
Calculates the Log_2 of a number, used:
http://stackoverflow.com/a/758009
*/
double log2(double n)
{
    return log(n) / log(2);
}

/*
Analogous to Python's: random.randrange()
*/
int rand_range(int start, int stop)
{
    int range = stop - start;
    double random_number = ((double)rand()) / ((double)RAND_MAX);

    return start + random_number * range;
}

int rand_range_seed(int seed, int start, int stop) {
    uint32_t rand2_u, rand2_v, rand_result;
    rand2_u = seed;
    rand2_v = ~rand2_u;
    rand2_v = 36969 * (rand2_v & 65535) + (rand2_v >> 16);
    rand2_u = 18000 * (rand2_u & 65535) + (rand2_u >> 16);
    rand_result = (rand2_v << 16) + (rand2_u & 65535);
    return start + rand_result % (stop - start);
}

double patch_dist(FLOAT3_IMG *img1, 
                  FLOAT3_IMG *img2, 
                  int ax, int ay, 
                  int bx, int by, 
                  double dmax)
{
    int dx, dy, c;
    int ax_current, ay_current, bx_current, by_current;
    double a_color, b_color;
    double ans = 0.0;

    for(dy = 0; dy < PATCH_W; dy++)
    {
        ay_current = ay + dy;
        by_current = by + dy;

        for(dx = 0; dx < PATCH_W; dx++)
        {
            ax_current = ax + dx;
            bx_current = bx + dx;

            for(c = 0; c < 3; c++)
            {
                a_color = img1->data[ay_current][ax_current][c];
                b_color = img2->data[by_current][bx_current][c];
                ans += (b_color - a_color) * (b_color - a_color);
            }

            if(ans > dmax)
            {
                return ans;
            }
        }
    }

    return ans;
}

void optical_flow(FLOAT3_IMG *img1, 
                  FLOAT3_IMG *img2, 
                  FLOAT3_IMG *output_img,
                  FLOAT3_IMG *nnf)
{
    int max_offset = 11, nn_iters = 2;
    int eh = img1->height - PATCH_W + 1, ew = img1->width - PATCH_W + 1;
    int rs_iters = (int)(ceil(log2((double)(max_offset * 2))));
//    printf("rs_iters: %d\n", rs_iters);
    int ay, ax, bx, by, nn_iter, rs_iter;
    
    int r, c, k;

    // Initialization:

    
    for(k = 0; k < 3; k++)
    {
        for(r = 0; r < img1->rows; r++)
        {
            for(c = 0; c < img1->cols; c++)
                output_img->data[r][c][k] = img2->data[r][c][k];
        }
    }

    // Initialization:

#pragma omp parallel for
    for(ay = 0; ay < eh; ay++)
    {
        for(ax = 0; ax < ew; ax++)
        {
            int xmin = MAX(ax - max_offset, 0);
            int xmax = MIN(ax + max_offset + 1, ew);
            int ymin = MAX(ay - max_offset, 0);
            int ymax = MIN(ay + max_offset + 1, eh);

            int seed = (ay << 16) | ax;
            bx = rand_range_seed(seed, xmin, xmax);
            by = rand_range_seed(seed, ymin, ymax);

            nnf->data[ay][ax][0] = bx;
            nnf->data[ay][ax][1] = by;
            nnf->data[ay][ax][2] = patch_dist(img1, img2, ax, ay, bx, by, 
                1.0E10);
        }
    }

    // Iterations:

    for(nn_iter = 0; nn_iter < nn_iters; nn_iter++)
    {
        int y_start = 0, y_end = eh, y_step = 1;
        int x_start = 0, x_end = ew, x_step = 1;

        if(nn_iter % 2 == 1)
        {
            y_start = eh - 1;
            y_end = -1;
            y_step = -1;

            x_start = ew - 1;
            x_end = -1;
            x_step = -1;
        }

        for(ay = y_start; ay != y_end; ay += y_step)
        {
            for(ax = x_start; ax != x_end; ax += x_step)
            {
//                printf("nn_iter=%d, ax=%d, ay=%d, x_step=%d, y_step=%d\n", nn_iter, ax, ay, x_step, y_step);
                int bx_best = (int)(nnf->data[ay][ax][0]);
                int by_best = (int)(nnf->data[ay][ax][1]);
                double dist_best = nnf->data[ay][ax][2];

                int ax_p = ax - x_step;
                int ay_p = ay - y_step;
                int rs_max = max_offset * 2;

                // Propagation (x), incremental algorithm that takes O(patch_w) time

                if(ax_p >= 0 && ax_p < ew)
                {
                    bx = (int)(nnf->data[ay][ax_p][0]) + x_step;
                    by = (int)(nnf->data[ay][ax_p][1]);

                    if(bx >= 0 &&
                       bx < ew &&
                       abs(ax - bx) <= max_offset &&
                       abs(ay - by) <= max_offset)
                    {
                        double dist = nnf->data[ay][ax_p][2];

                        int delta_add = PATCH_W-1;
                        int delta_remove = -1;
                        if(x_step < 0) {
                            delta_add = 0;
                            delta_remove = PATCH_W;
                        }
                        for (int dy = 0; dy < PATCH_W; dy++) {
                            int ay_current = (ay + dy);
                            int by_current = (by + dy);

                            int ax_current = (ax + delta_remove);
                            int bx_current = (bx + delta_remove);
                            for (int c = 0; c < 3; c++) {
                                real acolor = img1->data[ay_current][ax_current][c];
                                real bcolor = img2->data[by_current][bx_current][c];
                                dist -= (bcolor - acolor)*(bcolor - acolor);
                            }

                            ax_current = (ax + delta_add);
                            bx_current = (bx + delta_add);
                            for (int c = 0; c < 3; c++) {
                                real acolor = img1->data[ay_current][ax_current][c];
                                real bcolor = img2->data[by_current][bx_current][c];
                                dist += (bcolor - acolor)*(bcolor - acolor);
                            }
                        }

                        if(dist < dist_best)
                        {
                            bx_best = bx;
                            by_best = by;
                            dist_best = dist;
                        }
                    }
                }

                // Propagation (y):

                if(ay_p >= 0 && ay_p < eh)
                {
                    bx = (int)(nnf->data[ay_p][ax][0]);
                    by = (int)(nnf->data[ay_p][ax][1]) + y_step;

                    if(by >= 0 &&
                       by < eh &&
                       abs(ax - bx) <= max_offset &&
                       abs(ay - by) <= max_offset)
                    {
                        double dist = nnf->data[ay_p][ax][2];
                        int delta_add = PATCH_W-1;
                        int delta_remove = -1;
                        if (y_step < 0) {
                            delta_add = 0;
                            delta_remove = PATCH_W;
                        }
                        int ay_current = (ay + delta_remove);
                        int by_current = (by + delta_remove);
                        for (int dx = 0; dx < PATCH_W; dx++) {
                            int ax_current = (ax + dx);
                            int bx_current = (bx + dx);

                            for (int c = 0; c < 3; c++) {
                                real acolor = img1->data[ay_current][ax_current][c];
                                real bcolor = img2->data[by_current][bx_current][c];
                                dist -= (bcolor - acolor)*(bcolor - acolor);
                            }
                        }
                        
                        ay_current = (ay + delta_add);
                        by_current = (by + delta_add);
                        for (int dx = 0; dx < PATCH_W; dx++) {
                            int ax_current = (ax + dx);
                            int bx_current = (bx + dx);
                            for (int c = 0; c < 3; c++) {
                                real acolor = img1->data[ay_current][ax_current][c];
                                real bcolor = img2->data[by_current][bx_current][c];
                                dist += (bcolor - acolor)*(bcolor - acolor);
                            }
                        }

                        if(dist < dist_best)
                        {
                            bx_best = bx;
                            by_best = by;
                            dist_best = dist;
                        }
                    }
                }

                // Random search:

                for(rs_iter = 0; rs_iter < rs_iters; rs_iter++)
                {
                    int xmin = MAX(MAX(bx_best - rs_max, ax - max_offset), 0);
                    int xmax = MIN(MIN(bx_best + rs_max + 1, 
                        ax + max_offset + 1), ew);
                    int ymin = MAX(MAX(by_best - rs_max, ay - max_offset), 0);
                    int ymax = MIN(MIN(by_best + rs_max + 1, 
                        ay + max_offset + 1), eh);

                    bx = rand_range(xmin, xmax);
                    by = rand_range(ymin, ymax);

                    double dist = patch_dist(img1, img2, ax, ay, bx, by, 
                        dist_best);

                    if(dist < dist_best)
                    {
                        bx_best = bx;
                        by_best = by;
                        dist_best = dist;
                    }
                    
                }

                nnf->data[ay][ax][0] = bx_best;
                nnf->data[ay][ax][1] = by_best;
                nnf->data[ay][ax][2] = dist_best;
            }
        }
    }

    for(ay = max_offset; ay < eh - max_offset; ay += 10)
    {
        for(ax = max_offset; ax < ew - max_offset; ax += 10)
        {
            float u = nnf->data[ay][ax][0] - ax;
            float v = nnf->data[ay][ax][1] - ay;
            float scale = sqrt(u * u + v * v) / ((float)max_offset);
            float data_vis[2] = {u/((float)max_offset), u/((float)max_offset)};
            
            if (u < -max_offset || v < -max_offset || u > max_offset || v > max_offset) {
                printf("bad offset at %d, %d: %f %f\n", ax, ay, u, v);
                exit(1);
            }
            
            float color[3];

            find_color_value(data_vis, scale, color);
            draw_arrow((float)ay, (float)ax, (float)(ay + v), (float)(ax + u), output_img, color);
        }
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    FLOAT3_IMG *input_img1;
    FLOAT3_IMG *input_img2;
    
    if(argc == 1)
    {
        input_img1 = read_img_rgb("../../images/opt_flow1_small.png");
        input_img2 = read_img_rgb("../../images/opt_flow2_small.png");
    }
    else if(argc == 4)
    {
        input_img1 = read_img_rgb(argv[1]);
        input_img2 = read_img_rgb(argv[2]);
    }
    else
    {
        printf("Usage: ./a.out in1.png in2.png out.png or ./a.out\n");
        return 0;
    }

    FLOAT3_IMG *output_img = zeros_rgb(input_img1->rows, input_img1->cols);
    FLOAT3_IMG *nnf = zeros_rgb(input_img1->height - PATCH_W + 1, 
        input_img1->width - PATCH_W + 1);
    
    double d1 = wall_time();
    optical_flow(input_img1, input_img2, output_img, nnf);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
    {
        write_img_rgb(output_img, "test.png");
    }
    else
    {
        write_img_rgb(output_img, argv[3]);
    }
    
    free_img_rgb(input_img1);
    free_img_rgb(input_img2);
    free_img_rgb(output_img);
    free_img_rgb(nnf);

    return 0;
}
    
