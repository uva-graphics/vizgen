#ifndef _timer_h
#define _timer_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#ifdef IS_CAMERA_PIPE
#include "../apps/camera_pipe/c/stb_image.h"
#else
#include "stb_image.h"
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define abs(x) ((x)<0 ? (-x) : (x))

typedef struct
{
    int rows;
    int cols;
    int width;
    int height;
    real ***data;
    real *flat_data;
    real **pointer;
} FLOAT3_IMG;

typedef struct
{
    int rows;
    int cols;
    int width;
    int height;
    real **data;
    real *flat_data;
} FLOAT_IMG;

typedef struct
{
    int rows;
    int cols;
    int width;
    int height;
    real ***data;
    real *flat_data;
    real **pointer;
} FLOAT4_IMG;

#if VISUAL_STUDIO_WORKAROUND
#include <windows.h>

bool wall_time_initialized = false;
LARGE_INTEGER wall_time_rate, wall_time_start;

/*
Returns a double representing the current time as seconds since the unix epoch.
This is the windows implementation.
*/
double wall_time() 
{
    if (!wall_time_initialized) 
    {
        wall_time_initialized = true;
        QueryPerformanceFrequency(&wall_time_rate);
        QueryPerformanceCounter(&wall_time_start);
    }

    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);

    return (t.QuadPart - wall_time_start.QuadPart) * 1.0 / 
            wall_time_rate.QuadPart;
}

#else
#include <sys/time.h>

/*
Returns a double representing the current time as seconds since the unix epoch.
This is the unix implementation.
*/
double wall_time() 
{
    struct timeval t;
    gettimeofday(&t, NULL);
    
    return t.tv_sec + t.tv_usec * 1e-6;
}
#endif


/*
Writes an image out to filename and saves it as a PNG; image must be RGB
*/
int write_img_rgb(FLOAT3_IMG *img, char const *filename)
{
    int i, j, index;
    unsigned char *data;

    if(img == NULL)
    {
        printf("write_img_rgb: img pointer is null\n");
        return -1;
    }

    if(filename == NULL)
    {
        printf("write_img_rgb: filename pointer is null\n");
        return -1;
    }

    data = malloc(img->rows * img->cols * 3 * sizeof(unsigned char));

    for(i = 0; i < img->rows; i++)
    {
        for(j = 0; j < img->cols; j++)
        {
            index = i * img->cols * 3 + j * 3;

            data[index + 0] = (unsigned char)(img->data[i][j][0] * 255);
            data[index + 1] = (unsigned char)(img->data[i][j][1] * 255);
            data[index + 2] = (unsigned char)(img->data[i][j][2] * 255);
        }
    }

    return stbi_write_png(filename, img->width, img->height, 3, 
        (void *)data, img->cols * 3 * sizeof(unsigned char));
}

int write_img_gray(FLOAT_IMG *img, char const *filename)
{
    int i, j, index;
    unsigned char *data;

    if(img == NULL)
    {
        printf("write_img_gray: img pointer is null\n");
        return -1;
    }

    if(filename == NULL)
    {
        printf("write_img_gray: filename pointer is null\n");
        return -1;
    }

    data = malloc(img->rows * img->cols * sizeof(unsigned char));

    for(i = 0; i < img->rows; i++)
    {
        for(j = 0; j < img->cols; j++)
        {
            index = i * img->cols + j;
            data[index] = (unsigned char)(img->data[i][j] * 255);
        }
    }

    return stbi_write_png(filename, img->width, img->height, 1,
        (void *)data, img->cols * sizeof(unsigned char));
}

void print_img_rgb(FLOAT3_IMG *img)
{
    int i, j;

    if(img == NULL)
    {
        printf("print_img: input image pointer is null\n");
        return;
    }

    for(i = 0; i < img->rows; i++)
    {
        for(j = 0; j < img->cols; j++)
        {
            printf("(%d, %d):\tR: %f G: %f B: %f\n", i, j, img->data[i][j][0],
                img->data[i][j][1], img->data[i][j][2]);
        }
    }
}

void print_img_gray(FLOAT_IMG *img)
{
    int i, j;

    if(img == NULL)
    {
        printf("print_img: input image pointer is null\n");
        return;
    }

    for(i = 0; i < img->rows; i++)
    {
        for(j = 0; j < img->cols; j++)
        {
            printf("%.3f ", img->data[i][j]);
        }

        printf("\n");
    }
}

/*
Reads in the image at location filename and returns a FLOAT3_IMG representation
of it; the returned image is 3 channel and each pixel color value is a real
between 0.0 and 1.0
*/
FLOAT3_IMG *read_img_rgb(char *filename)
{
    int i, j, index, w, h, d;
    FLOAT3_IMG *img = malloc(sizeof(FLOAT3_IMG));
    unsigned char *raw_data = stbi_load(filename, &w, &h, &d, 0);

    if(!raw_data)
    {
        printf("read_img_rgb: error when reading in image %s\n", filename);
        return NULL;
    }

    img->rows = img->height = h;
    img->cols = img->width = w;
    
    img->flat_data = malloc(img->rows * img->cols * d * sizeof(real));
    img->pointer = malloc(img->rows * img->cols * sizeof(real *) + img->rows * sizeof(real **));
    img->data = (real ***)img->pointer;

    // printf("image size: w = %d, h = %d, d = %d\n", w, h, d);

    if(d != 3)
    {
        printf("read_img_rgb: # of color channels is not 3 (channels = %d)\n",
            d); 
        return NULL;
    }

    for(i = 0; i < img->rows; i++)
    {
        img->data[i] = (real **)&img->pointer[img->rows + i * img->cols];

        for(j = 0; j < img->cols; j++)
        {
            index = i * img->cols * (d) + j * (d);
            
            img->data[i][j] = &img->flat_data[index];
            img->data[i][j][0] = (real)(raw_data[index + 0]) / 255.0f;
            img->data[i][j][1] = (real)(raw_data[index + 1]) / 255.0f;
            img->data[i][j][2] = (real)(raw_data[index + 2]) / 255.0f;
        }
    }

    stbi_image_free(raw_data);

    return img;
}

void free_img_rgb(FLOAT3_IMG *img)
{
    free(img->flat_data);
    free(img->pointer);
    free(img);
}

FLOAT_IMG *read_img_gray(char *filename)
{
    int i, j, index, w, h, d;
    FLOAT_IMG *img = malloc(sizeof(FLOAT_IMG));
    unsigned char *raw_data = stbi_load(filename, &w, &h, &d, 0);

    if(!raw_data)
    {
        printf("read_img_rgb: error when reading in image %s\n", filename);
        return NULL;
    }

    img->rows = img->height = h;
    img->cols = img->width = w;
    //img->data = malloc(img->rows * sizeof(real *));
    img->flat_data = malloc(img->rows * img->cols * sizeof(real *));
    img->data = malloc(img->rows * sizeof(real *));

    // printf("image size: w = %d, h = %d, d = %d\n", w, h, d);

    if(d != 1)
    {
        printf("read_img_gray: # of color channels is not 1 (channels = %d)\n",
            d); 
        return NULL;
    }

    for(i = 0; i < img->rows; i++)
    {
        img->data[i] = &img->flat_data[i * img->cols];

        for(j = 0; j < img->cols; j++)
        {
            index = i * img->cols * (d) + j * (d);
            img->data[i][j] = (real)(raw_data[index]) / 255.0f;
        }
    }

    stbi_image_free(raw_data);

    return img;
}

void free_img_gray(FLOAT_IMG *img)
{    
    free(img->data);
    free(img->flat_data);
    free(img);   
}

FLOAT4_IMG *read_img_rgba(char *filename)
{
    int i, j, index, w, h, d;
    FLOAT4_IMG *img = malloc(sizeof(FLOAT4_IMG));
    unsigned char *raw_data = stbi_load(filename, &w, &h, &d, 0);

    if(!raw_data)
    {
        printf("read_img_rgb: error when reading in image %s\n", filename);
        return NULL;
    }

    img->rows = img->height = h;
    img->cols = img->width = w;
    img->flat_data = malloc(img->rows * img->cols * d * sizeof(real));
    img->pointer = malloc(img->rows * sizeof(real **) + img->rows * img->cols * sizeof(real *));
    img->data = (real ***)img->pointer;
    //img->data = malloc(img->rows * sizeof(real **));

    // printf("image size: w = %d, h = %d, d = %d\n", w, h, d);

    if(d != 4)
    {
        printf("read_img_rgb: # of color channels is not 4 (channels = %d)\n",
            d); 
        return NULL;
    }

    for(i = 0; i < img->rows; i++)
    {
        img->data[i] = &img->pointer[img->rows + i * img->cols];

        for(j = 0; j < img->cols; j++)
        {
            index = i * img->cols * (d) + j * (d);
            img->data[i][j] = &img->flat_data[index];

            img->data[i][j][0] = (real)(raw_data[index + 0]) / 255.0f;
            img->data[i][j][1] = (real)(raw_data[index + 1]) / 255.0f;
            img->data[i][j][2] = (real)(raw_data[index + 2]) / 255.0f;
            img->data[i][j][3] = (real)(raw_data[index + 3]) / 255.0f;
        }
    }

    stbi_image_free(raw_data);

    return img;
}

void free_img_rgba(FLOAT4_IMG *img)
{
    free(img->pointer);
    
    free(img->flat_data);
    free(img);
}

/*
Creates a blank image where every pixel and color channel is equal to zero. The
"rgb" means that a 3-channel image is returned.
*/
FLOAT3_IMG *zeros_rgb(int rows, int cols)
{
    int i, j;
    FLOAT3_IMG *img = malloc(sizeof(FLOAT3_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    img->flat_data = malloc(img->rows * img->cols * 3 * sizeof(real));
    img->pointer = malloc(img->rows * sizeof(real **) + img->rows * img->cols * sizeof(real *));
    img->data = (real ***)img->pointer;
    
    int index;

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->pointer[img->rows + i * img->cols];

        for(j = 0; j < cols; j++)
        {
            index = i * img->cols * 3 + j * 3;
            img->data[i][j] = &img->flat_data[index];

            img->data[i][j][0] = 0.0f;
            img->data[i][j][1] = 0.0f;
            img->data[i][j][2] = 0.0f;
        }
    }

    return img;
}

FLOAT_IMG *zeros_gray(int rows, int cols)
{
    int i, j, index;
    FLOAT_IMG *img = malloc(sizeof(FLOAT_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    img->flat_data = malloc(img->rows * img->cols * sizeof(real));
    img->data = malloc(img->rows * sizeof(real *));

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->flat_data[i * img->cols];

        for(j = 0; j < cols; j++)
        {
            img->data[i][j] = 0.0f;
        }
    }

    return img;
}

void seed_rng()
{
    srand(time(NULL));
}

FLOAT_IMG *random_gray(int rows, int cols)
{
    int i, j, index;
    FLOAT_IMG *img = malloc(sizeof(FLOAT_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    img->flat_data = malloc(img->rows * img->cols * sizeof(real));
    img->data = malloc(img->rows * sizeof(real *));

    seed_rng();

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->flat_data[i * img->cols];

        for(j = 0; j < cols; j++)
        {
            img->data[i][j] = ((real)(rand())) / ((real)RAND_MAX);
        }
    }

    return img;
}

FLOAT_IMG *random_alpha(int rows, int cols)
{
    int i, j, index;
    FLOAT_IMG *img = malloc(sizeof(FLOAT_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    img->flat_data = malloc(img->rows * img->cols * sizeof(real));
    img->data = malloc(img->rows * sizeof(real *));

    seed_rng();

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->flat_data[i * img->cols];

        for(j = 0; j < cols; j++)
        {
            if(((real)(rand())) / ((real)RAND_MAX) > 0.5)
            {
                img->data[i][j] = 1.0f;
            }
            else
            {
                img->data[i][j] = 0.0f;
            }
        }
    }

    return img;
}

FLOAT4_IMG *zeros_rgba(int rows, int cols)
{
    int i, j, index;
    FLOAT4_IMG *img = malloc(sizeof(FLOAT4_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    //img->data = malloc(img->rows * sizeof(real **));
    img->flat_data = malloc(img->rows * img->cols * 4 * sizeof(real));
    img->pointer = malloc(img->rows * sizeof(real **) + img->rows * img->cols * sizeof(real *));
    img->data = (real ***)img->pointer;

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->pointer[img->rows + i * img->cols];

        for(j = 0; j < cols; j++)
        {
            if(i == 2558 && j == 1024)
                i = 2558;
            index = i * img->cols * 4 + j * 4;
            img->data[i][j] = &img->flat_data[index];

            img->data[i][j][0] = 0.0f;
            img->data[i][j][1] = 0.0f;
            img->data[i][j][2] = 0.0f;
            img->data[i][j][3] = 0.0f;
        }
    }

    return img;
}

/*
Creates an image where every pixel and color channel is equal to one. The
"rgb" means that a 3-channel image is returned.
*/
FLOAT3_IMG *ones_rgb(int rows, int cols)
{
    int i, j;
    FLOAT3_IMG *img = malloc(sizeof(FLOAT3_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    img->flat_data = malloc(img->rows * img->cols * 3 * sizeof(real));
    img->pointer = malloc(img->rows * sizeof(real **) + img->rows * img->cols * sizeof(real *));
    img->data = (real ***)img->pointer;
    
    int index;

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->pointer[img->rows + i * img->cols];

        for(j = 0; j < cols; j++)
        {
            index = i * img->cols * 3 + j * 3;
            img->data[i][j] = &img->flat_data[index];

            img->data[i][j][0] = 1.0f;
            img->data[i][j][1] = 1.0f;
            img->data[i][j][2] = 1.0f;
        }
    }

    return img;
}

FLOAT_IMG *ones_gray(int rows, int cols)
{
    int i, j, index;
    FLOAT_IMG *img = malloc(sizeof(FLOAT_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    img->flat_data = malloc(img->rows * img->cols * sizeof(real));
    img->data = malloc(img->rows * sizeof(real *));

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->flat_data[i * img->cols];

        for(j = 0; j < cols; j++)
        {
            img->data[i][j] = 1.0f;
        }
    }

    return img;
}

FLOAT4_IMG *ones_rgba(int rows, int cols)
{
    int i, j, index;
    FLOAT4_IMG *img = malloc(sizeof(FLOAT4_IMG));

    img->rows = img->height = rows;
    img->cols = img->width = cols;
    //img->data = malloc(img->rows * sizeof(real **));
    img->flat_data = malloc(img->rows * img->cols * 4 * sizeof(real));
    img->pointer = malloc(img->rows * sizeof(real **) + img->rows * img->cols * sizeof(real *));
    img->data = (real ***)img->pointer;

    for(i = 0; i < rows; i++)
    {
        img->data[i] = &img->pointer[img->rows + i * img->cols];

        for(j = 0; j < cols; j++)
        {
            index = i * img->cols * 4 + j * 4;
            img->data[i][j] = &img->flat_data[index];

            img->data[i][j][0] = 1.0f;
            img->data[i][j][1] = 1.0f;
            img->data[i][j][2] = 1.0f;
            img->data[i][j][3] = 1.0f;
        }
    }

    return img;
}

/*
Calculates the average difference between two images, pixel-by-pixel
*/
double imgcmp(FLOAT3_IMG *img1, FLOAT3_IMG *img2)
{
    int i, j;
    double diff;

    if(img1 == NULL)
    {
        printf("imgcmp: img1 pointer is null\n");
        return -1;
    }

    if(img2 == NULL)
    {
        printf("imgcmp: img2 pointer is null\n");
        return -1;
    }

    if(img1->width != img2->width || img1->height != img2->height)
    {
        printf("imgcmp: images do not have the same dimension\n");
        return -1;
    }

    for(i = 0; i < img1->rows; i++)
    {
        for(j = 0; j < img1->cols; j++)
        {
            diff += abs(img1->data[i][j][0] - img2->data[i][j][0]);
            diff += abs(img1->data[i][j][1] - img2->data[i][j][1]);
            diff += abs(img1->data[i][j][2] - img2->data[i][j][2]);
        }
    }

    return diff / (double)(img1->rows * img1->cols * 3);
}

/*
Sum of squared difference
*/
double imgcmp_gray(FLOAT_IMG *img1, FLOAT_IMG *img2)
{
    int i, j;
    double result = 0.0;

    if(img1 == NULL)
    {
        printf("imgcmp: img1 pointer is null\n");
        return -1;
    }

    if(img2 == NULL)
    {
        printf("imgcmp: img2 pointer is null\n");
        return -1;
    }

    if(img1->width != img2->width || img1->height != img2->height)
    {
        printf("imgcmp: images do not have the same dimension\n");
        return -1;
    }

    for(i = 0; i < img1->rows; i++)
    {
        for(j = 0; j < img1->cols; j++)
        {
            result += pow(img1->data[i][j] - img2->data[i][j], 2.0);
        }
    }

    return sqrt(result);
}
#endif