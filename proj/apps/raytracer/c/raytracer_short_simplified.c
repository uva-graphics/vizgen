#include "c_util.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>

void raytracer(FLOAT3_IMG *output_img, float t)
{
    int r, c, k;  

    float eye[3];
    eye[0] = 160.0f;
    eye[1] = 240.0f;
    eye[2] = -800.0f;

    float sphere_o[3];
    sphere_o[0] = 160.0f;
    sphere_o[1] = 240.0f;
    sphere_o[2] = 250.0f;

    float sphere_r = 150.0;

    float sphere_c[3];
    sphere_c[0] = 1.0f;
    sphere_c[1] = 0.0f;
    sphere_c[2] = 0.0f;

    float light[3];
    light[0] = 160.0f + 150.0f * cos(M_PI * t / 2.0f);
    light[1] = 240.0f + 150.0f * sin(M_PI * t / 2.0f);
    light[2] = -400.0f;

    float ambient = 0.1f;

    float background[3];
    background[0] = 0.0f;
    background[1] = 0.0f;
    background[2] = 0.0f;

    float diffuse = 0.6f;
    float specular = 0.3f;
    float alfa = 50.0f;

    float eye_to_sphere[3];
    for(k = 0; k < 3; k++)
    {
        eye_to_sphere[k] = sphere_o[k] - eye[k];
    }

    float *ray_dir;

    float *direction;
    float norm_direction;
    float v;
    float dot_product;
    float disc;
    float d;
    float *intersect_point;
    float *sphere_normal;
    float norm_sphere_normal;
    float *light_dir;
    float norm_light_dir;
    float *R;
    float diffuse_intensity;
    float specular_intensity;
    float output_intensity;

    #pragma omp parallel for private(c, k, ray_dir, direction, norm_direction, v, dot_product, disc, d, intersect_point, sphere_normal, norm_sphere_normal, light_dir, norm_light_dir, R, diffuse_intensity, specular_intensity, output_intensity)
    for(r = 0; r < output_img->rows; r++)
    {
	ray_dir = malloc(3 * sizeof(float));
	ray_dir[0] = 0.0f;
	ray_dir[1] = 0.0f;
	ray_dir[2] = 0.0f;
	
	direction = malloc(3 * sizeof(float));
	intersect_point = malloc(3 * sizeof(float));
	sphere_normal = malloc(3 * sizeof(float));
	light_dir = malloc(3 * sizeof(float));
	R = malloc(3 * sizeof(float));
	for(c = 0; c < output_img->cols; c++)
	{
	    ray_dir[0] = r;
	    ray_dir[1] = c;

	    for(k = 0; k < 3; k++)
	    {
		direction[k] = ray_dir[k] - eye[k];
	    }

	    norm_direction = 0.0f;
	    for(k = 0; k < 3; k++)
	    {
		norm_direction += direction[k] * direction[k];
	    }
		norm_direction = sqrt(norm_direction);

	    for(k = 0; k < 3; k++)
	    {
		direction[k] /= norm_direction;
	    }

	    v = 0.0f;
	    for(k = 0; k < 3; k++)
	    {
		v += direction[k] * eye_to_sphere[k];
	    }

	    dot_product = 0.0f;
	    for(k = 0; k < 3; k++)
	    {
		dot_product += eye_to_sphere[k] * eye_to_sphere[k];
	    }

	    disc = sphere_r * sphere_r - (dot_product - v * v);

	    if(disc < 0)
	    {
		for(k = 0; k < 3; k++)
		{
		    output_img->data[r][c][k] = background[k];
		}
	    }
	    else
	    {
		d = sqrt(disc);
		for(k = 0; k < 3; k++)
		{
		    intersect_point[k] = eye[k] + (v - d) * direction[k];
		}

		for(k = 0; k < 3; k++)
		{
		    sphere_normal[k] = intersect_point[k] - sphere_o[k];
		}

		norm_sphere_normal = 0.0f;
		for(k = 0; k < 3; k++)
		{
		    norm_sphere_normal += sphere_normal[k] * sphere_normal[k];
		}
		norm_sphere_normal = sqrt(norm_sphere_normal);

		for(k = 0; k < 3; k++)
		{
		    sphere_normal[k] /= norm_sphere_normal;
		}

		for(k = 0; k < 3; k++)
		{
		    light_dir[k] = light[k] - intersect_point[k];
		}

		norm_light_dir = 0.0f;
		for(k = 0; k < 3; k++)
		{
		    norm_light_dir += light_dir[k] * light_dir[k];
		}
		norm_light_dir = sqrt(norm_light_dir);

		for(k = 0; k < 3; k++)
		{
		    light_dir[k] /= norm_light_dir;
		}

		dot_product = 0.0f;
		for(k = 0; k < 3; k++)
		{
		    dot_product += light_dir[k] * sphere_normal[k];
		}

		for(k = 0; k < 3; k++)
		{
		    R[k] = 2.0 * dot_product * sphere_normal[k] - light_dir[k];
		}

		diffuse_intensity = 0.0f;
		for(k = 0; k < 3; k++)
		{
		    diffuse_intensity += light_dir[k] * sphere_normal[k];
		}

		dot_product = 0.0f;
		for(k = 0; k < 3; k++)
		{
		    dot_product += R[k] * direction[k];
		}

		specular_intensity = pow(dot_product, alfa);

		if(diffuse_intensity < 0)
		{
		    diffuse_intensity = 0.0f;
		    specular_intensity = 0.0f;
		}

		output_intensity = (ambient + diffuse * diffuse_intensity + specular * specular_intensity);

		if(output_intensity > 1)
		{
		    output_intensity = 1.0f;
		}

		for(k = 0; k < 3; k++)
		{
		    output_img->data[r][c][k] = sphere_c[k] * output_intensity;
		}
	    }
        }
	
	free(direction);
	free(intersect_point);
	free(sphere_normal);
	free(light_dir);
	free(R);
    }
}

int main(int argc, char **argv)
{
    if(argc > 3)
    {
	printf("Usage: ./a.out out.png or ./a.out\n");
	return 0;
    }
    
    int view[2];
    view[0] = 320;
    view[1] = 480;
    FLOAT3_IMG *output_img = zeros_rgb(view[0], view[1]);
    
    float t;
    
    if(argc == 3)
	t = atof(argv[2]);
    else
	t = 0.0f;
    
    double d1 = wall_time();
    raytracer(output_img, t);
    double d2 = wall_time();
    
    printf("%f\n", d2 - d1);
    
    if(argc == 1)
	write_img_rgb(output_img, "test.png");
    else
	write_img_rgb(output_img, argv[1]);
    
    free_img_rgb(output_img);
}
