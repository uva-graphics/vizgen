"""raytracer.py

A application that draws a ball in the image using ray tracing.

Author: Yuting Yang
"""

import numpy as np
import numpy
import sys; sys.path +=['../../compiler']
import util
#import skimage
#import skimage.io

def raytracer(time=0.0):
    """Draws the desired scene using ray tracing method.
    """
    
    # define view window size
    view = np.array([320.0, 480.0])
    
    # define eye position
    eye = np.array([160.0, 240.0, -800.0])
    
    # define sphere origin, radius and normalized color
    sphere_o = np.array([160.0, 240.0, 250.0])
    sphere_r = 150.0
    sphere_c = np.array([1.0, 0, 0])
    
    # define light source position
    #light = np.array([160.0, 400.0, -400.0])
    light = np.array([160.0 + 150.0 * np.cos(np.pi * time / 2.0), 240.0 + 150.0 * np.sin(np.pi * time / 2.0), -400.0])
    
    # define ambient
    ambient = 0.1
    
    # define background color
    background = np.array([0.0, 0.0, 0.0])
    
    diffuse = 0.6
    
    specular = 0.3
    alfa = 50
    
    output_img = np.zeros([int(view[0]), int(view[1]), 3])
    
    eye_to_sphere = sphere_o - eye
    ray_dir = np.array([0.0, 0.0, 0.0])
    
    for r in range(int(view[0])):
        for c in range(int(view[1])):
            ray_dir[0] = r
            ray_dir[1] = c
            
            direction = ray_dir - eye
            norm_direction = numpy.linalg.norm(direction)
            direction /= norm_direction
            v = numpy.dot(direction, eye_to_sphere)
            dot_product = numpy.dot(eye_to_sphere, eye_to_sphere)
            disc = sphere_r ** 2 - (dot_product - v ** 2)
            
            # when light don't intersect with the sphere
            if disc < 0:
                
                output_img[r, c, :] = background
                
            else:
                
                d = disc ** 0.5
                intersect_point = eye + (v - d) * direction
                sphere_normal = intersect_point - sphere_o
                norm_sphere_normal = numpy.linalg.norm(sphere_normal)
                sphere_normal /= norm_sphere_normal
                
                light_dir = light - intersect_point
                norm_light_dir = numpy.linalg.norm(light_dir)
                light_dir /= norm_light_dir
                dot_product = numpy.dot(light_dir, sphere_normal)
                R = 2.0 * dot_product * sphere_normal - light_dir
                diffuse_intensity = numpy.dot(light_dir, sphere_normal)
                dot_product = numpy.dot(R, direction)
                specular_intensity = dot_product ** alfa
                
                if diffuse_intensity < 0:
                
                    diffuse_intensity = 0.0
                    specular_intensity = 0.0
                
                output_intensity = (ambient + diffuse * diffuse_intensity + specular * specular_intensity)
                if output_intensity > 1:
                
                    output_intensity = 1
                
                output_img[r, c, :] = sphere_c * output_intensity
    
    return output_img
    
def test(time=0.0, n=None):
    """Default unit tests
    """
    util.is_initial_run = True
    ans = util.test_image_pipeline_filename(raytracer, (), n, name = 'ray_tracer', additional_args=(time,))
    return util.combine_tests([ans])
    
if __name__ == '__main__':
    test(time=0.4)

#for i in range(0, 40):
#    ans = test(time=i / 10.0)
#    skimage.io.imsave('output/' + str(i) + '.png', np.clip(ans['output'], 0, 1))
