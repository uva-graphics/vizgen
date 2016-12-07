"""raytracer.py

A application that draws a ball in the image using ray tracing.

Author: Yuting Yang
"""

import numpy as np
import sys; sys.path += ['../../compiler']
import util

DEFAULT_WIDTH  = 960//16
DEFAULT_HEIGHT = 640//16

#transform(TypeSpecialize(checks=False))
def get_sphere_normal(sphere, direction, eye):

    """Calculate the sphere's normal at the intersection of the sphere and the view's direction
    """
    
    eye_to_sphere = sphere['origin'] - eye
    v = 0.0
    for k in range(len(direction)):
        v += direction[k] * eye_to_sphere[k]
        
    if v <= 0:
    
        return ([0, 0, 0], [0, 0, 0], float('Inf'))
    
    norm_eye_to_sphere = 0.0
    for k in range(len(eye_to_sphere)):
        norm_eye_to_sphere += eye_to_sphere[k] ** 2
        
    disc = sphere['radius']**2 - (norm_eye_to_sphere - v**2)
    
    # if sphere doesn't intersect with eye's view
    if disc < 0:
        
        return ([0, 0, 0], [0, 0, 0], float('Inf'))
        
    else:
        
        d = disc ** 0.5
        intersect_point = eye + (v - d) * direction
        sphere_normal = intersect_point - sphere['origin']
        norm_sphere_normal = 0.0
        for k in range(len(sphere_normal)):
            norm_sphere_normal += sphere_normal[k] ** 2
        norm_sphere_normal = norm_sphere_normal ** 0.5
        sphere_normal /= norm_sphere_normal
        return (sphere_normal, intersect_point, v - d)


#transform(TypeSpecialize(checks=False))
def calculate_intensity(light_dir, intersect_point, normal, view_direction, isglass):
    """Calculates the intensity (diffusion and reflection) resulting from a light source
    """
    
    alfa = 50
    
    shade = 0.0
    for k in range(len(light_dir)):
        shade += light_dir[k] * normal[k]
    reflection = 0.0
    R = np.array([0.0, 0.0, 0.0])
    
    if shade < 0.0:
    
        shade = 0.0
        
    else:
    
        if isglass:
    
            light_dir_dot_normal = 0.0
            for k in range(len(light_dir)):
                light_dir_dot_normal += light_dir[k] * normal[k]
            R = 2.0 * light_dir_dot_normal * normal - light_dir
            norm_R = 0.0
            for k in range(len(R)):
                norm_R += R[k] ** 2
            norm_R = norm_R ** 0.5
            R /= norm_R
            reflection = 0.0
            for k in range(len(R)):
                reflection += R[k] * view_direction[k]
            reflection = reflection ** alfa
            
    return(shade, reflection)
    
#transform(TypeSpecialize(checks=False))
def check_blockage(intersect_point, light, sphere_group, intersect_index):
    """Check whether light is blocked by other objects at a certain point_color
    """
    
    return_value = True
    
    for k in range(len(sphere_group)):
    
        if k != intersect_index:
    
            origin = sphere_group[k]['origin']
            radius = sphere_group[k]['radius']
            
            cross_product = np.zeros(3)
            cross_product[0] = (light[1] - intersect_point[1]) * (intersect_point[2] - origin[2]) - (light[2] - intersect_point[2]) * (intersect_point[1] - origin[1])
            cross_product[1] = (light[2] - intersect_point[2]) * (intersect_point[0] - origin[0]) - (light[0] - intersect_point[0]) * (intersect_point[2] - origin[2])
            cross_product[2] = (light[0] - intersect_point[0]) * (intersect_point[1] - origin[1]) - (light[1] - intersect_point[1]) * (intersect_point[0] - origin[0])
            norm_cross_product = (cross_product[0] ** 2 + cross_product[1] ** 2 + cross_product[2] ** 2) ** 0.5
            norm_light_intersect_point = 0.0
            for c in range(len(light)):
                norm_light_intersect_point += (light[c] - intersect_point[c]) ** 2
            norm_light_intersect_point = norm_light_intersect_point ** 0.5
            disc = abs(norm_cross_product / norm_light_intersect_point)
        
            if disc <= radius:
        
                dot_product1 = 0.0
                dot_product2 = 0.0
                for c in range(len(intersect_point)):
                    dot_product1 += (intersect_point[c] - origin[c]) * (light[c] - intersect_point[c])
                    dot_product2 += (light[c] - intersect_point[c]) * (light[c] - intersect_point[c])
                t = -dot_product1 / dot_product2
            
                if t < 0:
            
                    norm_return = 0.0
                    for c in range(len(intersect_point)):
                        norm_return += (intersect_point[c] - origin[c]) ** 2
                    norm_return = norm_return ** 2
                    return_value = return_value and norm_return >= radius
            
                else:
                
                    if t > 1:
                
                        norm_return = 0.0
                        for c in range(len(light)):
                            norm_return += (light[c] - origin[c]) ** 2
                        norm_return = norm_return ** 2
                        return_value = return_value and norm_return >= radius
                
                    else:
                
                        return False
    
    return return_value
    
#transform(TypeSpecialize(checks=False))
def get_color(scene, parameter, direction, depth, cutoff):
    """Calculate the color of a certain point between the objects defined by scene and parameter,
    and from the view designated by direction. depth counts the number of iteration needed,
    and cutoff gives a threshold for terminating the iteration.
    """
    
    intensity = np.array([0.0, 0.0, 0.0])
    shade = 0.0
    reflection = 0.0
    disc_min = float('Inf')
    intersect_index = -1
    
    for k in range(len(scene['objects'])):
                
        sphere_normal, intersect_point, disc = get_sphere_normal(scene['objects'][k], direction, scene['eye_position'])
        
        if disc < disc_min:
                
            disc_min = disc
            intersect_index = k
    
    sphere_normal, intersect_point, disc = get_sphere_normal(scene['objects'][intersect_index], direction, scene['eye_position'])
        
    light_dir = scene['light_source'] - intersect_point
    norm_light_dir = 0.0
    for c in range(len(light_dir)):
        norm_light_dir += light_dir[c] ** 2
    norm_light_dir = norm_light_dir ** 0.5
    light_dir /= norm_light_dir
    
    norm_sphere_normal = 0.0
    for c in range(len(sphere_normal)):
        norm_sphere_normal += sphere_normal[c] ** 2
    norm_sphere_normal = norm_sphere_normal ** 0.5
    if norm_sphere_normal == 0:
        
        intensity = intensity
        
    else:
    
        if intersect_index == 0:
        
            #using spherical coordinate
            theta = np.arccos(sphere_normal[2])
            phi = np.arctan2(sphere_normal[1], sphere_normal[0])        
            
            #map angles to the bump map
            u = int(scene['bump_u'].shape[1] * (phi + np.pi) / (2 * np.pi))
            v = int(scene['bump_v'].shape[0] * theta / np.pi)
            point_color = scene['tennis_color'][v - 1, u - 1, :]
            
            if check_blockage(intersect_point, scene['light_source'], scene['objects'], intersect_index):
            
                new_normal_x = sphere_normal[0] + scene['bump_u'][v - 1, u - 1] * np.cos(phi)
                new_normal_y = sphere_normal[1] + scene['bump_u'][v - 1, u - 1] * np.sin(phi)
                new_normal_z = sphere_normal[2] - scene['bump_v'][v - 1, u - 1]
                new_normal = [new_normal_x, new_normal_y, new_normal_z]
                norm_new_normal = 0.0
                for c in range(len(new_normal)):
                    norm_new_normal += new_normal[c] ** 2
                norm_new_normal = norm_new_normal ** 0.5
                new_normal /= norm_new_normal
                normal = new_normal
                shade, reflection = calculate_intensity(light_dir, intersect_point, normal, direction, isglass = False)
                
            intensity = intensity + point_color * (parameter['ambient'] + parameter['diffuse'] * shade + parameter['specular'] * reflection)
                
        else:
            
            point_color = scene['objects'][intersect_index]['color']
            normal = sphere_normal
            
            if check_blockage(intersect_point, scene['light_source'], scene['objects'], intersect_index):
                shade, reflection = calculate_intensity(light_dir, intersect_point, normal, direction, isglass = True)
                
            intensity = intensity + point_color * (parameter['ambient'] + parameter['diffuse'] * shade + parameter['specular'] * reflection)
            
            if depth > 0 and parameter['kspec'] > cutoff:
        
                dot_product = 0.0
                for c in range(len(direction)):
                    dot_product += -direction[c] * normal[c]
                dot_product2 = 0.0
                reflection_direction = 2.0 * dot_product * normal + direction
                
                norm_reflection_direction = 0.0
                for c in range(len(reflection_direction)):
                    norm_reflection_direction += reflection_direction ** 2
                norm_reflection_direction = norm_reflection_direction ** 0.5
                new_scene = {'objects': scene['objects'], 'light_source': scene['light_source'], 'eye_position': intersect_point, 'bump_u': scene['bump_u'], 'bump_v': scene['bump_v'], 'tennis_color': scene['tennis_color']}
                
                add_intensity = get_color(new_scene, parameter, reflection_direction, depth - 1, cutoff / parameter['kspec'])
                intensity[0] += point_color[0] * parameter['kspec'] * add_intensity[0]
                intensity[1] += point_color[1] * parameter['kspec'] * add_intensity[1]
                intensity[2] += point_color[2] * parameter['kspec'] * add_intensity[2]
    
    return intensity
    
#transform(TypeSpecialize(checks=False))
def raytracer(input_img1, input_img2, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Draws the desired scene using ray tracing method.
    """
    
    # define view window size
    view = np.array([float(height), float(width)])
    
    # define eye position
    eye = np.array([320.0, 480.0, -1600.0])
    
    # define sphere origin, radius and normalized color
    sphere_o = np.array([320.0, 480.0, 800.0])
    sphere_r = 200.0
    sphere_c = np.array([0.0, 1.0, 1.0])
    sphere_tennis = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([320.0, 160.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass1 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([320.0, 800.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass2 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([640.0, 480.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass3 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([0.0, 480.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass4 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([546.0, 706.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass5 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([93.0, 706.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass6 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([93.0, 253.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass7 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    sphere_o = np.array([546.0, 253.0, 900.0])
    sphere_r = 100.0
    sphere_c = np.array([1.0, 1.0, 1.0])
    sphere_glass8 = {'origin': sphere_o, 'radius': sphere_r, 'color': sphere_c}
    
    
    sphere_group = [sphere_tennis, sphere_glass1, sphere_glass2, sphere_glass3, sphere_glass4, sphere_glass5, sphere_glass6, sphere_glass7, sphere_glass8]
    
    # define light source position
    light = np.array([320.0, 480.0, 0.0])
    #ight = [320.0, 1000.0, 500.0]
    
    # define ambient
    ambient = 0.2
    
    
    
    diffuse = 1.0 - ambient
    specular = 0.7
    alfa = 50
    kspec = 0.9
    
    parameter = {'ambient': ambient, 'diffuse': diffuse, 'specular': specular, 'alfa': alfa, 'kspec': kspec}
    
    output_img = np.zeros([int(view[0]), int(view[1]), 3])
    
    # add bump to sphere
    bump_map = 0.299 * input_img1[:, :, 0] + 0.587 * input_img1[:, :, 1] + 0.114 * input_img1[:,:,2]
    bump_gradient = np.gradient(bump_map)
    bump_u = 2.0 * bump_gradient[0]
    bump_v = 2.0 * bump_gradient[1]
    scene = {'objects': sphere_group, 'light_source': light, 'eye_position': eye, 'bump_u': bump_u, 'bump_v': bump_v, 'tennis_color': input_img2}
    
    for r in range(int(view[0])):
        for c in range(int(view[1])):
        
            direction = np.array([r/view[0]*640.0, c/view[1]*960.0, 0.0]) - eye
            norm_direction = 0.0
            for k in range(len(direction)):
                norm_direction += direction[k] ** 2
            norm_direction = norm_direction ** 0.5
            direction /= norm_direction
            
            intensity = get_color(scene, parameter, direction, depth = 3, cutoff = 0.1)
            
            output_img[r, c, 0] = intensity[0] / (1 + intensity[0])
            output_img[r, c, 1] = intensity[1] / (1 + intensity[1])
            output_img[r, c, 2] = intensity[2] / (1 + intensity[2])
    
    return output_img

input_img1 = util.image_filename('ball_bump_map.png')
input_img2 = util.image_filename('ball_color_map.png')
    
    
def test(n=1, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Default unit tests
    """
    util.is_initial_run = True
    ans = util.test_image_pipeline_filename(raytracer, [input_img1, input_img2], n, name = 'ray_tracer', additional_args=(width, height))
    return util.combine_tests([ans])
    
if __name__ == '__main__':
    test()
