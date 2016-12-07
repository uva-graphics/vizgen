
cimport numpy
cimport numpy as np
import numpy
cimport cython
cimport cython.parallel


import util
import numpy








"""
C functions called by text-replacing macros in macros.py.
"""

cdef extern from "stdbool.h":
  pass

cimport cython
cimport numpy
cimport libc.math
import random
from libcpp cimport bool

def range_shuffled(*args):
    L = list(range(*args))
    L0 = list(L)
    random.shuffle(L)
    while len(L) >= 2 and L == L0:
        random.shuffle(L)
    return L

cdef double numpy_clip_double(double x, double a, double b) nogil:
    if x < a:
        return a
    elif x > b:
        return b
    return x

cdef int numpy_clip_int(int x, int a, int b) nogil:
    if x < a:
        return a
    elif x > b:
        return b
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:] numpy_clip_vec3f_ptr(float *x, float a, float b):
    cdef float[:] ans
    cdef int i
    cdef float x_i
    ans = numpy.empty(3, 'float32')
    for i in range(3):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] numpy_clip_vec3g_ptr(double *x, double a, double b):
    cdef double[:] ans
    cdef int i
    cdef double x_i
    ans = numpy.empty(3, 'float64')
    for i in range(3):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:] numpy_clip_vec3f(float[:] x, float a, float b):
    cdef float[:] ans
    cdef int i
    cdef float x_i
    ans = numpy.empty(3, 'float32')
    for i in range(3):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] numpy_clip_vec3g(double[:] x, double a, double b):
    cdef double[:] ans
    cdef int i
    cdef double x_i
    ans = numpy.empty(3, 'float64')
    for i in range(3):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans

cdef double square_double(double x) nogil:
    return x*x

cdef int square_int(int x) nogil:
    return x*x

cdef int int_to_int(int x) nogil:
    return x

cdef int float_to_int(float x) nogil:
    return <int>(x)

cdef float int_to_float(int x) nogil:
    return <float>(x)

cdef float float_to_float(float x) nogil:
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float numpy_linalg_norm_vec3f_ptr(float *v) nogil:
    return libc.math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double numpy_linalg_norm_vec3g_ptr(double *v) nogil:
    return libc.math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float numpy_dot_vec3f_ptr(float *u, float *v) nogil:
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double numpy_dot_vec3g_ptr(double *u, double *v) nogil:
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]

cdef extern from "vector_headers.h":
    ctypedef float v2float
    ctypedef float v4float
    ctypedef double v2double
    ctypedef double v4double

"""raytracer.py

A application that draws a ball in the image using ray tracing.

Author: Yuting Yang
"""

import numpy as np
import sys

sys.path += ['../../compiler']
import util
cdef bool _prealloc_cross_product_init, _prealloc_return_value_init, _prealloc_bump_u_init, _prealloc_bump_v_init, _prealloc_parameter_init, _prealloc_intensity_init
_prealloc_intensity_init = False
_prealloc_cross_product_init = False
_prealloc_return_value_init = False
_prealloc_bump_u_init = False
_prealloc_bump_v_init = False
_prealloc_parameter_init = False

DEFAULT_WIDTH  = 960 // 10
DEFAULT_HEIGHT = 640 // 10

def get_sphere_normal(sphere, direction, eye):
    #_chosen_typespecialize None
    if (isinstance(sphere, numpy.ndarray) and sphere.dtype == numpy.float64 and sphere.ndim == 2) and (isinstance(direction, numpy.ndarray) and direction.dtype == numpy.float64 and direction.ndim == 1) and (isinstance(eye, numpy.ndarray) and eye.dtype == numpy.float64 and eye.ndim == 1):
        return get_sphere_normal_array2float64_array1float64_array1float64_typespec_(sphere, direction, eye)
    
    
    """Calculate the sphere's normal at the intersection of the sphere and the view's direction
        return_value = 
        [ sphere_normal
          intersect_point
          disc
        ]
    """
    
    
    global _prealloc_return_value_global, _prealloc_return_value_init
    if not _prealloc_return_value_init or any([[3, 3][_prealloc_j] > _prealloc_return_value_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_return_value_global).shape))]):
        _prealloc_return_value_init = True
        _prealloc_return_value_global = np.zeros([3, 3])
    return_value = _prealloc_return_value_global
            
    
    eye_to_sphere = sphere[0, :] - eye
    v = 0.0
    for k in range(len(direction)):
        v += direction[k] * eye_to_sphere[k]
        
    if v <= 0:
        return_value[2, 0] = 1e20
        return return_value
    
    norm_eye_to_sphere = 0.0
    for k in range(len(eye_to_sphere)):
        norm_eye_to_sphere += eye_to_sphere[k] ** 2
        
    disc = sphere[1, 0]**2 - (norm_eye_to_sphere - v**2)
    
    # if sphere doesn't intersect with eye's view
    if disc < 0:
        
        return_value[2, 0] = 1e20
        return return_value
        
    else:
        
        d = pow(disc, 0.5)
        intersect_point = eye + (v - d) * direction
        sphere_normal = intersect_point - sphere[0, :]
        norm_sphere_normal = 0.0
        for k in range(len(sphere_normal)):
            norm_sphere_normal += sphere_normal[k] ** 2
        norm_sphere_normal = pow(norm_sphere_normal, 0.5)
        sphere_normal /= norm_sphere_normal
        return_value[0, :] = sphere_normal
        return_value[1, :] = intersect_point
        return_value[2, 0] = v - d
        return return_value
    
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef numpy.ndarray[numpy.float64_t, ndim=2] get_sphere_normal_array2float64_array1float64_array1float64_typespec_(numpy.ndarray[numpy.float64_t, ndim=2] sphere,numpy.ndarray[numpy.float64_t, ndim=1] direction,numpy.ndarray[numpy.float64_t, ndim=1] eye):
    #_chosen_typespecialize {'_nonvar_np.zeros([3,3])': (['int'],), 'norm_eye_to_sphere': 'double', 'k': 'int', '_nonvar_eye_to_sphere[k]**2': ('double',), '_nonvar_range(len(direction))': ('int',), 'direction': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', '_nonvar_len(eye_to_sphere)': ('numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])',), '_nonvar_sphere[1,0]**2': ('double',), 'sphere': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(3, 3),shape_list=[])', '_nonvar_range(len(eye_to_sphere))': ('int',), 'disc': 'double', '_nonvar_v**2': ('double',), 'return_value': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(3, 3),shape_list=[])', '_nonvar_len(direction)': ('numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])',), 'eye': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'v': 'double', 'eye_to_sphere': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])'}
    cdef double norm_eye_to_sphere
    cdef int k
    cdef double disc
    cdef numpy.ndarray[numpy.float64_t, ndim=2] return_value
    cdef double v
    cdef double[3] eye_to_sphere
    cdef double norm_sphere_normal
    cdef double d
    cdef double[3] intersect_point
    cdef double[3] sphere_normal

    #_after_cdefs_
    
    
    """Calculate the sphere's normal at the intersection of the sphere and the view's direction
        return_value = 
        [ sphere_normal
          intersect_point
          disc
        ]
    """
    
    
    global _prealloc_return_value_global, _prealloc_return_value_init
    if not _prealloc_return_value_init:
        _prealloc_return_value_init = True
        _prealloc_return_value_global = np.zeros([3, 3])
    return_value = _prealloc_return_value_global
            
    for k in range(3):
        eye_to_sphere[k] = sphere[0, k] - eye[k]
    v = 0.0
    for k in range(3):
        v += direction[k] * eye_to_sphere[k]
        
    if v <= 0:
        return_value[2, 0] = 1e20
        return return_value
    
    norm_eye_to_sphere = 0.0
    for k in range(3):
        norm_eye_to_sphere += square_double(eye_to_sphere[k])
        
    disc = square_double(sphere[1, 0]) - (norm_eye_to_sphere - square_double(v))
    
    # if sphere doesn't intersect with eye's view
    if disc < 0:
        
        return_value[2, 0] = 1e20
        return return_value
        
    else:
        
        d = libc.math.sqrt(disc)
        for k in range(3):
            intersect_point[k] = eye[k] + (v - d) * direction[k]
        for k in range(3):
            sphere_normal[k] = intersect_point[k] - sphere[0, k]
        norm_sphere_normal = 0.0
        for k in range(3):
            norm_sphere_normal += sphere_normal[k] ** 2
        norm_sphere_normal = libc.math.sqrt(norm_sphere_normal)
        for k in range(3):
            sphere_normal[k] /= norm_sphere_normal
        for k in range(3):
            return_value[0, k] = sphere_normal[k]
        for k in range(3):
            return_value[1, k] = intersect_point[k]
        return_value[2, 0] = v - d
        return return_value
    
    










def calculate_intensity(light_dir, intersect_point, normal, view_direction, isglass):
    #_chosen_typespecialize None
    if (isinstance(light_dir, numpy.ndarray) and light_dir.dtype == numpy.float64 and light_dir.ndim == 1) and (isinstance(intersect_point, numpy.ndarray) and intersect_point.dtype == numpy.float64 and intersect_point.ndim == 1) and (isinstance(normal, numpy.ndarray) and normal.dtype == numpy.float64 and normal.ndim == 1) and (isinstance(view_direction, numpy.ndarray) and view_direction.dtype == numpy.float64 and view_direction.ndim == 1) and isinstance(isglass, (int, numpy.int64)):
        return calculate_intensity_array1float64_array1float64_array1float64_array1float64_int_typespec_(light_dir, intersect_point, normal, view_direction, isglass)
    
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
            norm_R = pow(norm_R, 0.5)
            R /= norm_R
            reflection = 0.0
            for k in range(len(R)):
                reflection += R[k] * view_direction[k]
            reflection = pow(reflection, alfa)
            
    return np.array([shade, reflection])
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef numpy.ndarray[numpy.float64_t, ndim=1] calculate_intensity_array1float64_array1float64_array1float64_array1float64_int_typespec_(numpy.ndarray[numpy.float64_t, ndim=1] light_dir,numpy.ndarray[numpy.float64_t, ndim=1] intersect_point,numpy.ndarray[numpy.float64_t, ndim=1] normal,numpy.ndarray[numpy.float64_t, ndim=1] view_direction,int isglass):
    #_chosen_typespecialize {'_nonvar_np.array([shade,reflection])': (['double'],), 'k': 'int', 'isglass': 'int', 'shade': 'double', 'light_dir': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', '_nonvar_len(light_dir)': ('numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])',), 'view_direction': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'reflection': 'double', '_nonvar_np.array([0.0,0.0,0.0])': (['double'],), 'alfa': 'int', 'normal': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', '_nonvar_range(len(light_dir))': ('int',), 'R': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'intersect_point': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])'}
    cdef int k
    cdef double shade
    cdef double reflection
    cdef int alfa
    cdef double[3] R
    cdef double light_dir_dot_normal
    #_after_cdefs_
    
    """Calculates the intensity (diffusion and reflection) resulting from a light source
    """
    
    alfa = 50
    
    shade = 0.0
    for k in range(3):
        shade += light_dir[k] * normal[k]
    reflection = 0.0
    #R = np.array([0.0, 0.0, 0.0])
    R[0] = 0.0
    R[1] = 0.0
    R[2] = 0.0
    
    if shade < 0.0:
    
        shade = 0.0
        
    else:
    
        if isglass:
    
            light_dir_dot_normal = 0.0
            for k in range(3):
                light_dir_dot_normal += light_dir[k] * normal[k]
            for k in range(3):
                R[k] = 2.0 * light_dir_dot_normal * normal[k] - light_dir[k]
            norm_R = 0.0
            for k in range(3):
                norm_R += square_double(R[k])
            norm_R = libc.math.sqrt(norm_R)
            for k in range(3):
                R[k] /= norm_R
            reflection = 0.0
            for k in range(3):
                reflection += R[k] * view_direction[k]
            reflection = libc.math.pow(reflection, alfa)
            
    return np.array([shade, reflection])
    







def check_blockage(intersect_point, light, sphere_group, intersect_index):
    #_chosen_typespecialize None
    if (isinstance(intersect_point, numpy.ndarray) and intersect_point.dtype == numpy.float64 and intersect_point.ndim == 1) and (isinstance(light, numpy.ndarray) and light.dtype == numpy.float64 and light.ndim == 1) and (isinstance(sphere_group, numpy.ndarray) and sphere_group.dtype == numpy.float64 and sphere_group.ndim == 2) and isinstance(intersect_index, (int, numpy.int64)):
        return check_blockage_array1float64_array1float64_array2float64_int_typespec_(intersect_point, light, sphere_group, intersect_index)
    
    """Check whether light is blocked by other objects at a certain point_color
    """
    
    return_value = True
    
    for k in range(9):
    
        if k != intersect_index:
    
            origin = sphere_group[3 * k, :]
            radius = sphere_group[1 + 3 * k, 0]
            
            
            global _prealloc_cross_product_global, _prealloc_cross_product_init
            if not _prealloc_cross_product_init or any([(3,)[_prealloc_j] > _prealloc_cross_product_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_cross_product_global).shape))]):
                _prealloc_cross_product_init = True
                _prealloc_cross_product_global = np.zeros(3)
            cross_product = _prealloc_cross_product_global
                    
            cross_product[0] = (light[1] - intersect_point[1]) * (intersect_point[2] - origin[2]) - (light[2] - intersect_point[2]) * (intersect_point[1] - origin[1])
            cross_product[1] = (light[2] - intersect_point[2]) * (intersect_point[0] - origin[0]) - (light[0] - intersect_point[0]) * (intersect_point[2] - origin[2])
            cross_product[2] = (light[0] - intersect_point[0]) * (intersect_point[1] - origin[1]) - (light[1] - intersect_point[1]) * (intersect_point[0] - origin[0])
            norm_cross_product = pow((cross_product[0] ** 2 + cross_product[1] ** 2 + cross_product[2] ** 2), 0.5)
            norm_light_intersect_point = 0.0
            for c in range(len(light)):
                norm_light_intersect_point += (light[c] - intersect_point[c]) ** 2
            norm_light_intersect_point = pow(norm_light_intersect_point, 0.5)
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
    
                    return False
    
    return return_value
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef bool check_blockage_array1float64_array1float64_array2float64_int_typespec_(numpy.ndarray[numpy.float64_t, ndim=1] intersect_point,numpy.ndarray[numpy.float64_t, ndim=1] light,numpy.ndarray[numpy.float64_t, ndim=2] sphere_group,int intersect_index):
    #_chosen_typespecialize {'_nonvar_pow(norm_light_intersect_point,0.5)': ('double', 'double'), 'norm_cross_product': 'double', '_nonvar_(light[c]-intersect_point[c])**2': ('double',), 'origin': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'cross_product': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'radius': 'double', 'sphere_group': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(27, 3),shape_list=[])', 'return_value': 'int', 'intersect_point': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'light': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'k': 'int', '_nonvar_cross_product[0]**2': ('double',), '_nonvar_range(9)': ('int',), 'c': 'int', '_nonvar_len(light)': ('numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])',), '_nonvar_cross_product[1]**2': ('double',), 'norm_light_intersect_point': 'double', '_nonvar_cross_product[2]**2': ('double',), '_nonvar_np.zeros(3)': ('int',), 'disc': 'double', '_nonvar_pow((cross_product[0]**2+cross_product[1]**2+cross_product[2]**2),0.5)': ('double', 'double'), 'intersect_index': 'int', '_nonvar_abs(norm_cross_product/norm_light_intersect_point)': ('double',), '_nonvar_range(len(light))': ('int',)}
    cdef double norm_cross_product
    cdef double[3] origin
    cdef double[3] cross_product
    cdef double radius
    cdef int return_value
    cdef int k, i
    cdef int c
    cdef double norm_light_intersect_point
    cdef double disc, dot_product1, dot_product2, t, norm_return

    #_after_cdefs_
    
    """Check whether light is blocked by other objects at a certain point_color
    """
    
    return_value = True
    
    for k in range(9):
    
        if k != intersect_index:

            for i in range(3):
                origin[i] = sphere_group[3 * k, i]
            radius = sphere_group[1 + 3 * k, 0]
            
            
            #global _prealloc_cross_product_global, _prealloc_cross_product_init
            #if not _prealloc_cross_product_init or any([(3,)[_prealloc_j] > _prealloc_cross_product_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_cross_product_global).shape))]):
            #    _prealloc_cross_product_init = True
            #    _prealloc_cross_product_global = np.zeros(3)
            #cross_product = _prealloc_cross_product_global
                    
            cross_product[0] = (light[1] - intersect_point[1]) * (intersect_point[2] - origin[2]) - (light[2] - intersect_point[2]) * (intersect_point[1] - origin[1])
            cross_product[1] = (light[2] - intersect_point[2]) * (intersect_point[0] - origin[0]) - (light[0] - intersect_point[0]) * (intersect_point[2] - origin[2])
            cross_product[2] = (light[0] - intersect_point[0]) * (intersect_point[1] - origin[1]) - (light[1] - intersect_point[1]) * (intersect_point[0] - origin[0])
            norm_cross_product = libc.math.pow((square_double(cross_product[0]) + square_double(cross_product[1]) + square_double(cross_product[2])), 0.5)
            norm_light_intersect_point = 0.0
            for c in range(3):
                norm_light_intersect_point += square_double((light[c] - intersect_point[c]))
            norm_light_intersect_point = libc.math.pow(norm_light_intersect_point, 0.5)
            disc = libc.math.fabs(norm_cross_product / norm_light_intersect_point)
        
            if disc <= radius:
        
                dot_product1 = 0.0
                dot_product2 = 0.0
                for c in range(3):
                    dot_product1 += (intersect_point[c] - origin[c]) * (light[c] - intersect_point[c])
                    dot_product2 += (light[c] - intersect_point[c]) * (light[c] - intersect_point[c])
                t = -dot_product1 / dot_product2
            
                if t < 0:
            
                    norm_return = 0.0
                    for c in range(3):
                        norm_return += (intersect_point[c] - origin[c]) ** 2
                    norm_return = norm_return ** 2
                    return_value = return_value and norm_return >= radius
            
                else:
    
                    return False
    
    return return_value
    





def get_color(parameter, bump_u, bump_v, input_img2, direction, depth, cutoff):
    #_chosen_typespecialize None
    if (isinstance(parameter, numpy.ndarray) and parameter.dtype == numpy.float64 and parameter.ndim == 2) and (isinstance(bump_u, numpy.ndarray) and bump_u.dtype == numpy.float64 and bump_u.ndim == 2) and (isinstance(bump_v, numpy.ndarray) and bump_v.dtype == numpy.float64 and bump_v.ndim == 2) and (isinstance(input_img2, numpy.ndarray) and input_img2.dtype == numpy.float64 and input_img2.ndim == 3) and (isinstance(direction, numpy.ndarray) and direction.dtype == numpy.float64 and direction.ndim == 1) and isinstance(depth, (int, numpy.int64)) and isinstance(cutoff, (float, numpy.float64)):
        return get_color_array2float64_array2float64_array2float64_array3float64_array1float64_int_double_typespec_(parameter, bump_u, bump_v, input_img2, direction, depth, cutoff)
    
    """Calculate the color of a certain point between the objects defined by scene and parameter,
    and from the view designated by direction. depth counts the number of iteration needed,
    and cutoff gives a threshold for terminating the iteration.
    """
    
    intensity = np.array([0.0, 0.0, 0.0])

    shade = 0.0
    reflection = 0.0
    disc_min = 1e20
    intersect_index = -1
    
    for k in range(9):
                
        return_value = get_sphere_normal(parameter[5 + 3 * k : 8 + 3 * k, :], direction, parameter[1, :])
        sphere_normal = return_value[0, :]
        intersect_point = return_value[1, :]
        disc = return_value[2, 0]
        
        if disc < disc_min:
                
            disc_min = disc
            intersect_index = k
    
    if disc_min == 1e20:
        return intensity
        
    return_value = get_sphere_normal(parameter[5 + 3 * intersect_index : 8 + 3 * intersect_index, :], direction, parameter[1, :])
    sphere_normal = return_value[0, :]
    intersect_point = return_value[1, :]
    disc = return_value[2, 0]
        
    light_dir = parameter[2, :] - intersect_point
    norm_light_dir = 0.0
    for c in range(len(light_dir)):
        norm_light_dir += light_dir[c] ** 2
    norm_light_dir = pow(norm_light_dir, 0.5)
    light_dir /= norm_light_dir
    
    norm_sphere_normal = 0.0
    for c in range(len(sphere_normal)):
        norm_sphere_normal += sphere_normal[c] ** 2
    norm_sphere_normal = pow(norm_sphere_normal, 0.5)
    if norm_sphere_normal == 0:
        
        intensity = intensity
        
    else:
    
        if intersect_index == 0:
        
            #using spherical coordinate
            theta = np.arccos(sphere_normal[2])
            phi = np.arctan2(sphere_normal[1], sphere_normal[0])        
            
            #map angles to the bump map
            u = int(bump_u.shape[1] * (phi + np.pi) / (2 * np.pi))
            v = int(bump_v.shape[0] * theta / np.pi)
            point_color = input_img2[v - 1, u - 1, :]
            
            if check_blockage(intersect_point, parameter[2, :], parameter[5 : 32], intersect_index):
                new_normal_x = sphere_normal[0] + bump_u[v - 1, u - 1] * np.cos(phi)
                new_normal_y = sphere_normal[1] + bump_u[v - 1, u - 1] * np.sin(phi)
                new_normal_z = sphere_normal[2] - bump_v[v - 1, u - 1]
                new_normal = [new_normal_x, new_normal_y, new_normal_z]
                norm_new_normal = 0.0
                for c in range(len(new_normal)):
                    norm_new_normal += new_normal[c] ** 2
                norm_new_normal = pow(norm_new_normal, 0.5)
                new_normal /= norm_new_normal
                normal = new_normal
                return_value_i = calculate_intensity(light_dir, intersect_point, normal, direction, isglass = False)
                shade = return_value_i[0]
                reflection = return_value_i[1]
                
            intensity = intensity + point_color * (parameter[3, 0] + parameter[3, 1] * shade + parameter[3, 2] * reflection)
                
        else:
            
            point_color = parameter[7 + 3 * intersect_index, :]
            normal = sphere_normal
            
            if check_blockage(intersect_point, parameter[2, :], parameter[5 : 32], intersect_index):
                return_value_i = calculate_intensity(light_dir, intersect_point, normal, direction, isglass = True)
                shade = return_value_i[0]
                reflection = return_value_i[1]
                
            intensity = intensity + point_color * (parameter[3, 0] + parameter[3, 1] * shade + parameter[3, 2] * reflection)
            
            if depth > 0 and parameter[4, 1] > cutoff:
        
                dot_product = 0.0
                for c in range(len(direction)):
                    dot_product += -direction[c] * normal[c]
                dot_product2 = 0.0
                reflection_direction = 2.0 * dot_product * normal + direction
                
                norm_reflection_direction = 0.0
                for c in range(len(reflection_direction)):
                    norm_reflection_direction += reflection_direction ** 2
                norm_reflection_direction = pow(norm_reflection_direction, 0.5)
                
                new_parameter = np.zeros((<object>parameter).shape)
                new_parameter[:, :] = parameter[:, :]
                new_parameter[1, :] = intersect_point
                add_intensity = get_color(new_parameter, bump_u, bump_v, input_img2, reflection_direction, depth - 1, cutoff / parameter[4, 1])
                intensity[0] += point_color[0] * parameter[4, 1] * add_intensity[0]
                intensity[1] += point_color[1] * parameter[4, 1] * add_intensity[1]
                intensity[2] += point_color[2] * parameter[4, 1] * add_intensity[2]
    
    return intensity
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef get_color_array2float64_array2float64_array2float64_array3float64_array1float64_int_double_typespec_(numpy.ndarray[numpy.float64_t, ndim=2] parameter,numpy.ndarray[numpy.float64_t, ndim=2] bump_u,numpy.ndarray[numpy.float64_t, ndim=2] bump_v,numpy.ndarray[numpy.float64_t, ndim=3] input_img2,numpy.ndarray[numpy.float64_t, ndim=1] direction,int depth,double cutoff): 
    #_chosen_typespecialize {'bump_u': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(200, 400),shape_list=[])', 'bump_v': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(200, 400),shape_list=[])', 'input_img2': 'numpy.ndarray[numpy.float64_t, ndim=3](shape=(200, 400, 3),shape_list=[])', 'sphere_normal': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'shade': 'double', 'disc_min': 'double', '_nonvar_range(9)': ('int',), 'depth': 'int', 'parameter': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(32, 3),shape_list=[])', 'cutoff': 'double', 'k': 'int', 'reflection': 'double', '_nonvar_get_sphere_normal(parameter[5+3*k:8+3*k,:],direction,parameter[1,:])': ('numpy.ndarray[numpy.float64_t, ndim=2](shape=(3, 3),shape_list=[])', 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])'), '_nonvar_np.array([0.0,0.0,0.0])': (['double'],), 'direction': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'disc': 'double', 'return_value': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(3, 3),shape_list=[])', 'intensity': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'intersect_index': 'int', 'intersect_point': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])'}
    cdef numpy.ndarray[numpy.float64_t, ndim=1] sphere_normal
    cdef double shade
    cdef double disc_min
    cdef int k
    cdef double reflection
    cdef double disc
    cdef numpy.ndarray[numpy.float64_t, ndim=2] return_value
    cdef numpy.ndarray[numpy.float64_t, ndim=1] intensity
    cdef int intersect_index
    cdef numpy.ndarray[numpy.float64_t, ndim=1] intersect_point
    cdef numpy.ndarray[numpy.float64_t, ndim=1] light_dir
    cdef double norm_light_dir
    cdef double theta, phi
    cdef int u, v
    cdef double[3] point_color
    cdef double new_normal_x, new_normal_y, new_normal_z
    cdef double[3] new_normal
    cdef double norm_new_normal
    cdef numpy.ndarray[numpy.float64_t, ndim=1] normal
    cdef double dot_product
    cdef double norm_reflection_direction

    #_after_cdefs_
    
    """Calculate the color of a certain point between the objects defined by scene and parameter,
    and from the view designated by direction. depth counts the number of iteration needed,
    and cutoff gives a threshold for terminating the iteration.
    """
    
#    intensity = np.array([0.0, 0.0, 0.0])
    global _prealloc_intensity_global, _prealloc_intensity_init
    if not _prealloc_intensity_init:
        _prealloc_intensity_init = True
        _prealloc_intensity_global = np.zeros(3)
    intensity = np.zeros(3) # _prealloc_intensity_global
#    point_color = np.zeros(3)
    light_dir = np.zeros(3)

    shade = 0.0
    reflection = 0.0
    disc_min = 1e20
    intersect_index = -1
    
    for k in range(9):
                
        return_value = get_sphere_normal(parameter[5 + 3 * k : 8 + 3 * k, :], direction, parameter[1, :])
        sphere_normal = return_value[0, :]
        intersect_point = return_value[1, :]
        disc = return_value[2, 0]
        
        if disc < disc_min:
                
            disc_min = disc
            intersect_index = k
    
    if disc_min == 1e20:
        return intensity
        
    return_value = get_sphere_normal(parameter[5 + 3 * intersect_index : 8 + 3 * intersect_index, :], direction, parameter[1, :])
    sphere_normal = return_value[0, :]
    intersect_point = return_value[1, :]
    disc = return_value[2, 0]

    for c in range(3):
        light_dir[c] = parameter[2, c] - intersect_point[c]
    norm_light_dir = 0.0
    for c in range(3):
        norm_light_dir += square_double(light_dir[c])
    norm_light_dir = libc.math.sqrt(norm_light_dir)
    for c in range(3):
        light_dir[c] /= norm_light_dir
    
    norm_sphere_normal = 0.0
    for c in range(3):
        norm_sphere_normal += square_double(sphere_normal[c])
    norm_sphere_normal = libc.math.sqrt(norm_sphere_normal)
    if norm_sphere_normal == 0:
        
        pass #intensity = intensity
        
    else:
    
        if intersect_index == 0:
        
            #using spherical coordinate
            theta = libc.math.acos(sphere_normal[2])
            phi = libc.math.atan2(sphere_normal[1], sphere_normal[0])
            
            #map angles to the bump map
            u = int(bump_u.shape[1] * (phi + 3.141592653589793) / (2 * 3.141592653589793))
            v = int(bump_v.shape[0] * theta / 3.141592653589793)
            for c in range(3):
                point_color[c] = input_img2[v - 1, u - 1, c]
            
            if check_blockage_array1float64_array1float64_array2float64_int_typespec_(intersect_point, parameter[2, :], parameter[5 : 32], intersect_index):
                new_normal_x = sphere_normal[0] + bump_u[v - 1, u - 1] * np.cos(phi)
                new_normal_y = sphere_normal[1] + bump_u[v - 1, u - 1] * np.sin(phi)
                new_normal_z = sphere_normal[2] - bump_v[v - 1, u - 1]
                new_normal[0] = new_normal_x
                new_normal[1] = new_normal_y
                new_normal[2] = new_normal_z
                norm_new_normal = 0.0
                for c in range(3):
                    norm_new_normal += square_double(new_normal[c])
                norm_new_normal = libc.math.sqrt(norm_new_normal)
                for c in range(3):
                    new_normal[c] /= norm_new_normal
                normal = np.zeros(3)
                for c in range(3):
                    normal[c] = new_normal[c]
                return_value_i = calculate_intensity_array1float64_array1float64_array1float64_array1float64_int_typespec_(light_dir, intersect_point, normal, direction, False)
                shade = return_value_i[0]
                reflection = return_value_i[1]

            for c in range(3):
                intensity[c] = point_color[c] * (parameter[3, 0] + parameter[3, 1] * shade + parameter[3, 2] * reflection)
                
        else:

            for c in range(3):
                point_color[c] = parameter[7 + 3 * intersect_index, c]
            normal = sphere_normal
            
            if check_blockage_array1float64_array1float64_array2float64_int_typespec_(intersect_point, parameter[2, :], parameter[5 : 32], intersect_index):
                return_value_i = calculate_intensity(light_dir, intersect_point, normal, direction, isglass = True)
                shade = return_value_i[0]
                reflection = return_value_i[1]

            for c in range(3):
                intensity[c] = point_color[c] * (parameter[3, 0] + parameter[3, 1] * shade + parameter[3, 2] * reflection)
            
            if depth > 0 and parameter[4, 1] > cutoff:
        
                dot_product = 0.0
                for c in range(len(direction)):
                    dot_product += -direction[c] * normal[c]
                reflection_direction = 2.0 * dot_product * normal + direction
                
                norm_reflection_direction = 0.0
                for c in range(3):
                    norm_reflection_direction += square_double(reflection_direction[c])
                norm_reflection_direction = libc.math.sqrt(norm_reflection_direction)
                
                new_parameter = np.zeros((<object>parameter).shape)
                new_parameter[:, :] = parameter[:, :]
                new_parameter[1, :] = intersect_point
                add_intensity = get_color_array2float64_array2float64_array2float64_array3float64_array1float64_int_double_typespec_(new_parameter, bump_u, bump_v, input_img2, reflection_direction, depth - 1, cutoff / parameter[4, 1])
                intensity[0] += point_color[0] * parameter[4, 1] * add_intensity[0]
                intensity[1] += point_color[1] * parameter[4, 1] * add_intensity[1]
                intensity[2] += point_color[2] * parameter[4, 1] * add_intensity[2]
    
    return intensity
    



def raytracer(input_img1, input_img2, width, height, time, output_img):
    #_chosen_typespecialize None
    if (isinstance(input_img1, numpy.ndarray) and input_img1.dtype == numpy.float64 and input_img1.ndim == 3) and (isinstance(input_img2, numpy.ndarray) and input_img2.dtype == numpy.float64 and input_img2.ndim == 3) and isinstance(width, (int, numpy.int64)) and isinstance(height, (int, numpy.int64)) and isinstance(time, (float, numpy.float64)) and (isinstance(output_img, numpy.ndarray) and output_img.dtype == numpy.float64 and output_img.ndim == 3):
        return raytracer_array3float64_array3float64_int_int_double_array3float64_typespec_(input_img1, input_img2, width, height, time, output_img)
    
    """Draws the desired scene using ray tracing method.
    
    parameter format:
    [view         (1 * 2)        , 0
     eys          (1 * 3)
     light        (1 * 3)
     ambient             , diffuse, specular
     alfa                , kspec , 0
     tennis_origin(1 * 3)
     tennis_radius       , 0     , 0
     tennic_color (1 * 3)
     glass1_origin(1 * 3)
     glass1_radius       , 0     , 0
     glass1_color (1 * 3)
     glass2_origin(1 * 3)
     glass2_radius       , 0     , 0
     glass2_color (1 * 3)
     glass3_origin(1 * 3)
     glass3_radius       , 0     , 0
     glass3_color (1 * 3)
     glass4_origin(1 * 3)
     glass4_radius       , 0     , 0
     glass4_color (1 * 3)
     glass5_origin(1 * 3)
     glass5_radius       , 0     , 0
     glass5_color (1 * 3)
     glass6_origin(1 * 3)
     glass6_radius       , 0     , 0
     glass6_color (1 * 3)
     glass7_origin(1 * 3)
     glass7_radius       , 0     , 0
     glass7_color (1 * 3)
     glass8_origin(1 * 3)
     glass8_radius       , 0     , 0
     glass8_color (1 * 3)
    ]
    """
    
    
    global _prealloc_parameter_global, _prealloc_parameter_init
    if not _prealloc_parameter_init or any([[32, 3][_prealloc_j] > _prealloc_parameter_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_parameter_global).shape))]):
        _prealloc_parameter_init = True
        _prealloc_parameter_global = np.zeros([32, 3])
    parameter = _prealloc_parameter_global
            
    assert output_img.shape[0] == height, (output_img.shape[0], height)
    assert output_img.shape[1] == width, (output_img.shape[1], width)
    
    #view
    parameter[0, 0] = float(height)
    parameter[0, 1] = float(width)
    
    #eyes
    parameter[1, :] = np.array([320.0, 480.0, -1600.0])
    
    #light
    parameter[2, :] = np.array([320.0 + 200.0 * np.cos(-np.pi * time / 2.0), 480.0 + 200.0 * np.sin(-np.pi * time / 2.0), 0.0])
    
    #ambient, difuse, specular
    parameter[3, 0] = 0.2
    parameter[3, 1] = 1.0 - parameter[3, 0]
    parameter[3, 2] = 0.7
    
    #alfa, kspec
    parameter[4, 0] = 50
    parameter[4, 1] = 0.9
    
    #tennis origin, radius, color
    parameter[5, :] = np.array([320.0, 480.0, 800.0])
    parameter[6, 0] = 200.0
    parameter[7, :] = np.array([0.0, 1.0, 1.0])
    
    #glass1 origin, radius, color
    parameter[8, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0), 900.0])
    parameter[9, 0] = 100.0
    parameter[10, :] = np.array([1.0, 1.0, 1.0])
    
    #glass2 origin, radius, color
    parameter[11, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + np.pi / 4.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + np.pi / 4.0), 900.0])
    parameter[12, 0] = 100.0
    parameter[13, :] = np.array([1.0, 1.0, 1.0])
    
    #glass3 origin, radius, color
    parameter[14, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + np.pi / 2.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + np.pi / 2.0), 900.0])
    parameter[15, 0] = 100.0
    parameter[16, :] = np.array([1.0, 1.0, 1.0])
    
    #glass4 origin, radius, color
    parameter[17, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + 3.0 * np.pi / 4.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + 3.0 * np.pi / 4.0), 900.0])
    parameter[18, 0] = 100.0
    parameter[19, :] = np.array([1.0, 1.0, 1.0])
    
    #glass5 origin, radius, color
    parameter[20, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + np.pi), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + np.pi), 900.0])
    parameter[21, 0] = 100.0
    parameter[22, :] = np.array([1.0, 1.0, 1.0])
    
    #glass6 origin, radius, color
    parameter[23, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + 5.0 * np.pi / 4.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + 5.0 * np.pi / 4.0), 900.0])
    parameter[24, 0] = 100.0
    parameter[25, :] = np.array([1.0, 1.0, 1.0])
    
    #glass7 origin, radius, color
    parameter[26, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + 3.0 * np.pi / 2.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + 3.0 * np.pi / 2.0), 900.0])
    parameter[27, 0] = 100.0
    parameter[28, :] = np.array([1.0, 1.0, 1.0])
    
    #glass7 origin, radius, color
    parameter[29, :] = np.array([320.0 + 320.0 * np.cos(np.pi * time / 3.0 + 7.0 * np.pi / 4.0), 480.0 + 320.0 * np.sin(np.pi * time / 3.0 + 7.0 * np.pi / 4.0), 900.0])
    parameter[30, 0] = 100.0
    parameter[31, :] = np.array([1.0, 1.0, 1.0])
    
    #    output_img = np.zeros([int(parameter[0, 0]), int(parameter[0, 1]), 3])
    
    # add bump to sphere
    bump_map = 0.299 * input_img1[:, :, 0] + 0.587 * input_img1[:, :, 1] + 0.114 * input_img1[:,:,2]
    
    global _prealloc_bump_u_global, _prealloc_bump_u_init
    if not _prealloc_bump_u_init or any([bump_map.shape[_prealloc_j] > _prealloc_bump_u_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_bump_u_global).shape))]):
        _prealloc_bump_u_init = True
        _prealloc_bump_u_global = np.zeros((<object>bump_map).shape)
    bump_u = _prealloc_bump_u_global
            
    
    global _prealloc_bump_v_global, _prealloc_bump_v_init
    if not _prealloc_bump_v_init or any([bump_map.shape[_prealloc_j] > _prealloc_bump_v_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_bump_v_global).shape))]):
        _prealloc_bump_v_init = True
        _prealloc_bump_v_global = np.zeros((<object>bump_map).shape)
    bump_v = _prealloc_bump_v_global
            
    
    for r in range(1, bump_u.shape[0] - 1):
        for c in range(1, bump_v.shape[1] - 1):
            bump_u[r, c] = (-bump_map[r - 1, c - 1] - 2 * bump_map[r - 1, c] - bump_map[r - 1, c + 1] + bump_map[r + 1, c - 1] + 2 * bump_map[r + 1, c] + bump_map[r + 1, c + 1]) / 2.0
            bump_v[r, c] = (-bump_map[r - 1, c - 1] - 2 * bump_map[r, c - 1] - bump_map[r + 1, c - 1] + bump_map[r - 1, c + 1] + 2 * bump_map[r, c + 1] + bump_map[r + 1, c + 1]) / 2.0
    
    for r in range(output_img.shape[0]):
        for c in range(output_img.shape[1]):
        
            direction = np.array([r/output_img.shape[0]*640.0, c/output_img.shape[1]*960.0, 0.0]) - parameter[1, :]
            norm_direction = 0.0
            for k in range(len(direction)):
                norm_direction += direction[k] ** 2
            norm_direction = pow(norm_direction, 0.5)
            direction /= norm_direction
            
            intensity = get_color(parameter, bump_u, bump_v, input_img2, direction, depth = 3, cutoff = 0.1)
            
            output_img[r, c, 0] = intensity[0] / (1 + intensity[0])
            output_img[r, c, 1] = intensity[1] / (1 + intensity[1])
            output_img[r, c, 2] = intensity[2] / (1 + intensity[2])
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef raytracer_array3float64_array3float64_int_int_double_array3float64_typespec_(numpy.ndarray[numpy.float64_t, ndim=3] input_img1,numpy.ndarray[numpy.float64_t, ndim=3] input_img2,int width,int height,double time,output_img): 
    #_chosen_typespecialize {'input_img1': 'numpy.ndarray[numpy.float64_t, ndim=3](shape=(200, 400, 3),shape_list=[])', '_nonvar_pow(norm_direction,0.5)': ('double', 'double'), 'input_img2': 'numpy.ndarray[numpy.float64_t, ndim=3](shape=(200, 400, 3),shape_list=[])', '_nonvar_np.sin(-np.pi*time/2.0)': ('double',), 'height': 'int', '_nonvar_range(len(direction))': ('int',), '_nonvar_float(width)': ('int',), 'output_img': 'numpy.ndarray[numpy.float64_t, ndim=3](shape=(64, 96, 3),shape_list=[])', '_nonvar_np.cos(np.pi*time/3.0+3.0*np.pi/4.0)': ('double',), 'time': 'double', '_nonvar_np.sin(np.pi*time/3.0+3.0*np.pi/2.0)': ('double',), '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+3.0*np.pi/4.0),480.0+320.0*np.sin(np.pi*time/3.0+3.0*np.pi/4.0),900.0])': (['double'],), '_nonvar_np.sin(np.pi*time/3.0+3.0*np.pi/4.0)': ('double',), '_nonvar_np.sin(np.pi*time/3.0+5.0*np.pi/4.0)': ('double',), 'k': 'int', '_nonvar_range(output_img.shape[1])': ('int',), '_nonvar_np.array([r/output_img.shape[0]*640.0,c/output_img.shape[1]*960.0,0.0])': (['double'],), 'width': 'int', '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+np.pi/2.0),480.0+320.0*np.sin(np.pi*time/3.0+np.pi/2.0),900.0])': (['double'],), 'intensity': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', 'c': 'int', '_nonvar_np.array([320.0,480.0,-1600.0])': (['double'],), '_nonvar_range(1,bump_u.shape[0]-1)': ('int', 'int'), '_nonvar_np.cos(np.pi*time/3.0+np.pi/4.0)': ('double',), '_nonvar_np.cos(np.pi*time/3.0+np.pi)': ('double',), 'r': 'int', '_nonvar_np.sin(np.pi*time/3.0+np.pi/2.0)': ('double',), '_nonvar_direction[k]**2': ('double',), '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+7.0*np.pi/4.0),480.0+320.0*np.sin(np.pi*time/3.0+7.0*np.pi/4.0),900.0])': (['double'],), '_nonvar_np.array([0.0,1.0,1.0])': (['double'],), '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+5.0*np.pi/4.0),480.0+320.0*np.sin(np.pi*time/3.0+5.0*np.pi/4.0),900.0])': (['double'],), '_nonvar_np.cos(np.pi*time/3.0)': ('double',), 'norm_direction': 'double', '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+np.pi),480.0+320.0*np.sin(np.pi*time/3.0+np.pi),900.0])': (['double'],), 'parameter': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(32, 3),shape_list=[])', '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+np.pi/4.0),480.0+320.0*np.sin(np.pi*time/3.0+np.pi/4.0),900.0])': (['double'],), '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0+3.0*np.pi/2.0),480.0+320.0*np.sin(np.pi*time/3.0+3.0*np.pi/2.0),900.0])': (['double'],), '_nonvar_np.sin(np.pi*time/3.0+7.0*np.pi/4.0)': ('double',), '_nonvar_np.cos(np.pi*time/3.0+5.0*np.pi/4.0)': ('double',), '_nonvar_len(direction)': ('numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])',), '_nonvar_np.array([320.0+320.0*np.cos(np.pi*time/3.0),480.0+320.0*np.sin(np.pi*time/3.0),900.0])': (['double'],), 'bump_u': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(200, 400),shape_list=[])', 'bump_v': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(200, 400),shape_list=[])', '_nonvar_np.cos(np.pi*time/3.0+7.0*np.pi/4.0)': ('double',), '_nonvar_range(output_img.shape[0])': ('int',), '_nonvar_np.array([320.0+200.0*np.cos(-np.pi*time/2.0),480.0+200.0*np.sin(-np.pi*time/2.0),0.0])': (['double'],), '_nonvar_np.array([1.0,1.0,1.0])': (['double'],), 'direction': 'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])', '_nonvar_np.sin(np.pi*time/3.0+np.pi)': ('double',), '_nonvar_np.array([320.0,480.0,800.0])': (['double'],), '_nonvar_np.cos(np.pi*time/3.0+np.pi/2.0)': ('double',), '_nonvar_float(height)': ('int',), '_nonvar_np.zeros(bump_map.shape)': (('int', 'int'),), '_nonvar_np.cos(-np.pi*time/2.0)': ('double',), '_nonvar_np.cos(np.pi*time/3.0+3.0*np.pi/2.0)': ('double',), '_nonvar_np.sin(np.pi*time/3.0)': ('double',), '_nonvar_np.zeros([32,3])': (['int'],), 'bump_map': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(200, 400),shape_list=[])', '_nonvar_np.sin(np.pi*time/3.0+np.pi/4.0)': ('double',), '_nonvar_range(1,bump_v.shape[1]-1)': ('int', 'int')}
    cdef int k
    cdef numpy.ndarray[numpy.float64_t, ndim=1] intensity
    cdef int c
    cdef int r
    cdef double norm_direction
    cdef numpy.ndarray[numpy.float64_t, ndim=2] parameter
    cdef numpy.ndarray[numpy.float64_t, ndim=2] bump_u
    cdef numpy.ndarray[numpy.float64_t, ndim=2] bump_v
    cdef numpy.ndarray[numpy.float64_t, ndim=1] direction
    cdef numpy.ndarray[numpy.float64_t, ndim=2] bump_map
    #_after_cdefs_
    
    """Draws the desired scene using ray tracing method.
    
    parameter format:
    [view         (1 * 2)        , 0
     eys          (1 * 3)
     light        (1 * 3)
     ambient             , diffuse, specular
     alfa                , kspec , 0
     tennis_origin(1 * 3)
     tennis_radius       , 0     , 0
     tennic_color (1 * 3)
     glass1_origin(1 * 3)
     glass1_radius       , 0     , 0
     glass1_color (1 * 3)
     glass2_origin(1 * 3)
     glass2_radius       , 0     , 0
     glass2_color (1 * 3)
     glass3_origin(1 * 3)
     glass3_radius       , 0     , 0
     glass3_color (1 * 3)
     glass4_origin(1 * 3)
     glass4_radius       , 0     , 0
     glass4_color (1 * 3)
     glass5_origin(1 * 3)
     glass5_radius       , 0     , 0
     glass5_color (1 * 3)
     glass6_origin(1 * 3)
     glass6_radius       , 0     , 0
     glass6_color (1 * 3)
     glass7_origin(1 * 3)
     glass7_radius       , 0     , 0
     glass7_color (1 * 3)
     glass8_origin(1 * 3)
     glass8_radius       , 0     , 0
     glass8_color (1 * 3)
    ]
    """
    
    
    global _prealloc_parameter_global, _prealloc_parameter_init
    if not _prealloc_parameter_init or any([[32, 3][_prealloc_j] > _prealloc_parameter_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_parameter_global).shape))]):
        _prealloc_parameter_init = True
        _prealloc_parameter_global = np.zeros([32, 3])
    parameter = _prealloc_parameter_global
            
    assert output_img.shape[0] == height, (output_img.shape[0], height)
    assert output_img.shape[1] == width, (output_img.shape[1], width)
    
    #view
    parameter[0, 0] = int_to_float(height)
    parameter[0, 1] = int_to_float(width)
    
    #eyes
    parameter[1, :] = np.array([320.0, 480.0, -1600.0])
    
    #light
    parameter[2, :] = np.array([320.0 + 200.0 * libc.math.cos(-np.pi * time / 2.0), 480.0 + 200.0 * libc.math.sin(-np.pi * time / 2.0), 0.0])
    
    #ambient, difuse, specular
    parameter[3, 0] = 0.2
    parameter[3, 1] = 1.0 - parameter[3, 0]
    parameter[3, 2] = 0.7
    
    #alfa, kspec
    parameter[4, 0] = 50
    parameter[4, 1] = 0.9
    
    #tennis origin, radius, color
    parameter[5, :] = np.array([320.0, 480.0, 800.0])
    parameter[6, 0] = 200.0
    parameter[7, :] = np.array([0.0, 1.0, 1.0])
    
    #glass1 origin, radius, color
    parameter[8, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0), 900.0])
    parameter[9, 0] = 100.0
    parameter[10, :] = np.array([1.0, 1.0, 1.0])
    
    #glass2 origin, radius, color
    parameter[11, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + np.pi / 4.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + np.pi / 4.0), 900.0])
    parameter[12, 0] = 100.0
    parameter[13, :] = np.array([1.0, 1.0, 1.0])
    
    #glass3 origin, radius, color
    parameter[14, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + np.pi / 2.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + np.pi / 2.0), 900.0])
    parameter[15, 0] = 100.0
    parameter[16, :] = np.array([1.0, 1.0, 1.0])
    
    #glass4 origin, radius, color
    parameter[17, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + 3.0 * np.pi / 4.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + 3.0 * np.pi / 4.0), 900.0])
    parameter[18, 0] = 100.0
    parameter[19, :] = np.array([1.0, 1.0, 1.0])
    
    #glass5 origin, radius, color
    parameter[20, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + np.pi), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + np.pi), 900.0])
    parameter[21, 0] = 100.0
    parameter[22, :] = np.array([1.0, 1.0, 1.0])
    
    #glass6 origin, radius, color
    parameter[23, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + 5.0 * np.pi / 4.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + 5.0 * np.pi / 4.0), 900.0])
    parameter[24, 0] = 100.0
    parameter[25, :] = np.array([1.0, 1.0, 1.0])
    
    #glass7 origin, radius, color
    parameter[26, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + 3.0 * np.pi / 2.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + 3.0 * np.pi / 2.0), 900.0])
    parameter[27, 0] = 100.0
    parameter[28, :] = np.array([1.0, 1.0, 1.0])
    
    #glass7 origin, radius, color
    parameter[29, :] = np.array([320.0 + 320.0 * libc.math.cos(np.pi * time / 3.0 + 7.0 * np.pi / 4.0), 480.0 + 320.0 * libc.math.sin(np.pi * time / 3.0 + 7.0 * np.pi / 4.0), 900.0])
    parameter[30, 0] = 100.0
    parameter[31, :] = np.array([1.0, 1.0, 1.0])
    
    #    output_img = np.zeros([int(parameter[0, 0]), int(parameter[0, 1]), 3])
    
    # add bump to sphere
    bump_map = 0.299 * input_img1[:, :, 0] + 0.587 * input_img1[:, :, 1] + 0.114 * input_img1[:,:,2]
    
    global _prealloc_bump_u_global, _prealloc_bump_u_init
    if not _prealloc_bump_u_init or any([bump_map.shape[_prealloc_j] > _prealloc_bump_u_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_bump_u_global).shape))]):
        _prealloc_bump_u_init = True
        _prealloc_bump_u_global = np.zeros((<object>bump_map).shape)
    bump_u = _prealloc_bump_u_global
            
    
    global _prealloc_bump_v_global, _prealloc_bump_v_init
    if not _prealloc_bump_v_init or any([bump_map.shape[_prealloc_j] > _prealloc_bump_v_global.shape[_prealloc_j] for _prealloc_j in range(len((<object>_prealloc_bump_v_global).shape))]):
        _prealloc_bump_v_init = True
        _prealloc_bump_v_global = np.zeros((<object>bump_map).shape)
    bump_v = _prealloc_bump_v_global
            
    
    for r in range(1, bump_u.shape[0] - 1):
        for c in range(1, bump_v.shape[1] - 1):
            bump_u[r, c] = (-bump_map[r - 1, c - 1] - 2 * bump_map[r - 1, c] - bump_map[r - 1, c + 1] + bump_map[r + 1, c - 1] + 2 * bump_map[r + 1, c] + bump_map[r + 1, c + 1]) / 2.0
            bump_v[r, c] = (-bump_map[r - 1, c - 1] - 2 * bump_map[r, c - 1] - bump_map[r + 1, c - 1] + bump_map[r - 1, c + 1] + 2 * bump_map[r, c + 1] + bump_map[r + 1, c + 1]) / 2.0
    
    for r in range(output_img.shape[0]):
        for c in range(output_img.shape[1]):
        
            direction = np.array([r/output_img.shape[0]*640.0, c/output_img.shape[1]*960.0, 0.0]) - parameter[1, :]
            norm_direction = 0.0
            for k in range(3):
                norm_direction += square_double(direction[k])
            norm_direction = libc.math.sqrt(norm_direction)
            for k in range(3):
                direction[k] /= norm_direction
            
            intensity = get_color_array2float64_array2float64_array2float64_array3float64_array1float64_int_double_typespec_(parameter, bump_u, bump_v, input_img2, direction, 3, 0.1)
            
            output_img[r, c, 0] = intensity[0] / (1 + intensity[0])
            output_img[r, c, 1] = intensity[1] / (1 + intensity[1])
            output_img[r, c, 2] = intensity[2] / (1 + intensity[2])


input_img1 = util.image_filename('ball_bump_map.png')
input_img2 = util.image_filename('ball_color_map.png')
    
    
def test(time=0.0, n=1, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Default unit tests
    """
    util.is_initial_run = True
    ans = util.test_image_pipeline_filename(raytracer, [input_img1, input_img2], n, name = 'ray_tracer', additional_args=(width, height, time), use_output_img=True, output_img_shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3))
    return util.combine_tests([ans])
    










if __name__ == '__main__':
    test()










