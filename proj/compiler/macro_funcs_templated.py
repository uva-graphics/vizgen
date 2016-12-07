import macros

MAX_NUM_LEN_WRAP = 10

templated_func = {}

def add_templated_func(n):
    """
    add templated function of vector length n
    """
    assert(isinstance(n, int))
    
    if 'numpy_clip_vec' + str(n) + 'f' in templated_func:
        return
    
    numpy_clip_f_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[numpy.float32_t, ndim=1] numpy_clip_vec{n}f_ptr(float *x, float a, float b):
    cdef numpy.ndarray[numpy.float32_t, ndim=1] ans
    cdef int i
    cdef float x_i
    ans = numpy.empty({n}, 'float32')
    for i in range({n}):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans
""".format(**locals())

    templated_func['numpy_clip_vec' + str(n) + 'f_ptr'] = numpy_clip_f_ptr
    
    numpy_clip_g_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[numpy.float64_t, ndim=1] numpy_clip_vec{n}g_ptr(double *x, double a, double b):
    cdef numpy.ndarray[numpy.float64_t, ndim=1] ans
    cdef int i
    cdef double x_i
    ans = numpy.empty({n}, 'float64')
    for i in range({n}):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans
""".format(**locals())

    templated_func['numpy_clip_vec' + str(n) + 'g_ptr'] = numpy_clip_g_ptr
    
    numpy_clip_f = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[numpy.float32_t, ndim=1] numpy_clip_vec{n}f(float[:] x, float a, float b):
    cdef numpy.ndarray[numpy.float32_t, ndim=1] ans
    cdef int i
    cdef float x_i
    ans = numpy.empty({n}, 'float32')
    for i in range({n}):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans
""".format(**locals())

    templated_func['numpy_clip_vec' + str(n) + 'f'] = numpy_clip_f
    
    numpy_clip_g = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[numpy.float64_t, ndim=1] numpy_clip_vec{n}g(double[:] x, double a, double b):
    cdef numpy.ndarray[numpy.float64_t, ndim=1] ans
    cdef int i
    cdef double x_i
    ans = numpy.empty({n}, 'float64')
    for i in range({n}):
        x_i = x[i]
        ans[i] = a if x_i < a else (b if x_i > b else x_i)
    return ans
""".format(**locals())

    templated_func['numpy_clip_vec' + str(n) + 'g'] = numpy_clip_g
    
    if n < MAX_NUM_LEN_WRAP:
        numpy_linalg_norm_f_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float numpy_linalg_norm_vec{n}f_ptr(float *v) nogil:
    return libc.math.sqrt(""".format(**locals())
    
        for i in range(n):
            numpy_linalg_norm_f_ptr += """v[{i}]*v[{i}]+""".format(**locals())
        numpy_linalg_norm_f_ptr = numpy_linalg_norm_f_ptr[:-1]
        numpy_linalg_norm_f_ptr += """)
"""
        
        numpy_linalg_norm_g_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double numpy_linalg_norm_vec{n}g_ptr(double *v) nogil:
    return libc.math.sqrt(""".format(**locals())
    
        for i in range(n):
            numpy_linalg_norm_g_ptr += """v[{i}]*v[{i}]+""".format(**locals())
        numpy_linalg_norm_g_ptr = numpy_linalg_norm_g_ptr[:-1]
        numpy_linalg_norm_g_ptr += """)
"""
        
        numpy_dot_f_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float numpy_dot_vec{n}f_ptr(float *u, float *v) nogil:
    return """.format(**locals())
        
        for i in range(n):
            numpy_dot_f_ptr += """u[{i}]*v[{i}]+""".format(**locals())
        numpy_dot_f_ptr = numpy_dot_f_ptr[:-1]
        numpy_dot_f_ptr += """
"""
        
        numpy_dot_g_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double numpy_dot_vec{n}g_ptr(double *u, double *v) nogil:
    return """.format(**locals())
    
        for i in range(n):
            numpy_dot_g_ptr += """u[{i}]*v[{i}]+""".format(**locals())
        numpy_dot_g_ptr = numpy_dot_g_ptr[:-1]
        numpy_dot_g_ptr += """
"""
        
    else:
        numpy_linalg_norm_f_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float numpy_linalg_norm_vec{n}f_ptr(float *v) nogil:
    cdef float ans
    cdef int i
    ans = 0.0
    for i in range({n}):
        ans += v[i]*v[i]
    return ans
""".format(**locals())

        numpy_linalg_norm_g_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double numpy_linalg_norm_vec{n}g_ptr(double *v) nogil:
    cdef double ans
    cdef int i
    ans = 0.0
    for i in range({n}):
        ans += v[i]*v[i]
    return ans
""".format(**locals())

        numpy_dot_f_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float numpy_dot_vec{n}f_ptr(float *u, float *v) nogil:
    cdef float ans
    cdef int i
    ans = 0.0
    for i in range({n}):
        ans += u[i]*v[i]
    return ans
""".format(**locals())

        numpy_dot_g_ptr = """
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double numpy_dot_vec{n}g_ptr(double *u, double *v) nogil:
    cdef double ans
    cdef int i
    ans = 0.0
    for i in range({n}):
        ans += u[i]*v[i]
    return ans
""".format(**locals())

    templated_func['numpy_linalg_norm_vec' + str(n) + 'f_ptr'] = numpy_linalg_norm_f_ptr
    templated_func['numpy_linalg_norm_vec' + str(n) + 'g_ptr'] = numpy_linalg_norm_g_ptr
    templated_func['numpy_dot_vec' + str(n) + 'f_ptr'] = numpy_dot_f_ptr
    templated_func['numpy_dot_vec' + str(n) + 'g_ptr'] = numpy_dot_g_ptr
    
    macros.macro_to_scalar['numpy_clip_vec' + str(n) + 'f'] = 'numpy_clip_double'
    macros.macro_to_scalar['numpy_clip_vec' + str(n) + 'g'] = 'numpy_clip_double'
    macros.macro_to_scalar['numpy_clip_vec' + str(n) + 'f_ptr'] = 'numpy_clip_double'
    macros.macro_to_scalar['numpy_clip_vec' + str(n) + 'g_ptr'] = 'numpy_clip_double'
    
    