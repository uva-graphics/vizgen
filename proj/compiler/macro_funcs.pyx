
"""
C functions called by text-replacing macros in macros.py.
"""

cdef extern from "stdbool.h":
  pass

cimport cython
cimport numpy
cimport libc.math
import random
from libc.stdint cimport uint32_t

def range_shuffled(*args):
    L = list(range(*args))
    L0 = list(L)
    random.shuffle(L)
    while len(L) >= 2 and L == L0:
        random.shuffle(L)
    return L

cdef int randrange_1arg(int x) nogil:
    return libc.stdlib.rand()%x

cdef int randrange_2arg(int x, int y) nogil:
    return x + (libc.stdlib.rand()%(y-x))

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
    
cdef int randrange_seed(int seed, int start, int stop) nogil:
    cdef uint32_t rand2_u
    cdef uint32_t rand2_v
    cdef uint32_t rand_result
    rand2_u = <uint32_t>seed
    rand2_v = ~rand2_u
    rand2_u = rand2_u ^ ((seed & 65535) << 16)
    rand2_v = 36969 * (rand2_v & 65535) + (rand2_v >> 16)
    rand2_u = 18000 * (rand2_u & 65535) + (rand2_u >> 16)
    rand_result = (rand2_v << 16) + (rand2_u & 65535)
    return start + rand_result % (stop - start)

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

cdef extern from "vector_headers.h":
    ctypedef float v2float
    ctypedef float v4float
    ctypedef double v2double
    ctypedef double v4double
