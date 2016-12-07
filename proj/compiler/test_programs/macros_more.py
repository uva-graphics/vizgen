# Run with:
# python compiler.py test_programs/macros_more.py --once

import numpy
import util
import math
import random

#transform(TypeSpecialize())
def f():
    aint = 2
    aint2 = 5
    ascalar = 2.0
    avec3d = numpy.ones(3)
    avec3f = numpy.ones(3, 'float32')
    avec4f = numpy.ones(4, 'float32')
    avec15d = numpy.ones(15)
    avec16f = numpy.ones(16, 'float32')
    amatd = numpy.ones((3,3))
    
    ascalar_clip = numpy.clip(ascalar, 0.0, 1.0)
    avec3d_clip   = numpy.clip(avec3d,   0.0, 1.0)
    avec4f_clip   = numpy.clip(avec4f,   0.0, 1.0)
    avec7d_n_clip = numpy.clip(numpy.ones(7), 0.0, 1.0)
    avec8f_n_clip = numpy.clip(numpy.ones(8, 'float32'), 0.0, 1.0)
    aint_clip    = numpy.clip(aint,    0,   10)

    ascalar_square = ascalar**2
    ascalar_square2 = numpy.square(ascalar)
    aint_square = aint**2
    aint_square2 = numpy.square(aint)

    afloat_to_int = int(ascalar)
    aint_to_int = int(aint)

    afloat_to_float = float(ascalar)
    aint_to_float = float(aint)

    aint_abs = abs(aint)
    aint_abs2 = numpy.abs(aint)
    afloat_abs = abs(ascalar)
    afloat_abs2 = numpy.abs(ascalar)
    
    avec3d_norm = numpy.linalg.norm(avec3d)
    avec4f_norm = numpy.linalg.norm(avec4f)
    avec15d_norm = numpy.linalg.norm(avec15d)
    avec16f_norm = numpy.linalg.norm(avec16f)
    
    avec3d_dot = numpy.dot(avec3d, avec3d)
    avec4f_dot = numpy.dot(avec4f, avec4f)
    avec15d_dot = numpy.dot(avec15d, avec15d)
    avec16f_dot = numpy.dot(avec16f, avec16f)

    arandrange = random.randrange(aint)
    arandrange2 = random.randrange(aint, aint2)
    arandrange3 = random.randrange(aint, aint+5)

    avecd_len = len(avec3d)
    avecf_len = len(avec3f)

    ascalar_pow = pow(ascalar, ascalar)
    aint_pow = pow(aint, ascalar)
    ascalar_int_pow = pow(ascalar, aint)

    math_sin = math.sin(ascalar)
    numpy_sin = numpy.sin(ascalar)

    amatd[0,:] = numpy.clip(avec3d, 0.0, 1.0)

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="macros_more.f python")
