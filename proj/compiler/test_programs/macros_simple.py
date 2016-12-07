# Run with:
# python compiler.py test_programs/macros.py --no-tune

import numpy
import util
import math

#transform(TypeSpecialize())
def f():
    a = 1.0
    b = 2.0
    c = numpy.arctan2(a, b)
    c = numpy.cos(a) + math.atan(b)

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")
