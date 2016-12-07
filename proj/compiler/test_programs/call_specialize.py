# Run with:
# python compiler.py test_programs/call_specialize.py --once

import numpy
import util
import math

#transform(TypeSpecialize())
def g(a, b):
    return a + b

#transform(TypeSpecialize())
def f():
    af = 1.0
    bf = 2.0
    g(af, bf)

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")
