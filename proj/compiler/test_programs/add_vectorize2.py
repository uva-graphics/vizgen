# Run with:
# python compiler.py test_programs/add.py --no-tune

import numpy
import util

#transform(TypeSpecialize())
def f():
    a = numpy.ones(2)
    b = numpy.ones(2)
    c = numpy.zeros(2)

    #transform(VectorizeInnermost())
    c[:] = a[:] + b[:]

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")
