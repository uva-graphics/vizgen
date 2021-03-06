# Run with:
# python compiler.py test_programs/add.py --no-tune

import numpy
import util

#transform(TypeSpecialize())
def f():
    a = numpy.ones(4)
    b = numpy.ones(4)
    c = numpy.zeros(4)

    #transform(LoopImplicit())
    c[:] = a[:] + b[:]

    d = numpy.ones(4)
    e = numpy.ones(4)
    f = numpy.zeros(4)

    #transform(LoopImplicit())
    f[:] = d[:] + e[:]

    g = numpy.ones(4)
    h = numpy.ones(4)
    i = numpy.zeros(4)

    #transform(LoopImplicit())
    i[:] = g[:] + i[:]

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")