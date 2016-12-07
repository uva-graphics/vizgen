# Run with:
# python compiler.py test_programs/add.py --no-tune

import numpy
import util

#transform(TypeSpecialize())
def f():
    a = numpy.ones((10, 5, 4))
    b = numpy.ones((10, 5, 4))
    c = numpy.zeros((10, 5, 4))

    for i in range(10):
        for j in range(5):
            #transform(VectorizeInnermost())
            c[i, j, :] = a[i, j, :] + b[i, j, :]

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")
