# Run with:
# python compiler.py test_programs/add.py --no-tune

import numpy
import util

#transform(TypeSpecialize())
def g(z):
    a = numpy.ones((10, (9 if z == 0 else 5), 8, 4), 'float32')
    b = numpy.ones((10, (9 if z == 0 else 5), 8, 4), 'float32')
    c = numpy.zeros((10, (9 if z == 0 else 5), 8, 4), 'float32')

    for i in range(10):
        for j in range(a.shape[1]):
            for k in range(8):
                #transform(VectorizeInnermost())
                c[i, j, k, :] = a[i, j, k, :] + b[i, j, k, :]

    if 1:
        assert (abs(c-2)<1e-8).all()

def f():
    for z in range(2):
        g(z)

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")
