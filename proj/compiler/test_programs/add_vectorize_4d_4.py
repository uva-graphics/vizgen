# Run with:
# python compiler.py test_programs/add.py --no-tune

import numpy
import util

#transform(TypeSpecialize())
def f():
    a = numpy.ones((10, 9, 8, 4), 'float32')
    b = numpy.ones((10, 9, 8, 4), 'float32')
    c = numpy.zeros((10, 9, 8, 4), 'float32')

    for i in range(10):
        for j in range(9):
            for k in range(8):
                #transform(VectorizeInnermost())
                c[i, j, k, :] = a[i, j, k, :] + b[i, j, k, :]

    assert (abs(c-2)<1e-8).all()

def test(n=None):
    return util.test_image_pipeline(
        image_func=f,
        input_imgL=tuple(),
        n=n,
        name="add.f python")
