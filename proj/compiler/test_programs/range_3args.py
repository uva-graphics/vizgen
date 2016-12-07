# Run with:
# python compiler.py --once test_programs/range_3args.py

import time
import numpy

default_range = range

#transform(TypeSpecialize())
def f(a, b, c):
    count = 0
    for i in range(a, b, c):
        count += 1
    count2 = 0
    for i in default_range(a, b, c):
        count2 += 1
    assert count == count2
    print('OK', a,b,c)

def test(n=None):
    T0 = time.time()
    for a in range(-10, 10):
        for b in range(-10, 10):
            for c in range(-10, 10):
                if c != 0:
                    f(a, b, c)
    return {'time': time.time()-T0, 'error': 0.0}

if __name__ == '__main__':
    test()