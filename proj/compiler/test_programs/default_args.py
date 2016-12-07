# Run with:
# python compiler.py --once test_programs/default_args.py

import time

#transform(TypeSpecialize())
def f(a, b=10.0, c=20.0):
    for i in range(100000):
        d = a + b + c

def test(n=None):
    T0 = time.time()
    f(1.0)
    f(1.0, 2.0)
    f(1.0, 2.0, 3.0)
    return {'time': time.time()-T0, 'error': 0.0}