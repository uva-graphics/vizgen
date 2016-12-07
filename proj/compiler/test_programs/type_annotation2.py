import numpy

def f():
    a = eval('0')                       # type: int
    b = eval("'abc'")                   # type: str
    c = eval('numpy.zeros([3, 3])')     # type: vizgen.ndarray('float64', [3, 3])
    
    A = 0
    B = 'abc'
    C = numpy.zeros([3, 3])
    
    d = c.shape[0]
    D = C.shape[0]

def test():
    f()

