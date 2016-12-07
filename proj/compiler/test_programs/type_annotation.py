
import vizgen
import numpy

def main(A: vizgen.ndarray('float32', (2, 2))) -> vizgen.ndarray('float32', (2, 2)):
    a = 5 + eval('0')       # type: int
    b = a + 1
    C1 = numpy.zeros((2,2))
    C2 = numpy.zeros((2,2)) + eval('0')             # type: vizgen.ndarray('float64', (2, 2))
    C3 = numpy.zeros((2,2),'float32') + eval('0')   #type: vizgen.ndarray('float32', 2)
    C4 = numpy.zeros((2,2),'float32') + eval('0')   # type: vizgen.ndarray('float32', (2, 2))
    C4b = numpy.zeros((2,2),'float32') + eval('0')   #type: vizgen.ndarray('float32', (2, 2))
    return C4 + C4b

def test():
    A = numpy.ones((2,2),'float32')
    B = main(A)

if __name__ == '__main__':
    test()
