import numpy

def foo():
    
    a = numpy.zeros(1, 100)
    b = a
    
    for i in range(99):
        for j in range(99):
            a[i] = b[j + 1]
           
    for i in range(100):
        a[i] = b[i]
