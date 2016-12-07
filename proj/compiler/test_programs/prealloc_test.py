
import numpy
import copy

eps = 1e-7

def fill_zeros(A):                    # TODO: fix the prealloc codegen so that this does not need to be manually called
    A[:] = 0

def f():
    A_prealloc = numpy.zeros(3)       # Aliases: f.A_prealloc
    B_prealloc = numpy.zeros(3)       # Needs to be re-filled with zeros each time. Aliases: f.B_prealloc
    fill_zeros(B_prealloc)
    B_prealloc += 1
    C_prealloc = numpy.zeros(3)       # Now can be preallocated due to better alias analysis
    C_ref = C_prealloc                # Aliases: f.C_noprealloc. In this particular case this is OK but we perform a non-flow-sensitive
                                      # alias analysis so we flag this as non-preallocatable.
    d = numpy.dot(A_prealloc+1, B_prealloc)
    sumA = A_prealloc.sum()
    minA = A_prealloc.min()
    maxA = A_prealloc.max()
    assert abs(d - 3) < eps
    return d

def g(a):
    C_prealloc = numpy.zeros(3)       # Needs to be re-filled with zeros each time. Aliases: g.C_prealloc
    fill_zeros(C_prealloc)
    C_prealloc[0] += 1
    assert (numpy.sum(C_prealloc) - 1) < eps
    D_noprealloc = numpy.zeros(3)     # Aliases: g.D_noprealloc
    D_noprealloc += a
    assert abs(numpy.sum(D_noprealloc) - a*3) < eps
    return D_noprealloc               # Aliases: g.D_noprealloc

def h():
    D_alias1 = g(1.0)                   # Aliases g.D_noprealloc
    D_alias2 = g(2.0)                   # Aliases g.D_noprealloc. Two references to same array present, disable preallocation.
    assert (numpy.dot(D_alias1, D_alias2)-6) < 1e-7

def h2():
    L = []
    for i in range(3):
        E_noprealloc = numpy.zeros(3)   # Aliases h2.E_noprealloc
        E_noprealloc += i
        L.append(E_noprealloc)          # Passes array to function we cannot analyze through. Could have two references present due
    Lp = [numpy.sum(A) for A in L]      # to being in a loop, so disable prealloc.
    assert (numpy.sum(Lp) - 9) < eps

A_L = []

def h3():
    for i in range(3):
        A_noprealloc = numpy.zeros(3)   # Aliases h3.A_noprealloc
        A_noprealloc += i
        A_L.append(A_noprealloc)        # Same deal as in h2.E_noprealloc. Disable prealloc.
    Lp = [numpy.sum(A) for A in A_L]
    assert (numpy.sum(Lp) - 9) < eps

def h4():
    A_prealloc = numpy.zeros(3)             # Aliases h4.A_prealloc
    B_prealloc = numpy.array(A_prealloc)    # Aliases h4.B_prealloc
    fill_zeros(B_prealloc)
    B_prealloc += 1
    C_prealloc = copy.copy(A_prealloc)      # Aliases h4.C_prealloc
    fill_zeros(C_prealloc)
    C_prealloc += 1
    assert abs(numpy.dot(A_prealloc+1, B_prealloc) - 3) < eps
    assert abs(numpy.dot(A_prealloc+1, C_prealloc) - 3) < eps

    L = []
    for i in range(2):
        A_noprealloc = numpy.zeros(3)           # Aliases h4.A_noprealloc
        A_noprealloc += i+1
        L.append(A_noprealloc)                  # Argument to append aliases h4.A_noprealloc. Cannot analyze through append, so disable prealloc.
        
        A_noprealloc2 = numpy.zeros(3)          # Same situation as h4.A_noprealloc
        A_noprealloc2 += i+1
        L.append(numpy.asarray(A_noprealloc2))

        A_noprealloc3 = numpy.zeros(3)          # Same situation as h4.A_noprealloc
        A_noprealloc3 += i+1
        L.append(A_noprealloc3[:])
    
        A_prealloc4 = numpy.zeros(3)            # Needs to be re-filled with zeros each time. Aliases h4.A_prealloc4
        fill_zeros(A_prealloc4)
        A_prealloc4 += i+1
        L.append(numpy.array(A_prealloc4))      # Argument aliases only the temporary value of numpy.array(A_prealloc4). So A_prealloc4 can be prealloced.

        A_prealloc5 = numpy.zeros(3)            # Needs to be re-filled with zeros each time. Same situation as h4.A_prealloc4
        fill_zeros(A_prealloc5)
        A_prealloc5 += i+1
        L.append(copy.copy(A_prealloc5))
    
        A_prealloc6 = numpy.zeros(3)
        fill_zeros(A_prealloc6)
        A_prealloc6 += i+1
        L.append(A_prealloc6.copy())
        
    Lp = [numpy.sum(A) for A in L]
    assert (numpy.sum(Lp) - (1+2)*3*6) < eps

def test():
    f()
    h()
    h2()
    h3()
    h4()
    print('prealloc_test: OK')
    return {'time': 0.0}

if __name__ == '__main__':
    test()
