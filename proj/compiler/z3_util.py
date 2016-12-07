
import z3
import numpy
import json

class ProveSmallestError(Exception):
    pass

def prove_smallest(L, z3_vars, constraints_L, cached=True, cache={}):
    """
    Return element of L which is smallest by using a theorem prover.
    
    Given a list of Python expressions (L), list of all variable names used by those expressions (z3_vars), and list of constraint strings (constraints_L).
    
     Attempts to prove and return the expression that can be proved to always be smaller than all other expressions. If this fails, returns any expression that can be proved to always be less than or equal to other expressions. If that fails, return the first item of the list.
    """
    if len(L) == 0:
        raise ProveSmallestError('expected non-empty list for prove_smallest()')
    
    if cached:
        cache_key = json.dumps((sorted(L), sorted(z3_vars), sorted(constraints_L)))
        if cache_key in cache:
            return cache[cache_key]

    always_less = numpy.zeros((len(L), len(L)), 'bool')
    always_leq = numpy.ones((len(L), len(L)), 'bool')

    d = {}
    for var in z3_vars:
        d[var] = z3.Int(var)
#    print('prove_smallest:', L, z3_vars, constraints_L)

    for i in range(len(L)):
        for j in range(len(L)):
            if j == i:
                continue

            # Attempt to prove that L[i] < L[j] always by showing that L[i] >= L[j] causes a contradiction
            s = z3.Solver()
            for constraint in constraints_L:
                try:
                    s.add(eval(constraint, d))
                except:
                    raise ProveSmallestError
            try:
                s.add(eval('({}) >= ({})'.format(L[i], L[j]), d))
            except:
                raise ProveSmallestError
            check = s.check()
            
            always_less[i][j] = (check == z3.unsat)
    
            # Attempt to prove that L[i] <= L[j] always by showing that L[i] > L[j] causes a contradiction
            s = z3.Solver()
            for constraint in constraints_L:
                try:
                    s.add(eval(constraint, d))
                except:
                    raise ProveSmallestError
            try:
                s.add(eval('({}) > ({})'.format(L[i], L[j]), d))
            except:
                raise ProveSmallestError
            check = s.check()
            
            always_leq[i][j] = (check == z3.unsat)

#    print('always_less:')
#    print(always_less)
#    print('always_leq:')
#    print(always_leq)

    has_ans = False
    for (k, A) in enumerate([always_less, always_leq]):
#        print('k=', k)
        for i in range(len(L)):
            all_succeed = True
            for j in range(len(L)):
                if j == i:
                    continue
                if not A[i][j]:
                    all_succeed = False
                    break
            if all_succeed:
                ans = L[i]
                has_ans = True
                break
        if has_ans:
            break
    if not has_ans:
        ans = L[0]

    if cached:
        cache[cache_key] = ans
    return ans
