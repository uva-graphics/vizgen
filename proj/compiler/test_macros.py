
from macros import *
import util
import numpy
import pprint
import tempfile
import os
import py_ast

macros_source_test = """
np.clip(y**2+1, 0.0, 1.0)
((x)**2)**2
3*numpy.array([1, 2])**2
4+numpy.array([1, 2*(3*x)])**0.5
funcname(4)
int(3.5)
-(x)
abs(-4)
abs(-4*(x+3*math.cos(5)))
numpy.linalg.norm([1,2])
np.linalg.norm([3,4])
scipy.optimize.solve([[1.0]], [1.0])
numpy.dot([1,2],[3,4])
np.dot([1,2],[3,4])
abs(x=10)
abs(**kw)
abs(*(1,))
"""

def unique(L):
    s = set()
    ans = []
    for x in L:
        if x not in s:
            s.add(x)
            ans.append(x)
    return ans

def test_find_macros(use_py_ast=True):
    r = py_ast.get_ast(macros_source_test)
    
    for all_func_calls in [False, True]:
        L = find_macros_ast(r, macros, all_func_calls)
        L_strs = unique([match.node_str for match in L])
    #    print(L_strs)
        normalize = lambda strv: strv
        if not all_func_calls:
            L_correct = ['y**2', 'np.clip(((y**2)+1),0.0,1.0)', 'x**2', '(x**2)**2', 'numpy.array([1,2])', 'numpy.array([1,2])**2', 'numpy.array([1,(2*(3*x))])', 'numpy.array([1,(2*(3*x))])**0.5', 'int(3.5)', 'abs((-4))', 'math.cos(5)', 'abs(((-4)*(x+(3*math.cos(5)))))', 'numpy.linalg.norm([1,2])', 'np.linalg.norm([3,4])', 'numpy.dot([1,2],[3,4])', 'np.dot([1,2],[3,4])']
            assert L_strs == [normalize(xp) for xp in L_correct], L_strs
        else:
            L_correct = ['y**2', 'np.clip(((y**2)+1),0.0,1.0)', 'x**2', '(x**2)**2', 'numpy.array([1,2])', 'numpy.array([1,2])**2', 'numpy.array([1,(2*(3*x))])', 'numpy.array([1,(2*(3*x))])**0.5', 'funcname(4)', 'int(3.5)', 'abs((-4))', 'math.cos(5)', 'abs(((-4)*(x+(3*math.cos(5)))))', 'numpy.linalg.norm([1,2])', 'np.linalg.norm([3,4])', 'scipy.optimize.solve([[1.0]],[1.0])', 'numpy.dot([1,2],[3,4])', 'np.dot([1,2],[3,4])']
            assert L_strs == [normalize(xp) for xp in L_correct], L_strs
        assert len(L[2].macroL) == 2

    util.print_twocol('find_macros:', 'OK')

def test_typed_macros():
    import compiler
    if compiler.use_type_infer:
        with tempfile.NamedTemporaryFile() as f:
            fname = f.name
        assert os.system('python compiler.py test_programs/macros_more.py --once --quiet --out-file {}'.format(fname)) == 0
        with open(fname, 'rt') as f:
            out_cython = f.read()
        os.remove(fname)

        lines = [x.strip() for x in """
        ascalar_clip = numpy_clip_double(
        avec3d_clip = numpy_clip_vec3g_ptr(
        avec4f_clip = numpy_clip_vec4f_ptr(
        avec7d_n_clip = numpy_clip_vec7g(
        avec8f_n_clip = numpy_clip_vec8f(
        aint_clip = numpy_clip_int(
        ascalar_square = square_double(ascalar)
        ascalar_square2 = square_double(ascalar)
        aint_square = square_int(aint)
        aint_square2 = square_int(aint)
        afloat_to_int = float_to_int(ascalar)
        aint_to_int = int_to_int(aint)
        afloat_to_float = float_to_float(ascalar)
        aint_to_float = int_to_float(aint)
        aint_abs = libc.stdlib.abs(aint)
        aint_abs2 = libc.stdlib.abs(aint)
        afloat_abs = libc.math.fabs(ascalar)
        afloat_abs2 = libc.math.fabs(ascalar)
        avec3d_norm = numpy_linalg_norm_vec3g_ptr(
        avec4f_norm = numpy_linalg_norm_vec4f_ptr(
        avec15d_norm = numpy_linalg_norm_vec15g_ptr(
        avec16f_norm = numpy_linalg_norm_vec16f_ptr(
        avec3d_dot = numpy_dot_vec3g_ptr(
        avec4f_dot = numpy_dot_vec4f_ptr(
        avec15d_dot = numpy_dot_vec15g_ptr(
        avec16f_dot = numpy_dot_vec16f_ptr(
        arandrange = randrange_1arg(
        arandrange2 = randrange_2arg(
        avecd_len = 3
        avecf_len = 3
        ascalar_pow = libc.math.pow(
        aint_pow = libc.math.pow(
        ascalar_int_pow = libc.math.pow(
        math_sin = libc.math.sin(
        numpy_sin = libc.math.sin(
        amatd[0, :] = numpy_clip_vec3g_ptr(
    """.strip().split('\n')]
# Missing line in redbaron mode (can add this back in in the future after we transition to ast variant):
#        arandrange3 = randrange_2arg(

        for line in lines:
            if line.replace(' ', '') not in out_cython.replace(' ', ''):
                print('Missing line:', line)
                print('In source:\n', out_cython)
                assert False, 'Missing line: {}'.format(line)
    #    print(out_cython)

    util.print_twocol('typed macros:', 'OK')

if __name__ == '__main__':
    test_find_macros()
    test_typed_macros()
