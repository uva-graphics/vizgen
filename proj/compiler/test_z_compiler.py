
import os
import util
import tempfile
import os

def test_compiler_prealloc():
    with tempfile.NamedTemporaryFile() as f:
        fname = f.name
    assert os.system('python compiler.py --once --quiet --out-file {} test_programs/prealloc_test.py'.format(fname)) == 0
    with open(fname, 'rt') as f:
        s = f.read()
        assert '_prealloc_A_prealloc_global' in s
        assert '_prealloc_A_prealloc_global = numpy.zeros(3)' in s
        assert 'A_prealloc = _prealloc_A_prealloc_global' in s
    os.remove(fname)
    util.print_twocol('compiler, preallocate:', 'OK')

def test_compiler_default_args():
    assert os.system('python compiler.py --once --quiet test_programs/default_args.py') == 0
    util.print_twocol('compiler, default args:', 'OK')

def test_compiler_vectorize():
    assert os.system('python compiler.py --once --quiet test_programs/add_vectorize_4d_4.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/add_vectorize_4d_4_changing.py') == 0
    util.print_twocol('compiler, vectorize:', 'OK')

def test_compiler_harris_corner():
    assert os.system('python compiler.py --once --quiet test_programs/harris_corner_annotated.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/harris_corner_annotated2.py') == 0
    util.print_twocol('compiler, harris_corner:', 'OK')

def test_compiler_raytracer():
    assert os.system('python compiler.py --once --quiet test_programs/raytracer_short_unrolled_annotated.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/raytracer_short_annotated.py') == 0
    util.print_twocol('compiler, raytracer:', 'OK')

def test_compiler_blur():
    assert os.system('python compiler.py --once --quiet test_programs/blur_one_stage_gray_remove_conditionals2.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/blur_one_stage_gray_remove_conditionals.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/blur_one_stage_rgb_remove_conditionals.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/blur_one_stage_no_parallel.py') == 0
    util.print_twocol('compiler, blur:', 'OK')

def test_compiler_arraystorage():
    assert os.system('python compiler.py --once --quiet test_programs/blur_one_stage_arraystorage.py') == 0
    assert os.system('python compiler.py --once --quiet test_programs/blur_one_stage_noarraystorage.py') == 0

def test_compiler_macros():
    import transforms
    with tempfile.NamedTemporaryFile() as f:
        fname = f.name

    assert os.system('python compiler.py --once --quiet --out-file {} test_programs/macros_simple.py'.format(fname)) == 0
    with open(fname, 'rt') as f:
        s = f.read()
        assert 'c = libc.math.atan2(a, b)' in s
        try:
            assert 'c = libc.math.cos(a) + libc.math.atan(b)' in s
        except:
            assert 'c = (libc.math.cos(a) + libc.math.atan(b))' in s
    os.remove(fname)
    
    assert os.system('python compiler.py --once --quiet --out-file {} test_programs/call_specialize.py'.format(fname)) == 0
    with open(fname, 'rt') as f:
        s = f.read()
        try:
            assert 'g' + transforms.typespecialize_header + '_double_double' + transforms.typespecialize_trailer + '(af,bf)' in s
        except:
            assert 'g' + transforms.typespecialize_header + '_double_double' + transforms.typespecialize_trailer + '(af, bf)' in s
    os.remove(fname)

    util.print_twocol('compiler, macros:', 'OK')

def test_compiler_range_3args():
    assert os.system('python compiler.py --quiet --once test_programs/range_3args.py') == 0
    
def test_compiler_fusion():
    assert os.system('python compiler.py --quiet --once test_programs/blur_two_stage_if_condition.py') == 0

if __name__ == '__main__':
    test_compiler_prealloc()
    test_compiler_raytracer()
    test_compiler_vectorize()
    test_compiler_harris_corner()
    test_compiler_blur()
    test_compiler_default_args()
    test_compiler_arraystorage()
    test_compiler_macros()
    test_compiler_range_3args()
