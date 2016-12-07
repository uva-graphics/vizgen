"""
Unit tests for transforms module
"""

from transforms import *

import os, os.path
import numpy
import pprint
import type_infer

debug_track_memory = False              # Debug memory leaks (should be disabled for checked-in code)

def ProgramInfo(*args, **kw):
    """
    Wrapper for test routines to skip determining which arrays are preallocated (for speed).
    """
    import compiler
    kw = dict(kw)
    kw['preallocate_arrays'] = {}
    return compiler.ProgramInfo(*args, **kw)

if debug_track_memory:
    import pympler.tracker

def source_pair_print(program_info, s, s_orig):
    for t in parse_transforms(s):
        print(t)

    print('Transforms to end:')
    print(move_transforms_to_end(program_info, s, s_orig))

    print('Transforms to body:')
    print(move_transforms_to_body(program_info, s, s_orig))

    print('Transforms round-trip 1:')
    print(move_transforms_to_body(program_info, move_transforms_to_end(program_info, s, s_orig), s_orig))

    print('Transforms round-trip 2:')
    print(move_transforms_to_end(program_info, move_transforms_to_body(program_info, s, s_orig), s_orig))

def source_pair_test(program_info, s, s_orig):
    if verbose:
        util.print_header('=== source_pair_test ===', '')
        util.print_header('s:', s)
        util.print_header('s_orig:', s_orig)
    if verbose:
        util.print_header('--> end (1)')
    s_end = move_transforms_to_end(program_info, s, s_orig)
    if verbose:
        util.print_header('--> body (1)')
    body = move_transforms_to_body(program_info, s_end, s_orig)

    if verbose:
        util.print_header('--> end (2)')
    s_end = move_transforms_to_end(program_info, body, s_orig)
    if verbose:
        util.print_header('--> body (2)')
    rt_body = move_transforms_to_body(program_info, s_end, s_orig)
    
    if body != rt_body or verbose:
        print('-'*80)
        print('=> body:')
        print('-'*80)
        print(body)
        print()
        print('-'*80)
        print('rt => body:')
        print('-'*80)
        print(rt_body)
        print()
        if body != rt_body:
            assert False
    
    end = move_transforms_to_end(program_info, move_transforms_to_body(program_info, s, s_orig), s_orig)
    rt_end = move_transforms_to_end(program_info, move_transforms_to_body(program_info, end, s_orig), s_orig)
    if end != rt_end or verbose:
        print('-'*80)
        print('=> end:')
        print('-'*80)
        print()
        print(end)
        print('-'*80)
        print('rt => end:')
        print('-'*80)
        print()
        print(rt_end)
        if end != rt_end:
            assert False

    #for s in [body, end]:
        #r = redbaron.RedBaron(s)
        #for transform in parse_transforms(program_info, s, r):
         #   assert isinstance(redbaron_util.get_transform_node(r, transform), redbaron.CommentNode)

def test_cython_type():
    if util.track_shape_list:
        target_type = "'numpy.ndarray[numpy.float64_t, ndim=3](shape=(30, 30, 3),shape_list=[(30, 30, 3)])'"
    else:
        target_type = "'numpy.ndarray[numpy.float64_t, ndim=3](shape=(30, 30, 3),shape_list=[])'"
    program_info = ProgramInfo(preprocess.preprocess_input_python('a = 10'))

    a = util.CythonType.from_value(numpy.float32(1.0), program_info)
    b = util.CythonType.from_value(numpy.float64(1.0), program_info)
    assert not a.is_subtype(b)
    assert not b.is_subtype(a)

    t = util.CythonType.from_value(numpy.ones((3,3),'int'), program_info)
    t.set_primitive_type('bool')
    assert str(t).startswith("'numpy.ndarray[numpy.bool_t, ndim=2]")
    
    a = util.CythonType.from_value(5, program_info)
    b = util.CythonType.from_value('b', program_info)
    assert str(util.union_cython_types(a, b)) == "'object'"
    b = util.CythonType.from_value(5.5, program_info)
    assert str(util.union_cython_types(a, b)) == "'double'"
    b = util.CythonType.from_value(object(), program_info)
    assert str(util.union_cython_types(a, b)) == "'object'", str(util.union_cython_types(a, b))
    a = util.CythonType.from_value(object(), program_info)
    assert str(util.union_cython_types(a, b)) == "'object'"

    # Test list types
    t=util.CythonType.from_value(1, program_info)
    t2=util.CythonType.from_cython_type([t]*3, program_info)
    assert repr(t2) == '''"['int'](shape=(3,))"''', repr(t2)
    t2 = util.CythonType.from_value([1,2,3],program_info)
    assert repr(t2) == '''"['int'](shape=(3,))"''', repr(t2)
    t2 = util.CythonType.from_value([[1,3],[2,4],[3,5]],program_info)
    assert repr(t2) == '''"[['int'](shape=(2,))](shape=(3,))"''', repr(t2)
    t=util.CythonType.from_cython_type(['int', 'int'], program_info)
    t2=util.CythonType.from_cython_type([t, t, t], program_info)
    assert repr(t2) == '''"[['int'](shape=(2,))](shape=(3,))"''', repr(t2)
    t2=util.CythonType.from_cython_type([repr(t)]*3, program_info)
    assert repr(t2) == '''"[['int'](shape=(2,))](shape=(3,))"''', repr(t2)

    assert repr(util.CythonType.from_value(numpy.zeros((30,30,3)), program_info)) == target_type
    assert repr(util.CythonType.from_cython_type(target_type, program_info)) == target_type, repr(util.CythonType.from_cython_type(target_type, program_info))

    if util.track_shape_list:
        target_type = "'numpy.ndarray[numpy.float32_t, ndim=2](shape=(30, 30),shape_list=[(30, 30)])'"
    else:
        target_type = "'numpy.ndarray[numpy.float32_t, ndim=2](shape=(30, 30),shape_list=[])'"
    assert repr(util.CythonType.from_value(numpy.zeros((30,30), numpy.float32), program_info)) == target_type
    assert repr(util.CythonType.from_value('a', program_info)) == "'str'"
    assert repr(util.CythonType.from_cython_type(repr(util.CythonType.from_value('a', program_info)), program_info)) == "'str'"
    assert util.CythonType.from_cython_type(repr(util.CythonType.from_value('a', program_info)), program_info).cython_type_str() == 'str'
    
    for value in [1.0, -1.0, False, 1, numpy.zeros((3,3)), numpy.zeros((3,4),'float32'), numpy.ones((2,2),'bool'),
                  'a', (1, 1), (1.0, 1), (1.0, numpy.zeros((2,2),'float32')), {'a': 1, 'b': numpy.ones((2,2),'int')},
                  [1.0, 1], [2.5, 1]]:
        t = util.CythonType.from_value(value, program_info)
        t.isinstance_check('a')
        repr_t = repr(t)
        assert repr(util.CythonType.from_cython_type(repr_t, program_info)) == repr_t

    t = util.CythonType.from_value([1,1.0], program_info)
    assert repr(t) == '''"['double'](shape=(2,))"''', repr(t)
    t = util.CythonType.from_value([1.0,1], program_info)
    assert repr(t) == '''"['double'](shape=(2,))"''', repr(t)

    # Test array shape inference
    t=util.CythonType.from_value(numpy.zeros((3,3)), program_info)
    t2=util.CythonType.from_value(numpy.zeros((3,4)), program_info)
    t.union_inplace(t2)
    assert repr(t) == "'numpy.ndarray[numpy.float64_t, ndim=2](shape=(3, None),shape_list=[])'"

    # Test numeric type promotion inside dict container type
    for i in range(2):
        t=util.CythonType.from_value({'a':numpy.zeros((3,3)),'b':10.5}, program_info)
        t2=util.CythonType.from_value({'a':numpy.zeros((3,4)),'b':10}, program_info)
        if i == 1:
            (t, t2) = (t2, t)
        t.union_inplace(t2)
        assert repr(t) == "{'a': 'numpy.ndarray[numpy.float64_t, ndim=2](shape=(3, None),shape_list=[])', 'b': 'double'}"

    for i in range(2):
        t1=util.CythonType.from_value(numpy.zeros(3), program_info)
        t2=util.CythonType.from_value([0,0,0], program_info)
        if i == 1:
            (t1, t2) = (t2, t1)
        t1.union_inplace(t2)
        assert repr(t1) == "'numpy.ndarray[numpy.float64_t, ndim=1](shape=(3,),shape_list=[])'"

    util.print_twocol('CythonType:', 'OK')

def test_move_transforms(print_benchmark=False):
    random.seed(0)
    
    s_orig = """
def f(n):
    ans = numpy.zeros(n)
    for i in range(n):
        ans[i] += i
    return ans
"""

    s = """
#transform(TypeSpecialize([{"n": "int", "i": "int", "ans": "numpy.ndarray[numpy.float64_t, ndim=1]"}]))
def f(n):
    ans = numpy.zeros(n)
    for i in range(n):
        ans[i] += i
    return ans
    
#transform*(Parallel(4))       # Apply transform to given line number in original source
#transform*(Parallel(6))
"""
    #source_pair_print(s, s_orig)
    program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig))
    source_pair_test(program_info, s, s_orig)
    
    s_orig = """
def f(a, b):
    def g(c):
        # Comment about g
        for i in range(10):
            a += i
    # Comment before h
    def h(q):
        # Comment after h
        for j in range(10):
            b += j
"""

    s = """
#transform(TypeSpecialize([{"a": "int", "b": "float"}]))
def f(a, b):
    #transform(TypeSpecialize([{"c": "double"}]))
    def g(c):
        # Comment about g
        #transform(Parallel())
        for i in range(10):
            a += i
    #transform(TypeSpecialize([{"c": "numpy.ndarray[numpy.float64_t, ndim=1]"}]))
    # Comment before h
    def h(q):
        # Comment after h
        for j in range(10):
            b += j
#transform*(Parallel(10))
"""
    program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig))
    #source_pair_print(s, s_orig)
    source_pair_test(program_info, s, s_orig)

    for (ifilename, filename) in enumerate('../apps/blur_one_stage/blur_one_stage_rgb.py ../apps/composite/composite_rgb.py'.split()):
        if verbose:
            print(filename)
        s_orig = open(filename, 'rt').read()
        lines = s_orig.split('\n')#[:55]
        s_orig = '\n'.join(lines)
        if verbose:
            print('original lines:', len(lines))
#        valid_lines = range(len(lines))
        valid_lines = []
        for i in range(len(lines)):
            line_strip = lines[i].strip()
            if any(line_strip.startswith(sym) for sym in 'def if print else for return elif class'.split()):
                valid_lines.append(i+1)
        annotateL = random.sample(valid_lines, len(valid_lines)*3//4)
        for annotate_lineno in annotateL:
#            if random.random() < 0.5:
#                lines.append('#transform*(Parallel({}))'.format(annotate_lineno))
#            else:
            chartag = ''.join(random.choice(string.ascii_letters) for j in range(4))
            lines.append('#transform*(TypeSpecialize({}, [{{"a": "{}"}}]))'.format(annotate_lineno, chartag))
#                    lines[i] = '#transform(Parallel())\n' + lines[i]
#                else:
#                    lines[i] = '#transform(TypeSpecialize({"a": "int"}))\n' + lines[i]
        s = '\n'.join(lines)
        if verbose:
            print('annotated lines:', len(s.split('\n')))
            print('max annotate_lineno: {}'.format(max(annotateL)))
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig))
        source_pair_test(program_info, s, s_orig)
        
        if ifilename == 0:
            T0 = time.time()
            move_transforms_to_body(program_info, s, s_orig)
            T_final = time.time()
            bench_str = 'move_transforms_to_body: {} secs ({} line file)'.format(T_final-T0, len(lines))
        
    util.print_twocol('move_transforms:', 'OK')
    if print_benchmark:
        print()
        print(bench_str)


copy_image_filenames = """
input_img_rgb = util.image_filename('temple_rgb.png')
input_img_gray = util.image_filename('temple_gray.png')
"""

copy_image_imports = """
import numpy
import util
"""

copy_image_source_base = copy_image_imports + """
def copy(in_image):
    out_image = numpy.zeros(in_image.shape)
    for y in range(in_image.shape[0]):
        for x in range(in_image.shape[1]):
            out_image[y,x] = in_image[y,x]
    return out_image

""" + copy_image_filenames

copy_image_source_two_types_test = """
def test(n=1):
    copy(numpy.zeros((50, 50, 3)))
    copy(numpy.zeros((50, 50)))
    ans1 = util.test_image_pipeline_filename(copy, (input_img_rgb,), n, name='copy rgb', verbose=False)
    ans2 = util.test_image_pipeline_filename(copy, (input_img_gray,), n, grayscale=True, name='copy gray', verbose=False)
    return util.combine_tests((ans1, ans2))
"""

copy_image_source_one_type_test = """
def test(n=1):
    copy(numpy.zeros((50, 50)))
    ans1 = util.test_image_pipeline_filename(copy, (input_img_gray,), n, grayscale=True, name='copy gray', verbose=False)
    return util.combine_tests((ans1,))
"""

copy_image_source_two_types_2d_test = """
def test(n=1):
    copy(numpy.zeros((50, 50)))
    copy(numpy.zeros((50, 50), numpy.float32))
    ans1 = util.test_image_pipeline_filename(copy, (input_img_gray,), n, grayscale=True, name='copy gray', verbose=False)
    ans2 = util.test_image_pipeline_filename(copy, (input_img_gray,), n, grayscale=True, name='copy gray', verbose=False, numpy_dtype=numpy.float32)
    return util.combine_tests((ans1, ans2))
"""

copy_image_source_two_types = copy_image_source_base + copy_image_source_two_types_test

copy_image_source_one_type = copy_image_source_base + copy_image_source_one_type_test

copy_image_source_two_types_2d = copy_image_source_base + copy_image_source_two_types_2d_test


def test_typespecialize():
    import compiler

    path = './'

    for (itest, s_orig) in enumerate([copy_image_source_one_type, copy_image_source_two_types]):
        program_info = compiler.ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types={}, preallocate=False)
        types = type_infer.type_infer(program_info, verbose=False)['types']
        if verbose:
            print('itest={}, types:'.format(itest))
            print(types)
        correct_types = {'copy': [{'in_image': 'numpy.ndarray[numpy.float64_t, ndim=2]', 'out_image': 'numpy.ndarray[numpy.float64_t, ndim=2]', 'y': 'int', 'x': 'int'}, {'in_image': 'numpy.ndarray[numpy.float64_t, ndim=3]', 'out_image': 'numpy.ndarray[numpy.float64_t, ndim=3]', 'y': 'int', 'x': 'int'}], 'test': [{'n': 'int', 'ans1': 'object'}]}
        assert sorted(types) == sorted(correct_types), (types, correct_types)

        if itest == 1:
            lineno = 4
            program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types)
            program_info.add_new_transform(TypeSpecialize(program_info, lineno, types['copy']))
            (s, transformL) = compiler.add_transforms_to_code(program_info, s_orig, s_orig, program_info.transformL)
            s = compiler.apply_transforms_and_finalize(program_info, s)
            assert 'cdef int' in s
            assert 'numpy.ndarray' in s
            compiler.run_code(program_info.path, s, clean=True, verbose=False, cython=True)
    
        extra_info = {}
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types)
        program_info.add_new_transform(TypeSpecialize)
        program_info.run(extra_info=extra_info)

        assert 'cdef' in extra_info['source']

        if itest == 1:
            try:
                program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types)
                program_info.add_new_transform(TypeSpecialize)
                program_info.add_new_transform(TypeSpecialize)
                program_info.run()
                ok = False
            except MutateError:
                ok = True
            if not ok:
                raise ValueError('applying TypeSpecialize() twice to a single-function program should fail')

    util.print_twocol('TypeSpecialize:', 'OK')

def test_parallel():

    # The source code variable copy_image_source_two_types does not succeed without unrolling because it has a Python object access in it.
    for (itest, s_orig) in enumerate([copy_image_source_one_type, copy_image_source_two_types_2d]):
        if verbose:
            util.print_header('s_orig:', s_orig)
        
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), types=None)
        if verbose:
            util.print_header('types:', program_info.types)
        if len(program_info.types['copy']) != itest+1:
            print(itest)
            print(s_orig)
            pprint.pprint(program_info.types)
            assert False
        program_info.add_new_transform(TypeSpecialize)
        program_info.add_new_transform(Parallel(program_info, 10))
        extra_info = {}
        program_info.run(verbose=False, clean=True, extra_info=extra_info)
        return_count = extra_info['source'].count('return out_image')
        assert return_count == 2 + itest, (return_count, 2 + itest)
        assert 'cdef' in extra_info['source']
        assert extra_info['source'].count('prange') == 1 + itest
    util.print_twocol('Parallel:', 'OK')

def test_typespecialize_no_checks():
    
    s_base = copy_image_imports + """
#transform(TypeSpecialize(checks=False))
def copy(in_image):
    out_image = numpy.zeros(in_image.shape)
    for y in range(in_image.shape[0]):
        for x in range(in_image.shape[1]):
            out_image[y,x] = in_image[y,x]
    return out_image
    """ + copy_image_filenames
    
    for (itest, s_cat) in enumerate([copy_image_source_one_type_test, copy_image_source_two_types_test]):
        s_orig = s_base + s_cat
        
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), types=None)
        if verbose:
            util.print_header('types:', program_info.types)
        assert len(program_info.types['copy']) == 1+itest, (len(program_info.types['copy']), 1+itest)
        extra_info = {}
        program_info.run(verbose=False, clean=True, extra_info=extra_info)
        return_count = extra_info['source'].count('return out_image')
        assert return_count == 2+itest, (return_count, 2+itest)
        assert 'cdef' in extra_info['source']
    
    util.print_twocol('TypeSpecialize(checks=False):', 'OK')

def sub_test_loopimplicit():
    path = './test_programs/'
#    for filename in ['blur_one_stage_annotated.py', 'blur_one_stage_annotated2.py']:
    for filename in ['blur_one_stage_annotated2.py']:
#        print('testing {}'.format(filename))
        s_orig = open(os.path.join(path, filename), 'rt').read()

        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types=None)
        extra_info = {}
        run_info = program_info.run(clean=True, verbose=False, cython=True, extra_info=extra_info)
        initial_run_info = program_info.get_initial_run_info()
#        util.print_header('Cython source:', extra_info['source'])
        speedup = initial_run_info['time'] / run_info['time']
        assert speedup >= 500.0, ('LoopImplicit failed', {'speedup': speedup, 'source': extra_info['source']})
#        print('speedup:', speedup)
#        print('original time:', program_info.test_result['time'], 'annotated time:', run_info['time'])

    util.print_twocol('LoopImplicit:', 'OK (parallelism speedup for small input image: {:.0f}x)'.format(speedup))

def test_loopimplicit():
    import compiler

    s_base1 = copy_image_imports + """
#transform(TypeSpecialize(checks=False))
def copy(in_image):
    out_image = numpy.zeros(in_image.shape)
    #transform(Parallel())
    for y in range(in_image.shape[0]):
        for x in range(in_image.shape[1]):
            #transform(LoopImplicit())
            out_image[y,x] = in_image[y,x]*0.6 + in_image[y,x]*0.4
    return out_image
    """ + copy_image_filenames

    s_base2 = copy_image_imports + """
#transform(TypeSpecialize(checks=False))
def copy(in_image):
    out_image = numpy.zeros(in_image.shape)
    #transform(Parallel())
    for y in range(in_image.shape[0]):
        #transform(LoopImplicit())
        out_image[y] = in_image[y]*0.6 + in_image[y]*0.4
    return out_image
    """ + copy_image_filenames
    
    """
    s_base = copy_image_source_base
    for (itest, s_cat) in enumerate([copy_image_source_one_type_test, copy_image_source_two_types_test]):
        s_orig = s_base + s_cat
        extra_info = {}
        program_info = compiler.ProgramInfo(s_orig, types=None, preallocate_arrays={})
        run_info = program_info.run(verbose=False, clean=True, extra_info=extra_info)
        util.print_header('source of original program:', extra_info['source'])
        print('run_info:', run_info)
    """
    
    for (i_sbase, s_base) in enumerate([s_base1, s_base2]):
        for (itest, s_cat) in enumerate([copy_image_source_one_type_test, copy_image_source_two_types_test]):
            s_orig = s_base + s_cat
            
            program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), types=None)
            if verbose:
                util.print_header('types:', program_info.types)
            assert len(program_info.types['copy']) == 1+itest, (len(program_info.types['copy']), 1+itest)
            extra_info = {}
            run_info = program_info.run(verbose=False, clean=True, extra_info=extra_info)
            return_count = extra_info['source'].count('return out_image')
            assert return_count == 2+itest, (return_count, 2+itest)
            assert 'cdef' in extra_info['source']
            assert 'prange' in extra_info['source']

            if verbose:
                util.print_header('final source:', extra_info['source'])
                print('run_info:', run_info)
            found_trailer = loopimplicit_trailer in extra_info['source']
            target_trailer = (s_base != s_base1 or itest != 0)
            assert found_trailer == target_trailer, (found_trailer, target_trailer)

    sub_test_loopimplicit()

def test_mutate(verbose=verbose, ntest=None):
    random.seed(0)
    verbose = False
    import compiler
    path = './test_programs/'
    ntested = 0
    
    for filename in ['blur_one_stage.py']:
        s_orig = open(os.path.join(path, filename), 'rt').read()

        required_success = [[TypeSpecialize], [TypeSpecialize, LoopImplicit, LoopImplicit, LoopImplicit, LoopImplicit, Parallel]]
        
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types=None)
        all_transforms_filtered = [cls for cls in all_transforms if cls != VectorizeInnermost]

        for is_required in range(2):
            subtests = len(all_transforms_filtered) if not is_required else len(required_success)
            for j in range(subtests):
                ntransforms = 1 if not is_required else len(required_success[j])
                program_info.transformL = []
                
                for i in range(ntransforms):
                    while True:
                        cls = all_transforms_filtered[j] if not is_required else required_success[j][i]
                        #print('current transform list:', program_info.transformL, 'adding', cls)
                        try:
                            program_info.add_new_transform(cls)
                            break
                        except MutateError:
                            if ntransforms == 1:
                                raise
                            continue
                def print_current():
                    print('transform list:', program_info.transformL, 'is_required:', is_required)
                if verbose:
                    print_current()
                
                try:
                    run_info = program_info.run(clean=True, verbose=False, verbose_on_fail=False, clean_on_fail=True, cython=True)
                    ok = True
                except (TransformError, compiler.RunFailed):
                    ok = False
                    if is_required:
                        if not verbose:
                            print_current()
                        raise
                ntested += 1
                
                if ntest is not None and ntested >= ntest:
                    print('mutate: finished {} tests, stopping'.format(ntest))
                    return
                if verbose:
                    print('result: ', ok)
                    print()

                if isinstance(program_info.transformL[0], TypeSpecialize):
#                    before_check = program_info.transformL[0].checks
#                    print('checking', program_info.transformL[0])
                    program_info.transformL[0].mutate()
#                    assert program_info.transformL[0].checks != before_check
#                    print('after mutate:', program_info.transformL[0])

    s_test = """
import numpy
def foo():
    a = numpy.zeros(10)
    for i in range(10):
        for j in range(10):
            a[j] = 1
    return a
"""
    program_info = ProgramInfo(preprocess.preprocess_input_python(s_test))
    program_info.add_new_transform(Parallel)
    program_info.transformL[0].mutate()
    assert len(program_info.parallel_mapping) == 2
                
    util.print_twocol('mutate:', 'OK')

def test_dependencies():
    path = './test_programs/'
    
    for (i, filename) in enumerate(['blur_one_stage.py', 'blur_one_stage_full.py']):
        if i == 0:
            continue
        s_orig = open(os.path.join(path, filename), 'rt').read()
        
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types=None)
        program_info.add_new_transform(Parallel)
        r = program_info.transformL[0].dependencies()
        if i == 0:
            assert len(r) == 6
            assert isinstance(r[0], TypeSpecialize)
            assert isinstance(r[1], TypeSpecialize)
            assert repr(r[2:]) == '[LoopImplicit(21), LoopImplicit(27), LoopImplicit(33), LoopImplicit(37)]'

            seen = set()
            for j in range(30):
                L1 = resolve_dependencies(program_info.transformL, randomize=False)
                L2 = resolve_dependencies(L1, randomize=False)
                def check(L):
                    assert len(L) == 6
                    assert isinstance(L[0], TypeSpecialize)
                    assert isinstance(L[1], Parallel)
                    assert L[1].line == program_info.transformL[0].line
                    assert repr(L[2:]) == '[LoopImplicit(21), LoopImplicit(27), LoopImplicit(33), LoopImplicit(37)]'
                    return L[0].checks
                
                assert L1 == L2
                v = check(L1)
                seen.add(v)
                if len(seen) == 2:
                    break
            if len(seen) < 2:
                raise ValueError('did not see both TypeSpecialize(checks=True) and TypeSpecialize(checks=False)')
        else:
            #print('dependencies:', r)
            program_info.resolve_dependencies(randomize=False)
            L = program_info.transformL
            assert len(L) == 12
            assert isinstance(L[0], TypeSpecialize)
            assert isinstance(L[1], Parallel)
            assert repr(L[2:]) == '[LoopImplicit(18), LoopImplicit(23), LoopImplicit(29), LoopImplicit(35), LoopImplicit(41), LoopImplicit(47), LoopImplicit(53), LoopImplicit(59), LoopImplicit(65), LoopImplicit(70)]'
            #print('revised transforms:', program_info.transformL)
            #run_info = program_info.run(clean=False, verbose=True, verbose_on_fail=False, clean_on_fail=False, cython=True)
            #print('run_info:', run_info)

    util.print_twocol('dependencies:', 'OK')

def test_is_consistent():
    
    path = './'
    s_orig = """
import numpy
import util

#transform(TypeSpecialize(checks=False))
def copy(in_image):
    out_image = numpy.zeros(in_image.shape)
    #transform(Parallel())
    for y in range(in_image.shape[0]):
        #transform(Parallel())
        for x in range(in_image.shape[1]):
            out_image[y,x] = in_image[y,x]*0.6 + in_image[y,x]*0.4
    return out_image
"""
    program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types={'copy': []})
    program_info.get_transforms_from_source()
    assert isinstance(program_info.transformL[1], Parallel)
    assert isinstance(program_info.transformL[2], Parallel)
    assert not program_info.transformL[1].is_consistent()
    
    s_orig = """
import numpy
import util

def copy(in_image):
    out_image = numpy.zeros(in_image.shape)
    for y in range(in_image.shape[0]):
        for x in range(in_image.shape[1]):
            out_image[y,x] = in_image[y,x]*0.6 + in_image[y,x]*0.4
    return out_image
"""
    program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types={'copy': []})
    program_info.transformL.append(TypeSpecialize(program_info, 7, []))
    program_info.transformL.append(Parallel(program_info, 10))
    program_info.transformL.append(Parallel(program_info, 11))
    assert not program_info.transformL[1].is_consistent()

    util.print_twocol('is_consistent:', 'OK')

def test_apply_macros():
    
    s_base = copy_image_imports + """
import numpy as np

#transform(TypeSpecialize(checks=False))
def copy(in_image):
    out = numpy.zeros(in_image.shape)
    a = 0.10 if len(in_image.shape) == 3 else 0.109
    b = numpy.clip(a, 0, 1)
    x = 0.105
    y = numpy.clip(x, 0.0, 1.0)**2
    u = numpy.array([0.0, 1.0, 2.0])
    v = numpy.array([1.0, 2.0, 5.0])
    r = np.dot(u, v)**0.5/20.0
    
    out[0,0] = b
    out[0,1] = y
    out[0,2] = r
    
    return out
    """ + copy_image_filenames
    
    for (itest, s_cat) in enumerate([copy_image_source_two_types_test]):
        s_orig = s_base + s_cat
        
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), types=None)
        if verbose:
            util.print_header('types:', program_info.types)
        extra_info = {}
        program_info.run(verbose=False, clean=True, extra_info=extra_info)
        s = extra_info['source']
#        print(s)
        assert 'libc.math.sqrt(numpy_dot_vec3g_ptr(' in s, s
        assert 'square_double(numpy_clip_double(' in s

    util.print_twocol('ApplyMacros:', 'OK')
    
def test_loop_fusion():

    s_base = copy_image_imports + """
def copy(in_image):
    temp_var = numpy.zeros(in_image.shape)
    out = numpy.zeros(in_image.shape)
    for i in range(in_image.shape[0]):
        for j in range(1, in_image.shape[1]):
            temp_var[i, j] = in_image[i, j] ** 1.5
        temp_var[i, 0] = in_image[i, 0] ** 1.5
    #transform(LoopFusion())
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            out[i, j] = temp_var[i, j] ** 2
    return out
""" + copy_image_filenames

    for (itest, s_cat) in enumerate([copy_image_source_one_type_test]):
        s_orig = s_base + s_cat
        program_info = ProgramInfo(preprocess.preprocess_input_python(s_orig), types=None)
        if verbose:
            util.print_header('types:', program_info.types)
        extra_info = {}
        program_info.run(verbose=False, clean=True, extra_info=extra_info)
        s = extra_info['source']
        assert 'temp_var' not in s
        
    util.print_twocol('LoopFusion:', 'OK')

def main():
    if debug_track_memory:
        memory_tracker = pympler.tracker.SummaryTracker()
    for func in [
        test_cython_type,
        test_move_transforms,
        test_typespecialize,
        test_typespecialize_no_checks,
        test_parallel,
        test_loopimplicit,
        test_mutate,
        test_dependencies,
        test_is_consistent,
        test_apply_macros,
        test_loop_fusion]:
        for i in range(3 if debug_track_memory else 1):
            func()
            if debug_track_memory:
                memory_tracker.print_diff()
        if debug_track_memory:
            print()

if __name__ == '__main__':
    main()
