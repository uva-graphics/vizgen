
import util
import time
import pprint
import test_transforms
import type_infer

def test_type_infer_simple():
    type_infer.test_type_infer_simple()
    type_infer.test_type_infer_simple(prealloc=True)
    util.print_twocol('test_type_infer_simple:', 'OK')

def test_type_infer(strict=True, all_programs=False, verbose=False, extra_verbose=False):
    import compiler

    path = './'

    filename_L = """
test_programs/blur_float32.py
test_programs/simple_pm.py
__simple_test_source2
__simple_test_source
test_programs/pacman.py
test_programs/raytracer_short_annotated.py
test_programs/interpolate_annotated.py
test_programs/harris_corner_annotated.py
__copy_image_source_one_type
__copy_image_source_two_types
test_programs/add.py
test_programs/add_lots_of_slices.py
test_programs/blur_one_stage_full.py
""".split()
    if all_programs:
        filename_L = """
../apps/composite_gray/composite.py
../apps/blur_one_stage_gray/blur_one_stage.py
../apps/blur_two_stage_gray/blur_two_stage.py
../apps/camera_pipe/camera_pipe_fcam.py
../apps/composite/composite_4channel.py
../apps/local_laplacian/local_laplacian_fuse.py
../apps/interpolate/interpolate.py
../apps/bilateral_grid/bilateral_grid_clean.py
../apps/blur_two_stage/blur_two_stage_4channel.py
../apps/raytracer/raytracer_short_simplified_animate.py
../apps/optical_flow_patchmatch/optical_flow_patchmatch.py
../apps/pacman/pacman_clean.py
../apps/blur_one_stage/blur_one_stage_4channel.py
../apps/harris_corner/harris_corner.py
../apps/mandelbrot/mandelbrot_animate.py
../apps/composite/composite_rgb.py
""".split() + filename_L

    get_macros = False
    
    for filename in filename_L:
        if verbose or extra_verbose:
            print(filename)
        if filename == '__copy_image_source_one_type':
            s_orig = test_transforms.copy_image_source_one_type
        elif filename == '__copy_image_source_two_types':
            s_orig = test_transforms.copy_image_source_two_types
        elif filename == '__simple_test_source':
            s_orig = type_infer.simple_test_source
        elif filename == '__simple_test_source2':
            s_orig = type_infer.simple_test_source2
        else:
            s_orig = open(filename, 'rt').read()
        if verbose:
            print()
            util.print_header('Testing {}'.format(filename))

        T0_program_info = time.time()
        program_info = compiler.ProgramInfo(s_orig, path, types={}, preallocate=False, compile_dir='out/type_infer')
        T_program_info = time.time() - T0_program_info
        
        T0 = time.time()
        types_unit_test = compiler.get_types_py_ast(path, s_orig)['types']
        T = time.time() - T0

        with util.SuppressOutput(verbose=extra_verbose):
            T0_infer = time.time()
            type_infer_d = type_infer.type_infer(program_info, both_formats=True, verbose=verbose, get_macros=get_macros)
            types_infer = type_infer_d['types']
            types_infer_argtuple = type_infer_d['types_internal']
            T_type_infer = time.time() - T0_infer

        if verbose or extra_verbose:
            util.print_header('Unit test types:')
            pprint.pprint(types_unit_test)
            print()
            util.print_header('Type inference result:')
            pprint.pprint(types_infer)
            print()
        
        if verbose:
            util.print_header(filename + ' get_types time:' + str(T), pprint.pformat({funcname: list(typesig) for (funcname, typesig) in types_unit_test.items()}))
            util.print_header(filename + ' type_infer time:' + str(T_program_info) + ' ' + str(T_type_infer), pprint.pformat(types_infer))

        for funcname in types_infer_argtuple:
#            print(funcname, types_infer_argtuple[funcname])
            if len(types_infer_argtuple[funcname]):
                d = types_infer_argtuple[funcname]
                d_keys = list(d.keys())
#                print('d:', d)
#                print('d_keys:', d_keys)
                argnames = [argname for (argname, argtype) in d_keys[0]]
#                print('argnames:', argnames)
                if funcname in types_unit_test:
                    for typesig in types_unit_test[funcname]:
                        # Verify that each unit test type signature is a subtype of the corresponding type inference signature
                        typesig_argtuple = tuple([(argname, typesig[argname]) for argname in argnames])
                        type_infer_sig = types_infer_argtuple[funcname][typesig_argtuple]
                        for (varname, unit_test_vartype) in sorted(typesig.items()):
                            if varname in type_infer_sig:
                                infer_vartype = type_infer_sig[varname]
#                                print(type(infer_vartype), varname, infer_vartype, type_infer_sig)
                                assert unit_test_vartype.is_subtype(infer_vartype), ('filename {}, function {} unit test variable {} is not subtype of inferred variable: {} is not a subtype of {}'.format(filename, funcname, varname, unit_test_vartype, infer_vartype))
                                if strict:
                                    assert infer_vartype.is_subtype(unit_test_vartype), ('filename {}, function {} inferred variable {} is not subtype of unit test variable: {} is not a subtype of {}'.format(filename, funcname, varname, infer_vartype, unit_test_vartype))
                                if verbose:
                                    print('*** Success: filename {}, function {} variable {} had unit test type {}, type inference type {}'.format(filename, funcname, varname, unit_test_vartype, infer_vartype))
                            else:
                                if get_macros:
                                    raise ValueError('variable {} did not have its type inferred by type inference'.format(varname))
                                if verbose:
                                    print('*** Warning: variable {} did not have its type inferred by type inference'.format(varname))
    util.print_twocol('test_type_infer:', 'OK')

if __name__ == '__main__':
    test_type_infer_simple()
    test_type_infer()
#    test_type_infer(all_programs=True)
