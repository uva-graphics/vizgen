
import traceback
import util
import hashlib
import importlib
import os
import sys
import copy
import time
import glob
import json
import random
import shutil
import pickle
import numpy
import argparse
import cProfile
import pstats
import operator
import warnings
import pprint

import transforms
import preprocess
import macros
import run_subprocess
import argparse_util
import type_infer
import py_ast

assert sys.version_info[:2] >= (3,0), 'expected Python 3'

verbose                         = False         # Whether this entire module is verbose. Set if --verbose level, where level is >= 2.
default_log_transforms          = True          # Whether to log the sequence of transformed code to a build file
tuner_error_on_fail             = False         # When RunFailed occurs in tuner, raise errors (useful for debugging, should be disabled for checked-in code)
tuner_error_on_fail_verbose     = False         # Extra verbosity to help for debugging with previous option (disable for checked-in code)
tuner_error_on_transform_error  = False         # Similar to tuner_error_on_fail, but also halts for TransformError (disable for checked-in code)
debug_track_memory              = False         # Debug memory leaks (should be disabled for checked-in code)
use_subprocess                  = True          # Use subprocess for run_code to prevent memory leaks
quiet                           = False         # Be extra quiet (low verbosity)
use_static_parallel_analysis    = False         # use static analysis on parallel instead of randomizing loop order
use_ast_parallel_analysis       = True          # use ast version of static analysis
use_type_infer                  = True          # Use type inference instead of obtaining types by unit testing
allow_clean                     = True          # Allow for output files to be cleaned up on successful run
use_prealloc_alias_analysis     = True          # Use alias analysis for preallocation
verify_parallel_compiles        = True          # Verify that each parallel loop compiles

if use_ast_parallel_analysis:
    import ast
    import loop_parallel_ast

if use_static_parallel_analysis:
    import loop_parallel

def is_quiet():
    return quiet

compile_filename_prefix = '_compile_'
default_final_speedup_filename = 'final_speedup.txt'
default_final_time_filename = 'final_ours_time.txt'

if debug_track_memory:
    import pympler.tracker
    
def move_to_sub_if_exists(filename, sub_dir):
    if os.path.exists(filename):
        new_path = os.path.join(os.path.abspath(sub_dir), filename)
        os.rename(os.path.abspath(filename), new_path)
        
def remove_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)

class RunFailed(Exception):
    pass

def c_edit_func(source_c, source_cython):
    """
    Edit C source. Return new C source str, given both Python and C source strs.
    """
    lines = source_c.split('\n')

    # Add OpenMP private clauses:
    #  1. Scan the Cython source for transforms.openmp_add_private_str.
    #  2. Find the corresponding C line (round up if line is missing) by looking up the #line directive.
    #  3. Scan and fix next #pragma omp parallel
    for (cython_lineno, cython_line) in enumerate(source_cython.split('\n')):
        cython_line = cython_line.strip()
        if cython_line.startswith(transforms.openmp_add_private_str):
            var = cython_line.split()[1]
            if verbose:
                util.print_log('c_edit_func: Found on Cython line {} {} {}'.format(cython_lineno, transforms.openmp_add_private_str, var))

            # Find corresponding C line
            c_found_line = 0
            for i in range(len(lines)):
                line_strip = lines[i].strip()
                if line_strip.startswith('#line'):
                    line_strip_split = line_strip.split()
                    if '.pyx' in line_strip_split[2]: #line_strip_split[2].startswith('"' + compile_filename_prefix):
                        pyx_lineno = int(line_strip_split[1])
                        if pyx_lineno >= cython_lineno:
                            c_found_line = i
                            break
            #print('compile_filename_prefix:', compile_filename_prefix)
            if verbose:
                util.print_log('c_edit_func: Corresponding C line {}'.format(c_found_line))
            
            # Scan for next #pragma omp parallel
            for j in range(c_found_line, len(lines)): #range(c_found_line)[::-1]:
                line_j_strip = lines[j].strip()
                if line_j_strip.startswith('#pragma omp parallel'):
                    if verbose:
                        util.print_log('  c_edit_func: replace at line {}'.format(j))
                    private_clause = 'private('
                    try:
                        private = lines[j].index(private_clause)
                    except ValueError:
                        raise ValueError('could not find {} after found C line {} corresponding to Cython line {} containing {}'.format(private_clause, c_found_line, cython_lineno, cython_line))
                    lines[j] = lines[j][:private+len(private_clause)] + '__pyx_v_' + var + ',' + lines[j][private+len(private_clause):]
                    break

    # Hack to replace output error in Cython 0.23.4 (but oddly, this does not occur in Cython 0.22.1).
    for i in range(len(lines)):
        for veclength in transforms.VectorizeInnermost.supported_shapes:
            for vectype in ['float', 'double']:
                lines[i] = lines[i].replace('((v{veclength}{vectype})(({vectype})'.format(**locals()), '((({vectype})'.format(**locals()))

    return '\n'.join(lines)

def insert_initial_run_info(source, is_initial_run):
    return source + """
try:
    import util
    util.is_initial_run = {}
except NameError:
    pass
""".format(is_initial_run)

def run_code(path, source, *args, **kw0):
    """
    Run code, wrapping it in a subprocess.
    """
    T_start = time.time()
    try:
        ans_d = {}
        if verbose:
            print('run_code:', path, '[source omitted]', args, kw0)
        kw = dict(kw0)

        disable_subprocess = False
        if 'use_subprocess' in kw and not kw['use_subprocess']:
            disable_subprocess = True
            del kw['use_subprocess']

        if not use_subprocess or disable_subprocess:
            ans_d = run_code_subprocess(path, source, *args, **kw)
            ans = ans_d['ans']
            return ans

        if 'extra_info' in kw:
            del kw['extra_info']
            kw['extra_info'] = {}
        python_binary = 'python'
        if 'python_binary' in kw:
            python_binary = kw['python_binary']
            del kw['python_binary']

        try:
            ans_d = run_subprocess.run_subprocess(path, """
import compiler
from transforms import *
result = compiler.run_code_subprocess({path!r}, {source!r}, *{args!r}, **{kw!r})
    """.format(**locals()), python_binary)
            extra_info = ans_d['extra_info']
            ans = ans_d['ans']
            if 'extra_info' in kw0:
                kw0['extra_info'].update(extra_info)
    #        print('resulting extra_info:', extra_info.keys() if extra_info is not None else extra_info)
    #        print('resulting kw0:', kw0.keys())

            if ans is None:       # If programmer forgot to return a dict from the test method then default to a dict with 'time' entry
                ans = {'time': ans_d.get('run_time', 0.0)}
    
            return ans
        except:
            raise RunFailed
    finally:
        if transforms.do_profile:
            transforms.profile['run_code'] += time.time()-T_start
#            print(ans.keys())
            if 'compile_time' in ans_d:
                transforms.profile['run_code: compile'] += ans_d['compile_time']
            if 'run_time' in ans_d:
                transforms.profile['run_code: run'] += ans_d['run_time']

def get_verbose():
    return verbose

def run_code_subprocess(path, source, clean=True, verbose=None, cython=True, extra_info=None, clean_on_fail=False, verbose_on_fail=True, is_initial_run=False, repeats=1, once=False, out_filename=None, override_input_image=None, override_n_tests=None, override_args=None, override_kw_args=None, use_4channel=False, vectorize=True, run_only=False, compile_c=True, temp_dir = None):
    """
    Run code without wrapping in a subprocess.
    
    Given source code string for a module, Cython-ize it (or use Python if cython is False), change directory to 'path', and report the result of module.test().
    
    On failure raises RunFailed exception.
    """
    if get_verbose():
        print('run_code_subprocess:', path, '[source omitted]', clean, verbose, cython, extra_info, clean_on_fail, verbose_on_fail, is_initial_run, repeats, once, out_filename, override_input_image, override_n_tests, override_args, override_kw_args)
    
    if verbose is None:
        verbose = get_verbose()
#    source = insert_initial_run_info(source, is_initial_run)
    util.is_initial_run = is_initial_run
    util.override_input_image = override_input_image
    util.override_n_tests = override_n_tests
    util.use_4channel = use_4channel
    if override_args is None:
        override_args = ()
    if override_kw_args is None:
        override_kw_args = {}
    compile_time = 0.0
    run_time = 0.0
    
    prefix = compile_filename_prefix + hashlib.md5(pickle.dumps(repr((source, cython)))).hexdigest()
    orig = os.getcwd()
    old_path = list(sys.path)
    success = False
    so_filename = None
    if verbose:
        util.print_header('run_code() source:', source)
    try:
        try:
            with util.SuppressOutput(verbose, verbose_on_fail):
                if cython:
                    with open(prefix + '.pyx', 'wt') as f:
                        f.write(source)
                    if out_filename is not None:
                        with open(out_filename, 'wt') as f:
                            f.write(source)
                    T_start_compile = time.time()
                    util.compile_cython_single_file(prefix, c_edit_func, extra_info, vectorize, compile_c)
                    compile_time = time.time() - T_start_compile
                else:
                    with open(prefix + '.py', 'wt') as f:
                        f.write(transforms.python_headers + source)
                    assert os.path.exists(prefix + '.py'), 'file missing:' + prefix + '.py'
                if run_only:
                    success = True
                    return {'compile_time': compile_time, 'extra_info': extra_info, 'ans': {}, 'run_time': 0.0}
                os.chdir(path)
                sys.path.append(path)
                T_start_run = time.time()
                importlib.invalidate_caches()           # If we do not clear the caches then module will sometimes not be found
                mod = importlib.import_module(prefix)
                so_filename = importlib.find_loader(prefix).path
                
                try:
                    for j_repeat in range(repeats):
                        ans = mod.test(*override_args, **override_kw_args)
                finally:
                    run_time = time.time() - T_start_run
                success = True
                if verbose:
                    print('got result of test:', ans)
        except (SystemExit, Exception) as e:
            if verbose:
                print('caught exception')
            raise RunFailed(str(e))
    finally:
        sys.path = old_path
        os.chdir(orig)
        if ((success and clean) or ((not success) and clean_on_fail)) and allow_clean:
            if cython:
                remove_if_exists(prefix + '.pyx')
                remove_if_exists(prefix + '.c')
                remove_if_exists(prefix + '.so')
                remove_if_exists(prefix + '.html')
                if so_filename is not None:
                    remove_if_exists(so_filename)
            else:
                remove_if_exists(prefix + '.py')
        elif temp_dir is not None:
            if cython:
                move_to_sub_if_exists(prefix + '.pyx', temp_dir)
                move_to_sub_if_exists(prefix + '.c', temp_dir)
                move_to_sub_if_exists(prefix + '.so', temp_dir)
                move_to_sub_if_exists(prefix + '.html', temp_dir)
                if so_filename is not None:
                    move_to_sub_if_exists(prefix + '.so', temp_dir)
            else:
                move_to_sub_if_exists(prefix + '.py', temp_dir)
                

    if extra_info is not None:
        extra_info['source'] = source
    return {'ans': ans, 'extra_info': extra_info, 'compile_time': compile_time, 'run_time': run_time}

def get_runner(path, **kw):
    """
    Partial application of path to run_code(). Return function 'run' so run(source) => result.
    """
    def run(source):
        return run_code(path, source, **kw)
    return run

def get_types_py_ast(path, source, is_verbose=False, use_4channel=False):
    """
    Get types used by all functions.
    
    Modifies source so each defined function tracks return types before it returns.
    Returns dict with run info. The key 'types' in the run info dict stores a dict mapping
    function name to a list of type configurations used for each function, where a type
    configuration is a dict mapping local variable name => type.
    """
    rootnode = py_ast.get_ast(source)
    py_ast.add_parent_info(rootnode)
    macroL = macros.find_macros_ast(rootnode, macros.macros, True)
    
    # Tag RedBaron function call and macro nodes
    defnode_id_to_macros = {}
    macro_id_to_defnode = {}
    for macro in macroL:
        for (i, arg) in enumerate(macro.arg_nodes):
            parent = macro.node.parent
            parent_nodes = []
            has_defnode = False
            while parent is not None:
                parent_nodes.append(parent)
                if isinstance(parent, ast.FunctionDef):
                    if i == 0:
                        defnode_id_to_macros.setdefault(id(parent), []).append(macro)
                        macro_id_to_defnode[id(macro)] = parent
                    has_defnode = True
                parent = parent.parent
            if (not any([isinstance(parent_node, ast.FunctionDef) and util.is_test_funcname(parent_node.name)
                         for parent_node in parent_nodes]) and has_defnode):
                new_arg = py_ast.get_ast('_log_argtype_value(' + str(id(arg)) + ', ' + py_ast.dump_ast(arg) + ')').body[0].value
                new_arg.args[0].n = id(new_arg)
                py_ast.replace_node(rootnode, arg, new_arg)
    
    defnodes = py_ast.find_all(rootnode, ast.FunctionDef)

    def get_arg_names(node):
        return [_arg.arg for _arg in node.args.args]

    func_name_to_argnames = {}

    names = []
    for node in defnodes:
        if not util.is_test_funcname(node.name):
            names.append(node.name)
#            node.value = 'try:\n' + node.value.dumps() + '\nfinally:\n    global {}_locals\n    try:\n        {}_locals\n    except:\n        {}_locals = set()\n    print("locals var:", {}_locals)\n    {}_locals.add(frozenset([(_k, util.get_cython_type(_v)) for (_k, _v) in locals().items()]))\n'.format(node.name, node.name, node.name, node.name, node.name)
            name = node.name
            node_str = ''
            for body_node in node.body:
                node_str +='\n' + py_ast.dump_ast(body_node)
            lines = node_str.split('\n')
            node_str = '\n'.join(4 * ' ' + line for line in lines)
            
            decode_argtypes_L = []
            for macro in defnode_id_to_macros.get(id(node), []):
#                print('***', macro.node_str)
                id0 = id(macro.arg_nodes[0])
                arg_ids = [id(arg) for arg in macro.arg_nodes]
                tuple_str = '(' + ','.join(['_argtype_values[{}]'.format(_id) for _id in arg_ids]) + (',' if len(arg_ids) == 1 else '') + ')'
                macro_key = util.types_non_variable_prefix + macro.node_str
                decode_argtypes_L.append(
"""
    if {id0} in _argtype_values:
        try:
            _locals_typeconfig[{macro_key!r}] = util.CythonType.from_value({tuple_str}, None)
        except NotImplementedError:
            pass
""".lstrip('\n').format(**locals()))

            decode_argtypes = '\n'.join(decode_argtypes_L)
            
            arg_names = get_arg_names(node)
            func_name_to_argnames[name] = arg_names
            
            node_body_str = \
"""_argtype_values = {{}}
def _log_argtype_value(id_num, v):
    try:
        if type(v) == type(_argtype_values[id_num]):
            return v
    except KeyError:
        _argtype_values[id_num] = v
        return v
    _argtype_values[id_num] = util.promote_numeric(_argtype_values[id_num], v)
    return v
try:
{node_str}
finally:
    global _{name}_locals
    try:
        _{name}_locals
    except:
        _{name}_locals = util.TypeSignatureSet({arg_names})
    _ignore_names = ['_log_argtype_value', '_argtype_values', '_ignore_names']
    _locals_typeconfig = dict([(_k, util.CythonType.from_value(_v, None, error_variable=_k)) for (_k, _v) in locals().items() if _k not in _ignore_names])
{decode_argtypes}
    _{name}_locals.add(_locals_typeconfig)
""".format(**locals())
            node.body = py_ast.get_ast(node_body_str).body

    for node in defnodes:
        if node.name == 'test':
            globalL = ['_{}_locals'.format(k) for k in names] # + ['_{}_argtypes'.format(k) for k in names]
            typesL_updateL = []
            for k in names:
                locals_name = '_{}_locals'.format(k)
                arg_names = func_name_to_argnames[k]
#                typesL_updateL.append('print(_{}_locals)'.format(k))
                typesL_updateL.append('if "{locals_name}" in globals(): _typesL_var["{k}"] = util.TypeSignatureSet({arg_names}, [{{_k: _v for (_k, _v) in _typeconfig.items() if isinstance(_k, str)}} for _typeconfig in {locals_name}])'.format(**locals()))
#            types_dL = ["'{}': _{}_locals".format(k, k) for k in names]
#            node.value = 'try:\n' + node.value.dumps() + '\nfinally:\n    global {}\n    ans = {{{}}}\nreturn ans'.format(','.join(globalL), ','.join(dL))
            node_str = ''
            for body_node in node.body:
                 node_str += '\n' + py_ast.dump_ast(body_node)
            lines = node_str.split('\n')
            node_str = '\n'.join(8 * ' ' + line for line in lines) 
            globalL_str = ','.join(globalL)
            typesL_update_str = '\n        '.join(typesL_updateL)
            
            node_body_str = \
"""_exc = None
try:
    def inner_func():
        {node_str}
    ans = inner_func()
except Exception as _e:
    _exc = _e
finally:
    if _exc is not None:
        raise _exc
    else:
        global {globalL_str}
        _typesL_var = {{}}
        {typesL_update_str}
        return {{"types": _typesL_var, "test_result": ans}}
""".format(**locals())
            node.body = py_ast.get_ast(node_body_str).body

#        _typesL_var = {{_k: _v for (_k, _v) in _typesL_var.items() if isinstance(_k, str)}}

    s = py_ast.dump_ast(rootnode)
    
    if verbose:
        print(s)
    result = run_code(path, s, cython=False, verbose=verbose or is_verbose, clean=True, is_initial_run=True, use_4channel=use_4channel)
    if verbose:
        print(result)

    return result

def add_transforms_to_code(program_info, s, s_orig, transformL):
    s = transforms.unparse_transforms(program_info, s_orig, transformL, False)
    transformL = transforms.parse_transforms(program_info, s)
    return (s, transformL)

def add_extra_cython_headers(program_info, s):
    headers = '\n' + '\n\n'.join(value for value in program_info.cython_headers.values()) + '\n'
    float32_switch = ''
    if program_info.is_rewrite_float32():
        float32_switch = 'util.default_test_dtype = "float32"'
    use_4channel_switch = ''
    if program_info.is_use_4channel():
        use_4channel_switch = 'util.use_4channel = True'
    return transforms.cython_imports + '\n' + headers + '\n' + float32_switch + '\n' + use_4channel_switch + '\n' + s

def apply_transforms_and_finalize(program_info, s, extra_info=None):
    if program_info.log_transforms:
        log_filename = os.path.join(program_info.stats_dir, 'transforms.txt')
        f = open(log_filename, 'wt')
    
    applied_macros = False
    preallocated = False
    
    transform_stats = {transform_cls.__name__: 0 for transform_cls in transforms.orig_all_transforms}    # Count number of times each transform was applied
    macro_extra_info = {}
    for transform in program_info.transformL:
        transform_stats.setdefault(transform.__class__.__name__, 0)
        transform_stats[transform.__class__.__name__] += 1
    if verbose:
        util.print_header('apply_transforms_and_finalize input program:', s)
    
    while True:
        start_time = time.time()
        #transformL = transforms.parse_transforms(program_info, s, apply_order=True)
        if transforms.do_profile:
            transforms.profile['apply_transforms_and_finalize: parse_transforms'] += time.time() - start_time
        
        # Apply preallocation first
        if not preallocated and transforms.enable_preallocate and program_info.preallocate:
            preallocated = True
            preallocate_extra_info = {}
            start_time = time.time()
            s = transforms.Preallocate(program_info).apply_auto_cache(s, extra_info=preallocate_extra_info)
            if transforms.do_profile:
                transforms.profile['apply_transforms_and_finalize: Preallocate'] += time.time() - start_time

            transform_stats['Preallocate'] = preallocate_extra_info['preallocate_count']
        
        transformL = transforms.parse_transforms(program_info, s, apply_order=True)
        # Apply macros immediately after type specialization
        if all([not isinstance(transform, transforms.TypeSpecialize) for transform in transformL]) and not applied_macros and program_info.apply_macros:
            applied_macros = True
            start_time = time.time()
            s = transforms.ApplyMacros(program_info).apply_auto_cache(s, extra_info=macro_extra_info)
            if transforms.do_profile:
                transforms.profile['apply_transforms_and_finalize: ApplyMacros'] += time.time() - start_time
            transform_stats['ApplyMacros'] = macro_extra_info['macro_count']
        
        if len(transformL) == 0:
            break

        T0 = transformL[0]
        start_time = time.time()
        s = transformL[0].apply_and_delete(s)
        if transforms.do_profile:
            transforms.profile['apply_transforms_and_finalize: {}'.format(T0.__class__.__name__)] += time.time() - start_time

        if program_info.log_transforms:
            util.print_header('Result after transform {}'.format(T0), s, file=f, linenos=True)

        if verbose:
            util.print_header('Result after transform {}'.format(T0), s)
#            util.print_header('Cython code after transform {}'.format(T0), transforms.finalize_cython(program_info, s))

    if extra_info is not None:
        extra_info['transform_stats'] = transform_stats
    start_time = time.time()
    s = transforms.finalize_cython(program_info, s)
    if transforms.do_profile:
        transforms.profile['apply_transforms_and_finalize: finalize_cython'] += time.time() - start_time
    start_time = time.time()
    s = add_extra_cython_headers(program_info, s)
    if transforms.do_profile:
        transforms.profile['apply_transforms_and_finalize: add_extra_cython_headers'] += time.time() - start_time
    return s

class ConsistencyFailed(Exception):
    pass

def str_list(L):
    return '[' + ', '.join(str(x) + ' original line number is ' + str(x.orig_num) for x in L) + ']'

class ProgramInfo:
    default_max_iters = 40
    default_compile_dir = 'out/result'
    default_final_subdir = 'final'          # Final tuned program
    default_stats_subdir = 'stats'
    
    def __deepcopy__(self, memo):           # Avoid duplicating ProgramInfo when copy.deepcopy() is used on data structures that have a ProgramInfo instance
        return self
    
    def program_prefix(self, i):
        return os.path.join(self.gen_prefix, 'program{:05d}'.format(i))
    
    def __init__(self, s_orig, path='./', types={}, transformL=None, log_transforms=False, compile_dir=None, is_verbose=False, filename=None, s_current=None, max_types=None, apply_macros=True, final_subdir=default_final_subdir, verbose_level=1, out_file=None, preallocate_arrays=None, preallocate=True, preallocate_verbose=True, full_filename=None, quiet=False, use_4channel=False, safe=False):
        # Set parallel analysis dict
        self.parallel_mapping = {}
        
        """
        Track program information: original source s_orig, path containing program, types (if None, will discover types), list of transforms.
        """
        assert not '\t' in s_orig, 'compiler input must currently use only spaces (no tabs allowed)'
        self.init_time = time.time()
        #self.test_result = None
        if transformL is None:
            transformL = []

        # Immediately set up all attributes that util.CythonType might access (type_infer.type_infer() creates CythonType instances)
        self.safe = safe
        self.path = path
        self.s_orig = s_orig
        self.transformL = transformL      # Transform list before dependency resolution
        self.cython_headers = {}          # Maps cache key (can be anything hashable, but usually e.g. a class name) to
                                          # a Cython header that should be inserted associated with that key.

        # Get types
        if types is None:
            start_get_types = time.time()
            if not use_type_infer:
                types = get_types(path, s_orig, is_verbose, use_4channel)['types']
            else:
#                types_discard = get_types(path, s_orig, is_verbose, use_4channel)['types']
                self.type_infer = type_infer.type_infer(self, verbose=False, get_macros=True)
                types = self.type_infer['types']
                try:
                    self.type_infer_4channel = type_infer.type_infer(self, verbose=False, get_macros=True, use_4channel=True)
                    has_4channel = True
                except:
                    has_4channel = False
                if verbose:
                    util.print_header('Preallocation from alias analysis alone:')
                    pprint.pprint(self.type_infer['prealloc'])
                if has_4channel:
                    types_4channel = self.type_infer_4channel['types']
    #                util.print_header('types_4channel:')
    #                pprint.pprint(types_4channel)
                    for funcname in types:
                        for i in range(len(types[funcname])):
                            for varname in types[funcname][i]:
                                type_4channel = types_4channel[funcname][i][varname]
    #                            print('{}.{} shape (ordinary) {} => shape (4-channel) {}'.format(funcname, varname, types[funcname][i][varname].shape, type_4channel.shape))
                                types[funcname][i][varname].bind_4channel_type(type_4channel)
            if transforms.do_profile:
                transforms.profile['get_types'] += time.time() - start_get_types
    #            self.test_result = type_info['test_result']
            
        self.s_current = None             # Most recent or "current" source after previous code transformations. Set by parse_transforms().
        self.types = types
        self.transformL_deps = None       # Transform list after dependency resolution
        self.log_transforms = log_transforms
        self.filename = os.path.abspath(filename) if filename is not None else None
        self.initial_run_info = None
        self.max_types = max_types
        self.apply_macros = apply_macros
        self.final_subdir = final_subdir
        self.verbose_level = verbose_level
        self.run_count = 0
        self.run_time = 0.0
        self.out_file = out_file
        self.quiet = quiet
        self.full_filename = os.path.abspath(full_filename) if full_filename is not None else None
        
        """if transforms.accept_annotated_type:
            rootnode = py_ast.get_ast(self.s_orig)
            defnodes = py_ast.find_all(rootnode, ast.FunctionDef)
            for defnode in defnodes:
                type_to_add = {}
                if defnode.name in types:
                    if isinstance(defnode.returns, ast.Name):
                        try:
                            type_to_add[defnode.returns.id] = util.CythonType.from_cython_type(defnode.returns.id, self)
                        except:
                            pass
                    for arg in defnode.args.args:
                        if isinstance(arg.annotation, ast.Name):
                            try:
                                type_to_add[arg.arg] = util.CythonType.from_cython_type(arg.annotation.id, self)
                            except:
                                pass
                    
                    defnode_dump = py_ast.dump_ast(defnode)
                    
                    lines = defnode_dump.split('\n')
                    
                    for line in lines:
                        if '# type:' in line:
                            try:
                                annotated_nodes = py_ast.get_ast(line)
                                annotated_type_str = py_ast.dump_ast(annotated_nodes[1]).strip().lstrip('# type:').strip()
                                annotated_type = util.CythonType.from_cyton_type(annotated_type_str, self)
                                if isinstance(annotated_nodes[0], ast.Assign) and len(annotated_nodes[0].targets) == 0 and isinstance(annotated_nodes[0].targets[0], ast.Name):
                                    type_to_add[annotated_nodes[0].targets[0].id] = annotated_type
                                elif isinstance(annotated_nodes[0], ast.For) and isinstance(annotated_nodes[0].target, ast.Name):
                                    type_to_add[annotated_nodes[0].target.id] = annotated_type
                            except:
                                pass
                    
                    for i in ranage(len(types[defnode.name])):
                        local_type = types[defnode.name][i]
                        for name_to_add in type_to_add.keys():
                            if name_to_add not in local_type:
                                local_type[name_to_add] = type_to_add[name_to_add]
                            elif local_type[name_to_add].is_object():
                                local_type[name_to_add] = type_to_add[name_to_add]"""
        
        if transforms.annotate_type_signature:
            rootnode = py_ast.get_ast(self.s_orig)
            defnodes = py_ast.find_all(rootnode, ast.FunctionDef)
            for key in types.keys():
                annotate_defnode = [dnode for dnode in defnodes if dnode.name == key][-1]
                if len(types[key]):
                    current_types = types[key][0]
                
                    def_flag = False
                
                    for arg in annotate_defnode.args.args:
                        arg_name = py_ast.dump_ast(arg)
                        if arg_name in current_types:
                            arg_type = transforms.unparse_type(current_types[arg_name]._cython_nickname)
                            if arg_type == 'object':
                                def_flag = True
                            arg.annotation = py_ast.get_ast(arg_type).body[0].value
                        else:
                            def_flag = True
                
                    if '_return_value' in current_types:
                        arg_type = transforms.unparse_type(current_types['_return_value']._cython_nickname)
                        if arg_type == 'object':
                            def_flag = True
                        annotate_defnode.returns = py_ast.get_ast(arg_type).body[0].value
                    else:
                        def_flag = True
                
                    if def_flag:
                        annotate_defnode.name = annotate_defnode.name + transforms.unknown_type_prefix
                
                    for assignnode in py_ast.find_all(annotate_defnode, ast.Assign):
                        if len(assignnode.targets) == 1 and isinstance(assignnode.targets[0], ast.Name):
                            if assignnode.targets[0].id in current_types:
                                arg_type = transforms.unparse_type(current_types[assignnode.targets[0].id]._cython_nickname)
                                if arg_type == 'object':
                                    annotate_str = '# type: ' + arg_type + transforms.unknown_type_prefix
                                else:
                                    annotate_str = '# type: ' + arg_type
                                py_ast.add_after_node(annotate_defnode, assignnode, py_ast.get_ast(annotate_str).body[0])
                            else:
                                assignnode.targets[0].id = assignnode.targets[0].id + transforms.unknown_type_prefix
                
                    for fornode in py_ast.find_all(annotate_defnode, ast.For):
                        if isinstance(fornode.target, ast.Name):
                            if fornode.target.id in current_types:
                                arg_type = transforms.unparse_type(current_types[fornode.target.id]._cython_nickname)
                                if arg_type == 'object':
                                    annotate_str = '# type: ' + arg_type + transforms.unknown_type_prefix
                                else:
                                    annotate_str = '# type: ' + arg_type
                                py_ast.add_before_node(annotate_defnode, fornode.body[0], py_ast.get_ast(annotate_str).body[0])
                            else:
                                fornode.target.id = fornode.target.id + transforms.unknown_type_prefix
                
                    if annotate_defnode.returns == None:
                        return_nodes = py_ast.find_all(annotate_defnode, ast.Return)
                        if len(return_nodes) == 0:
                            annotate_defnode.returns = 'None'
        
                self.type_annotation_s = py_ast.dump_ast(rootnode)
                
                lines = self.type_annotation_s.split('\n')
                
                html_str = "<html>\n<head>\n<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">\n<link rel=\"stylesheet\" type=\"text/css\" href=\"html.css\">\n</head>\n<body>\n<h2>test</h2>\n<table>\n<caption>test.py</caption>\n<tbody><tr>\n<td class=\"table-code\"><pre>"
                
                html_lines = ""
                
                for i in range(len(lines)):
                    html_str += "<span id=\"L" + str(i) + "\" class=\"lineno\"><a class=\"lineno\" href=\"#L" + str(i) + "\">" + str(i) + "</a></span>\n"
                    
                    if lines[i].strip() == '':
                        class_name = "\"line-empty\""
                    elif transforms.unknown_type_prefix in lines[i]:
                        class_name = "\"line-any\""
                        lines[i] = lines[i].replace(transforms.unknown_type_prefix, '')
                    else:
                        class_name = "\"line-precise\""
                    html_lines += "<span class=" + class_name + ">" + lines[i] + "</span>\n"
                    
                final_html = html_str + "</pre></td>\n<td class=\"table-code\"><pre>" + html_lines
                final_html += "</pre></td></tr></tbody></table></body></html>"
                self.html = final_html
                
                #file = open('example.html', 'w+')
                #file.write(final_html)
                #file.close()
                
                self.type_annotation_s = '\n'.join(lines)
        
        
            
            #file=open('example.py', 'w+')
            #file.write(self.type_annotation_s)
            #file.close()
                
        
        if self.out_file is not None:
            self.out_file = os.path.abspath(self.out_file)
        
        if compile_dir is None:
            if self.filename is not None:
                appname = os.path.splitext(os.path.split(self.filename)[1])[0]
                self.compile_dir = os.path.abspath(os.path.join('out', appname))
            else:
                self.compile_dir = os.path.abspath(self.default_compile_dir)
        else:
            self.compile_dir = os.path.abspath(compile_dir)
            
        self.temp_compile_dir = os.path.join(self.compile_dir, 'compile')

        self.stats_dir = os.path.abspath(os.path.join(self.compile_dir, self.default_stats_subdir))

        if not os.path.isdir(self.stats_dir):
            os.makedirs(self.stats_dir)

        self.gen_prefix = os.path.abspath(os.path.join(self.compile_dir, 'gen'))

        if not os.path.isdir(self.gen_prefix):
            os.makedirs(self.gen_prefix)

        self.final_dir = os.path.join(self.compile_dir, self.final_subdir)

        if not os.path.isdir(self.final_dir):
            os.makedirs(self.final_dir)

        self.final_pyx_filename = os.path.join(self.final_dir, 'program.pyx')
        self.final_py_filename = os.path.join(self.final_dir, 'program.py')
        self.final_html_filename = os.path.join(self.final_dir, 'program.html')

        self.output_filename = os.path.join(self.stats_dir, 'output.txt')
        self.speedup_filename = os.path.join(self.stats_dir, 'speedup.csv')
        self.final_py_time_filename = os.path.join(self.stats_dir, 'final_py_time.txt')
        self.final_ours_time_filename = os.path.join(self.stats_dir, default_final_time_filename)
        self.final_speedup_filename = os.path.join(self.stats_dir, default_final_speedup_filename)
        self.final_transforms_filename = os.path.join(self.stats_dir, 'final_transforms.txt')
        self.final_transform_stats_filename = os.path.join(self.stats_dir, 'final_transform_stats.txt')
        self.run_time_filename = os.path.join(self.stats_dir, 'run_time.txt')
        self.run_count_filename = os.path.join(self.stats_dir, 'run_count.txt')
        self.tune_filename = os.path.join(self.stats_dir, 'tune_filename.txt')

        self.validate_speedups_after_filename = os.path.join(self.stats_dir, 'validate_speedups_after.txt')
        
        if os.path.exists(self.compile_dir):
            pass        # Do not remove output files because it could cause problems for --no-tune mode.
#            for filename in [self.output_filename, self.speedup_filename, self.final_speedup_filename, self.final_transforms_filename, self.final_transform_stats_filename, self.run_time_filename, self.run_count_filename, self.final_py_time_filename, self.final_ours_time_filename]:
#                if os.path.exists(filename):
#                    os.remove(filename)
#            for filename in glob.glob(os.path.join(self.compile_dir, 'program*.pyx')):
#                os.remove(filename)
#            for filename in glob.glob(os.path.join(self.compile_dir, 'program*.html')):
#                os.remove(filename)
#            for filename in glob.glob(os.path.join(self.compile_dir, 'program*.py')):
#                os.remove(filename)
        else:
            os.makedirs(self.compile_dir)
            
        if not os.path.exists(self.temp_compile_dir):
            os.makedirs(self.temp_compile_dir)
        
        self.preallocate = preallocate
        self.preallocate_verbose = preallocate_verbose
        if transforms.enable_preallocate and preallocate:
            self.preallocate_arrays = preallocate_arrays if preallocate_arrays is not None else transforms.preallocate_find_arrays(self, self.s_orig)
        else:
            self.preallocate_arrays = None

        self.log_output_file = open(self.output_filename, 'wt')
        if transforms.do_profile:
            transforms.profile['ProgramInfo.init'] = time.time() - self.init_time

    def parallel_cache(self, ptransform, ptransform_node):
        """
        searches in cache to see whether ptransform is stored
        if not, compute parallel result and store in cache
        finally, return parallel result
        """
        if ptransform in self.parallel_mapping:
            return self.parallel_mapping[ptransform]
        success = loop_parallel_ast.check_loop_parallel(ptransform_node)
        self.parallel_mapping[repr(ptransform)] = success
        return success
    
    def arraystorage_cache_key(self):
        return (self.is_rewrite_float32(), self.is_use_4channel())

    def is_rewrite_float32(self):
        """
        Return whether an ArrayStorage transform is used such that float arrays are rewritten to float32 format.
        """
        return any(getattr(t, 'use_float32', False) for t in self.transformL)

    def is_use_4channel(self):
        """
        Return whether an ArrayStorage transform is used such that float arrays are rewritten to have 4 channels.
        """
        return any(getattr(t, 'use_4channel', False) for t in self.transformL)
    
    def resolve_dependencies(self, *args, **kw):
        start_time_resolve_dependencies = time.time()
        try:
            self.transformL = transforms.resolve_dependencies(self.transformL, *args, **kw)
        finally:
            if transforms.do_profile:
                transforms.profile['ProgramInfo: tune: resolve dependencies'] += time.time() - start_time_resolve_dependencies
    
    def check_consistency(self):
        """
        Calls is_consistent() on each transform, and if any returns False then raise ConsistencyFailed.
        """
        for transform in self.transformL:
            if not transform.is_consistent():
                raise ConsistencyFailed('not consistent')
    
    def run(self, *args, **kw):
        """
        Apply program transforms and return the result of run_code(*args, **kw).
        """
        T_start = time.time()
        self.check_consistency()
        if verbose:
            util.print_header('ProgramInfo.run, self.transformL:', repr(self.transformL))
        (s, transformL) = add_transforms_to_code(self, self.s_orig, self.s_orig, self.transformL)
        if verbose or (self.verbose_level == 1 and kw.get('once', False)) or kw.get('print_source', False):
            util.print_header('Program source after adding transforms to code:', s)
        if 'print_source' in kw:
            del kw['print_source']
        if verbose:
            util.print_header('compiler, transformL after add_transforms_to_code:', transformL)
        if 'extra_info' in kw:
            kw['extra_info']['transformL'] = transformL
            kw['extra_info']['source_add_transforms'] = s
        T_start_apply = time.time()
        s = apply_transforms_and_finalize(self, s, extra_info=(kw['extra_info'] if 'extra_info' in kw else None))
        if transforms.do_profile:
            transforms.profile['apply_transforms_and_finalize'] += time.time() - T_start_apply
        self.run_count += 1
        ans = run_code(self.path, s, *args, **kw)
        if 'total_time' in ans:
            self.run_time += ans['total_time']
        if transforms.do_profile:
            transforms.profile['ProgramInfo.run'] += time.time()-T_start
        return ans

    def get_transforms_from_source(self):
        """
        Read annotated transforms from original source code into self.transformL.
        """
        self.transformL = transforms.parse_transforms(self, self.s_orig)
        
    def add_new_transform(self, v):
        """
        Helper function to add a new transform to transformL. If it is a class, makes a new instance. If it is an instance, adds that instance.
        """
        if isinstance(v, type(ProgramInfo)):
            self.transformL.append(v(self))
        else:
            self.transformL.append(v)

    def get_initial_run_info(self, run_args=(), run_kw_args={}):
        """
        """
        if self.initial_run_info is not None:
            return self.initial_run_info
        self.run_count += 1
        run_kw_args = dict(run_kw_args)
        run_kw_args['cython'] = False
        self.initial_run_info = run_code(self.path, transforms.python_headers + '\n' + self.s_orig, *run_args, **run_kw_args)
        if 'total_time' in self.initial_run_info:
            self.run_time += self.initial_run_info['total_time']
        return self.initial_run_info

    def get_initial_guesses(self, run_args, run_kw_args, allow_array_storage=True):
        """
        Get a list of transform lists, each of which is a reasonable initial guess for tuning. These are used in the first tuner iterations.
        
        Returns (initial_guess_L, likely_fastest_guess), where likely_fastest_guess is the transform list that is likely the fastest.
        """
        start_time_get_initial = time.time()
        old_transformL = list(self.transformL)
        parallel_transformL = []
        parallel_init = False
        ans = []
        ans.append([])
    
        for checks in [False, True]:
            for use_parallel in [True, False]:
                if use_parallel and not transforms.Parallel in transforms.all_transforms:
                    continue
                #for use_array_storage in ([False, True] if allow_array_storage else [False]):
                for i_use_array_storage in [2, 1, 0]: #range(3):
                    if i_use_array_storage >= 1 and not transforms.ArrayStorage in transforms.all_transforms:
                        continue
#                    print('checks={}, use_parallel={}, use_array_storage={}'.format(checks, use_parallel, use_array_storage))
                    # TypeSpecialize all functions with checks=checks.
                    self.transformL = []                                    # Initialize the list to avoid confusing any mutate() methods
                    self.transformL = [transforms.TypeSpecialize(self)]
                    self.transformL[0].mutate(set_all=True, checks=checks)
                    typespecialize_L = copy.deepcopy(self.transformL)
                    if i_use_array_storage == 1:
                        self.transformL.append(transforms.ArrayStorage(self, None, True, False))
                    elif i_use_array_storage == 2:
                        self.transformL.append(transforms.ArrayStorage(self, None, True, True))
                    
                    if use_parallel:
                        if not parallel_init:
                            parallel_init = True
                            
                            # Parallelize all outer for loops
                            (lines, nodes) = self.transformL[0].get_line_for_mutation(ast.For, get_all=True)
                            #print('lines: {}'.format(lines))
                            line_to_node = {}
                            
                            parallel_prefilteredL = []
                            for i in range(len(lines)):
                                line = lines[i]
                                node = nodes[i]
                                line_to_node[line] = node
                                parent_list = []
                                parent = node.parent
                                
                                while parent is not None:
                                    parent_list.append(parent)
                                    parent = parent.parent
                                
                                ok = not any([isinstance(parent1, ast.For) for parent1 in parent_list])
                                if ok:
                                    parallel_prefilteredL.append(transforms.Parallel(self, line))
                    
                            # For each for loop, verify whether it can be parallelized safely
                            while len(parallel_prefilteredL):
                                if not quiet:
                                    print('Program analysis: checking parallelism of loops ({}+ remain)'.format(len(parallel_prefilteredL)))
                                
                                if use_ast_parallel_analysis:
                                    ptransform = parallel_prefilteredL.pop()
                                    ptransform_node = line_to_node[ptransform.line]
                                    success = loop_parallel_ast.check_loop_parallel(ptransform_node)
                                    self.parallel_mapping[repr(ptransform)] = success
                                
                                else:
                                    ptransform = parallel_prefilteredL.pop()
                                    #print('parallel_prefilteredL: {}, parallel_transformL: {}, ptransform: {}'.format(parallel_prefilteredL, parallel_transformL, ptransform))
                                    ptransform_randomized = copy.deepcopy(ptransform)
                                    ptransform_randomized.randomize = True
                                
                                    transformL_copy = self.transformL
                                    self.transformL = self.transformL + [ptransform_randomized]
                                    #print('transformL before resolve dependencies: {}'.format(self.transformL))
                                    self.resolve_dependencies(randomize=False)
                                    #print('transformL after resolve dependencies: {}'.format(self.transformL))
                                    run_kw_args_p = dict(run_kw_args)
                                    run_kw_args_p['use_subprocess'] = False
                                    try:
                                        run_info = self.run(*run_args, **run_kw_args_p)
                                        success = True
                                    except (RunFailed, ConsistencyFailed, transforms.TransformError):
                                        success = False
                                        
                                    self.transformL = transformL_copy
                                        
                                if success:
                                    parallel_transformL.append(ptransform)
                                else:
                                    # For each for loop nested immediately inside the current one, add it to the list
                                    ptransform_node = line_to_node[ptransform.line]
                                    for i in range(len(lines)):
                                        line = lines[i]
                                        node = nodes[i]
                                        ok = False
                                        parent = node.parent
                                        while parent is not None:
                                            if isinstance(parent, ast.For):
                                                if parent is ptransform_node:
                                                    ok = True
                                                    break
                                                else:
                                                    ok = False
                                                    break
                                            parent = parent.parent
                                        
                                        #print('line={}, node={}, ptransform_node={}, ok={}'.format(line, node, ptransform_node, ok))
                                        if ok:
                                            parallel_prefilteredL.append(transforms.Parallel(self, line))
                                  
                                #self.transformL = transformL_copy

                            if verify_parallel_compiles:
                                parallel_transformL_final = []
                                for (itransform, ptransform) in enumerate(parallel_transformL):
                                    if not quiet:
                                        print('Program analysis: checking whether parallelized loop {}/{} can be Cythonized'.format(itransform, len(parallel_transformL)))
                                    transformL_copy = self.transformL
                                    self.transformL = typespecialize_L + [ptransform]
                                    
                                    self.resolve_dependencies(randomize=False)
                                    run_kw_args_p = dict(run_kw_args)
                                    run_kw_args_p['use_subprocess'] = False
                                    run_kw_args_p['run_only'] = True
                                    run_kw_args_p['compile_c'] = False
                                    try:
                                        run_info = self.run(*run_args, **run_kw_args_p)
                                        success = True
                                    except (RunFailed, ConsistencyFailed, transforms.TransformError):
                                        success = False
                                    self.transformL = transformL_copy
                                    
                                    if success:
                                        parallel_transformL_final.append(ptransform)
                            else:
                                parallel_transformL_final = parallel_transformL

                        self.transformL.extend(parallel_transformL_final)

                    ans.append(copy.deepcopy(self.transformL))

        for sub in ans:
            util.print_header('Initial guess:', str_list(sub))

        self.transformL = old_transformL

        if transforms.do_profile:
            transforms.profile['ProgramInfo.get_initial_guesses'] += time.time()-start_time_get_initial

#        if len(ans) >= 4:
#            return (ans, ans[3])
#        else:
        return (ans, ans[1])

    def print_log(self, s):
        print(s, file=self.log_output_file)
        print(s)
        self.log_output_file.flush()


    def get_allow_transforms(self, stats_filename=None, s_annotated=None):
        if stats_filename is None:
            stats_filename = self.final_transform_stats_filename
        
        stats = json.loads(open(stats_filename, 'rt').read())
        has_apply_macros = bool(stats.get('ApplyMacros', 0))
        has_preallocate = bool(stats.get('Preallocate', 0))

        allow_transforms = list(transforms.transforms_reverse_dep_order)
        if s_annotated is not None:
            transformL = transforms.parse_transforms(self, s_annotated)
        else:
            transformL = copy.deepcopy(self.transformL_deps)
        transformL = [transform.__class__.__name__ for transform in transformL] + ['ApplyMacros', 'Preallocate']
        allow_transforms = [transform for transform in allow_transforms if transform in transformL]
        
        if not has_preallocate and 'Preallocate' in allow_transforms:
            allow_transforms.remove('Preallocate')
        if not has_apply_macros and 'ApplyMacros' in allow_transforms:
            allow_transforms.remove('ApplyMacros')

        return allow_transforms
    
    def convert_vectorize_to_loop_implicit(self):
        print('  Special handling of VectorizeInnermost: transform list before replacement with LoopImplicit:', self.transformL)
        for j in range(len(self.transformL)):
            if isinstance(self.transformL[j], transforms.VectorizeInnermost):
                self.transformL[j] = transforms.LoopImplicit(self, self.transformL[j].line)
        print('  Special handling of VectorizeInnermost: transform list after replacement with LoopImplicit:', self.transformL)
    
    def validate_speedups_after(self, run_args=(), run_kw_args={}):
        transformL_orig = copy.deepcopy(self.transformL_deps)
        run_kw_args['extra_info'] = {}
        run_kw_args['override_n_tests'] = 5
        
        allow_transforms = self.get_allow_transforms()
        print('Validating speedups after tuning: allowed transforms are:', allow_transforms)

        transform_speedup_table = [[numpy.nan for j in range(len(transforms.transforms_reverse_dep_order))]]
        app_list = ['Application']
        time_list = []
        for nremove_transforms in range(len(allow_transforms)+1):
            remain_transforms = allow_transforms[nremove_transforms:]
            print('  Current transforms:', remain_transforms)
            set_enable_transforms(self, None, remain_transforms)
            self.transformL = copy.deepcopy(transformL_orig)
            self.transformL = [t for t in self.transformL if isinstance(t, tuple(transforms.all_transforms))]
            print('  Transform list:', self.transformL)
            py_source = None
            try:
                run_info = self.run(**run_kw_args)
#                run_info = self.run(print_source=True, **run_kw_args)
                time_list.append(run_info['time'])
                py_source = run_kw_args['extra_info']['source_add_transforms']
#                speedup = self.initial_run_info['time'] / run_info['time']
            except:
                self.print_log('  *** Could not get speedup for transforms: {}'.format(repr(remain_transforms)))
                traceback.print_exc(file=self.log_output_file)
#                time_list.append(numpy.nan)
                time_list.append(-1)
            if py_source is not None:
                py_source_filename = os.path.join(self.stats_dir, 'program_validate_speedups_after_%02d.py'%nremove_transforms)
                with open(py_source_filename, 'wt') as py_source_file:
                    py_source_file.write(py_source)
            print('  Time list:', time_list)
        
        for itransform in range(len(allow_transforms)):
            current_transform = allow_transforms[itransform]
            current_time = time_list[itransform]
            next_time = time_list[itransform+1]
            if current_time < 0 or next_time < 0:
                speedup = -1
            else:
                speedup = next_time/current_time
            
            # Specially handle VectorizeInnermost so as to measure only the speedup due to vectorization (and exclude compilation).
            # Do this by replacing VectorizeInnermost with LoopImplicit.
            if current_time >= 0 and allow_transforms[itransform] == 'VectorizeInnermost':
                remain_transforms = allow_transforms[itransform:]
                set_enable_transforms(self, None, remain_transforms)
                self.transformL = copy.deepcopy(transformL_orig)
                self.transformL = [t for t in self.transformL if isinstance(t, tuple(transforms.all_transforms))]
                self.convert_vectorize_to_loop_implicit()

                py_source = None
                run_kw_args_special = copy.deepcopy(run_kw_args)
                run_kw_args_special['vectorize'] = False
                try:
                    run_info = self.run(**run_kw_args_special)
                    next_time = run_info['time']
                    py_source = run_kw_args['extra_info']['source_add_transforms']
                except:
                    self.print_log('  *** Special handling of VectorizeInnermost: could not get speedup for transforms: {}'.format(repr(remain_transforms)))
                    traceback.print_exc(file=self.log_output_file)
                    next_time = -1.0
                if py_source is not None:
                    py_source_filename = os.path.join(self.stats_dir, 'program_validate_speedups_after_vectorize_innermost.py')
                    with open(py_source_filename, 'wt') as py_source_file:
                        py_source_file.write(py_source)
            
                if next_time >= 0:
                    speedup = next_time/current_time
                else:
                    speedup = -1
            
            idx_transform = transforms.transforms_reverse_dep_order.index(current_transform)
            transform_speedup_table[0][idx_transform] = speedup
            
        print_transform_speedup_table(transform_speedup_table, app_list)

        with open(self.validate_speedups_after_filename, 'wt') as f_v:
            print_transform_speedup_table(transform_speedup_table, app_list, file=f_v)

    def tune(self, run_args=(), run_kw_args={}, prob_add=0.25, prob_mutate=0.2, prob_remove=0.15, prob_sample_initial=0.4, prob_duplicate=0.0, max_iters=default_max_iters, tries=50):
        """
        Run up to max_iters times, print information about each run, and store best program transforms back in self.transformsL.
        
        Returns a dict with 'run' => run information (returned by test function in module), 'speedup' => float speedup.
        """
        tune_start_time = time.time()
        
        def write_tune_time():
            tune_time = time.time()-self.init_time
            if transforms.do_profile:
                transforms.profile['ProgramInfo: total'] = tune_time
                transforms.profile['ProgramInfo: tune'] = time.time() - tune_start_time
            with open(os.path.join(self.stats_dir, 'tune_time.txt'), 'wt') as tune_time_file:
                tune_time_file.write(str(tune_time))

        with open(self.tune_filename, 'wt') as tune_file:
            tune_file.write(self.full_filename if self.full_filename is not None else '')
        
        scale = prob_add + prob_mutate + prob_remove + prob_sample_initial + prob_duplicate
        prob_add /= scale
        prob_mutate /= scale
        prob_remove /= scale
        prob_sample_initial /= scale
        prob_duplicate /= scale
        
        self.get_initial_run_info()
        
        log_speedup_file = open(self.speedup_filename, 'wt')

        self.print_log('Initial program run time: {} secs'.format(self.initial_run_info['time']))
        self.print_log('Writing tuner output to {}'.format(self.compile_dir))
        
        if self.transformL_deps is None:
            self.transformL_deps = copy.deepcopy(self.transformL)
        
        seen = set()
        prev_run_info = {}
        prev_transform_stats = {}
        prev_speedup = 0.0
        prev_time = 100000.0
        if debug_track_memory:
            self.memory_tracker = pympler.tracker.SummaryTracker()
        
        (initial_guessL, likely_fastest_guess) = self.get_initial_guesses(run_args, run_kw_args)
        self.print_log('Likely fastest initial guess: {}'.format(likely_fastest_guess))
        
        final_index = None
        
        i = 0                           # Iteration count, number of unique transform lists tried
        i_repeated_count = -1           # A counter that doubly counts if we revisit a transform list
        
        # Continue until we hit max_iters but also eventually bail if we cannot generate enough unique transform lists (due to i_repeated_count)
        while i < max_iters and i_repeated_count < 10*max_iters:
            i_repeated_count += 1
            if debug_track_memory:
                self.memory_tracker.print_diff()
            def prefix():
                return 'Tune iteration {} ({:.0f} mins tuning, {:.6f} secs program run, {:.1f}x speedup): '.format(i, (time.time()-self.init_time)/60.0, prev_time, prev_speedup)
            
            def write_log_file():
                log_speedup_file.write('{}, {}, {}, "{}"\n'.format(i, prev_speedup, prev_time, str_list(self.transformL_deps)))
                log_speedup_file.flush()
            
            prev = copy.deepcopy(self.transformL)
            prev_deps = copy.deepcopy(self.transformL_deps)
            reverse_deps = False
            
            if i_repeated_count < len(initial_guessL):
                # Get transforms from initial guess list
                self.transformL = copy.deepcopy(initial_guessL[i_repeated_count])
            elif i_repeated_count < 2 * len(initial_guessL):
                self.transformL = copy.deepcopy(initial_guessL[i_repeated_count - len(initial_guessL)])
                reverse_deps = True
            else:
                # Construct new transform list by hill climbing (add, mutate, remove, sample initial guess) from current best list

                start_time_new_transform = time.time()
                try:
                    r = random.random()
                    
                    if r <= prob_add:                                       # Add
    #                    print('* Add transform')
                        ok = False
                        for j in range(tries):
                            try:
                                self.add_new_transform(random.choice(transforms.all_transforms))
                                ok = True
                                break
                            except transforms.MutateError:
                                continue
                        if not ok:
                            self.print_log('{}Failed to add any transform'.format(prefix()))
                            write_log_file()
                            continue
                    elif r <= prob_add + prob_mutate:                       # Mutate
    #                    print('* Mutate transform')
                        if not len(self.transformL):
                            self.print_log('{}Could not mutate a transform because list is empty'.format(prefix()))
                            write_log_file()
                            continue
                        ok = False
                        for j in range(tries):
                            rsel = random.randrange(2)
                            if rsel == 0:
                                # Sample TypeSpecialize instance and turn off checks if it is on
                                sub_transformL = [transform for transform in self.transformL if isinstance(transform, transforms.TypeSpecialize) and transform.checks]
                                if len(sub_transformL):
                                    transform = random.choice(sub_transformL)
                                    transform.checks = False
                                    ok = True
                                    break
                                else:
                                    continue
                            else:
                                # Sample random instance and mutate it
                                k = random.randrange(len(self.transformL))
                                try:
                                    self.transformL[k].mutate()
                                    ok = True
                                    break
                                except transforms.MutateError:
                                    continue
                        if not ok:
                            self.print_log('{}Could not mutate any transform'.format(prefix()))
                    elif r <= prob_add + prob_mutate + prob_remove:         # Remove
    #                    print('* Remove transform')
                        if not len(self.transformL):
                            self.print_log('{}Could not remove a transform because list is empty'.format(prefix()))
                            write_log_file()
                            continue
                        k = random.randrange(len(self.transformL))
                        del self.transformL[k]
                    elif r <= prob_add + prob_mutate + prob_remove + prob_sample_initial:   # Sample initial guess (adding transform)
    #                    print('* Sample initial guess')
                        if not len(likely_fastest_guess):
                            self.print_log('{}Could not sample initial guess because it is empty'.format(prefix()))
                            write_log_file()
                            continue
                        ok = False
                        for j in range(tries):
                            try:
                                self.add_new_transform(copy.deepcopy(random.choice(likely_fastest_guess)))
                                ok = True
                                break
                            except transforms.MutateError:
                                continue
                        if not ok:
                            self.print_log('{}Failed to add any transform'.format(prefix()))
                            write_log_file()
                            continue
                    else:                                                   # Duplicate (duplicate existing and then mutate it)
    #                    print('* Duplicate and mutate transform')
                        if not len(self.transformL):
                            self.print_log('{}Could not duplicate a transform because list is empty'.format(prefix()))
                            write_log_file()
                            continue
                        ok = False
                        for j in range(tries):
                            k = random.randrange(len(self.transformL))
                            try:
                                transform_copy = copy.deepcopy(self.transformL[k])
                                transform_copy.mutate()
                                self.add_new_transform(transform_copy)
                                ok = True
                                break
                            except transforms.MutateError:
                                continue
                        if not ok:
                            self.print_log('{}Could not mutate any transform'.format(prefix()))
                finally:
                    if transforms.do_profile:
                        transforms.profile['ProgramInfo: tune: sample new transform'] += time.time() - start_time_new_transform

            self.print_log('{}(before dependency resolution): {}'.format(prefix(), str_list(self.transformL)))
            
            transformL_no_deps = list(self.transformL)
            
            success = True
            visited = False         # Did we visit a new transform list? (even if there were compile-time or run-time errors)
            
            self.resolve_dependencies(reverse=reverse_deps)
            for transform in self.transformL:
                assert not hasattr(transform, 'annotated_line')
                if not isinstance(transform, tuple(transforms.all_transforms)):
                    print('  Skipping: transform {} after dependency resolution is not in allowed list {}'.format(transform, transforms.all_transforms))
                    success = False
                if isinstance(transform, transforms.TypeSpecialize) and self.max_types is not None and len(transform.typesL) > self.max_types:
                    print('  Skipping: transform is TypeSpecialize with {} types which is greater than max_types={}'.format(len(transform.typesL), self.max_types))
                    success = False
            

            self.print_log('{}(after dependency resolution): {}'.format(prefix(), str_list(self.transformL)))
            
            transformL_repr = repr(self.transformL)
            extra_info = {}
            if tuner_error_on_fail or tuner_error_on_transform_error:
                #run_kw_args['verbose'] = tuner_error_on_fail_verbose
                run_kw_args['clean_on_fail'] = False
            run_kw_args['extra_info'] = extra_info
            if transformL_repr in seen:
                self.print_log('  Skipping (already visited)')
                success = False
            else:
                seen.add(transformL_repr)
                visited = True
                try:
                    run_info = self.run(*run_args, **run_kw_args)
    #            util.print_header('Cython source:', extra_info['source'])
    #            print('run result:', run_info)
                    speedup = self.initial_run_info['time'] / run_info['time']
                    self.print_log('  Speedup: {}'.format(speedup))
                    
                    if speedup < prev_speedup:
                        success = False
                    else:
                        final_index = i
                        with open(self.program_prefix(i) + '.pyx', 'wt') as source_file:
                            source_file.write(extra_info['source'])
                        with open(self.program_prefix(i) + '.html', 'wt') as html_file:
                            html_file.write(extra_info['html'])
                        with open(self.program_prefix(i) + '.py', 'wt') as py_file:
                            py_file.write(extra_info['source_add_transforms'])
                        write_tune_time()
                except RunFailed:
                    self.print_log('  Failed')
                    if tuner_error_on_fail:
                        raise
                    success = False
                except ConsistencyFailed:
                    self.print_log('  Failed consistency check')
                    success = False
                except transforms.TransformError:
                    self.print_log('  Failed due to transform error')
                    if tuner_error_on_transform_error:
                        raise
                    traceback.print_exc()
                    traceback.print_exc(file=self.log_output_file)
                    success = False
                except:
                    self.print_log('  *** Tuning terminated early due to unhandled exception')
                    traceback.print_exc(file=self.log_output_file)
                    raise
                finally:
                    for tune_no in range(final_index):
                        move_to_sub_if_exists(self.program_prefix(tune_no) + '.pyx', self.temp_compile_dir)
                        move_to_sub_if_exists(self.program_prefix(tune_no) + '.py', self.temp_compile_dir)
                        move_to_sub_if_exists(self.program_prefix(tune_no) + '.html', self.temp_compile_dir)
        
            if not success:
                self.transformL = prev
                self.transformL_deps = prev_deps
            else:
                prev_run_info = run_info
                prev_transform_stats = extra_info['transform_stats']
                prev_speedup = speedup
                prev_time = run_info['time']
                self.transformL_deps = self.transformL
                self.transformL = transformL_no_deps

            self.print_log('{}At end of iteration, transformL={}'.format(prefix(), str_list(self.transformL)))
            write_log_file()

            if visited:
                i += 1

        if final_index is not None:
            final_dir = self.final_dir
            pre = self.program_prefix(final_index)
            shutil.copyfile(pre + '.pyx', self.final_pyx_filename)
            shutil.copyfile(pre + '.py', self.final_py_filename)
            shutil.copyfile(pre + '.html', self.final_html_filename)
            if self.out_file is not None:
                shutil.copyfile(pre + '.pyx', self.out_file)
            with open(os.path.join(final_dir, 'cython_headers.pyx'), 'wt') as header_f:
                header_f.write(transforms.cython_headers)
            util.write_vectorization_header_file(
                os.path.join(final_dir, 'vector_headers.h'))
            
        final_speedup_str = str(prev_speedup)
        final_transform_str = str_list(self.transformL)
        final_transform_stats_str = json.dumps(prev_transform_stats, sort_keys=True)

        self.final_py_time_filename = os.path.join(self.stats_dir, 'final_py_time.txt')
        self.final_ours_time_filename = os.path.join(self.stats_dir, 'final_ours_time.txt')

        with open(self.final_py_time_filename, 'wt') as final_py_time_file:
            final_py_time_file.write(str(self.initial_run_info['time']))

        with open(self.final_ours_time_filename, 'wt') as final_ours_time_file:
            final_ours_time_file.write(str(prev_time))
        
        with open(self.final_speedup_filename, 'wt') as final_speedup_file:
            final_speedup_file.write(final_speedup_str)

        with open(self.final_transforms_filename, 'wt') as final_transforms_file:
            final_transforms_file.write(final_transform_str)

        with open(self.final_transform_stats_filename, 'wt') as final_transform_stats_file:
            final_transform_stats_file.write(final_transform_stats_str)

        with open(self.run_time_filename, 'wt') as run_time_file:
            run_time_file.write(str(self.run_time))

        with open(self.run_count_filename, 'wt') as run_count_file:
            run_count_file.write(str(self.run_count))

        util.print_header('Final Tuned Program', """
Speedup:              {final_speedup_str}
Transforms:           {final_transform_str}
Transform Statistics: {final_transform_stats_str}
""".format(**locals()), file_list=[sys.stdout, self.log_output_file])

        write_tune_time()

        return {'run': prev_run_info, 'speedup': prev_speedup}

def run_numba(program_info, run_kw_args={}):
    """
    Run program with numba.jit() enabled on all functions, return result from run_code().
    """
    s = program_info.s_orig
    L = s.split('\n')
    i = 0
    while i < len(L):
        if L[i].lstrip().startswith('def '):
            funcname = L[i].lstrip().split()[1]
            if not util.is_test_funcname(funcname):
                L[i:i] = ['@numba.jit']
                i += 1
        i += 1
    s = '\n'.join(L)
    s = 'import numba\n' + s
    if verbose:
        util.print_header('Numba decorated code:', s)
    return run_code(program_info.path, s, **run_kw_args)

def set_disable_transforms(program_info, args, disable_transforms):
    orig_all_transforms = list(transforms.orig_all_transforms)
    transforms.all_transforms = list(orig_all_transforms)
    if program_info is not None:
        program_info.preallocate = True
        program_info.apply_macros = True
    if args is not None:
        args.preallocate = True
        args.apply_macros = True
    for t in disable_transforms:
        if t == 'Preallocate':
            if args is not None:
                args.preallocate = False
            if program_info is not None:
                program_info.preallocate = False
        elif t == 'ApplyMacros':
            if args is not None:
                args.apply_macros = False
            if program_info is not None:
                program_info.apply_macros = False
        else:
            transform = getattr(transforms, t)
            transforms.all_transforms.remove(transform)

def set_enable_transforms(program_info, args, enable_transforms):
    orig_all_transforms = list(transforms.orig_all_transforms)
    all_transforms = [transform for transform in enable_transforms if len(transform)]
    
    # Modify flags for special transforms that are typically always enabled (and therefore not included in the list of transforms)
    for set_obj in [program_info, args]:
        if set_obj is None:
            continue
        set_obj.preallocate = False
        set_obj.apply_macros = False
        
        for t in all_transforms:
            if t == 'Preallocate':
                set_obj.preallocate = True
            elif t == 'ApplyMacros':
                set_obj.apply_macros = True

    # Place non-special transforms in transforms.all_transforms while maintaining their order
    transforms.all_transforms = []
    for tcls in orig_all_transforms:
        t = tcls.__name__
        if t in all_transforms:
            transforms.all_transforms.append(getattr(transforms, t))

def print_transform_speedup_table(transform_speedup_table, appname_L, file=sys.stdout):
    sorted_idx_to_unsorted_idx = [tup[1] for tup in sorted([(desc, j) for (j, desc) in enumerate(transforms.transforms_reverse_dep_order_descriptions)])]
    n = len(transforms.transforms_reverse_dep_order_descriptions)
    print('Application,', end='', file=file)
    for (j, transform) in enumerate(transforms.transforms_reverse_dep_order):
        unsorted_idx = sorted_idx_to_unsorted_idx[j]
        print(transforms.transforms_reverse_dep_order_descriptions[unsorted_idx] + (',' if j < n-1 else ''), end='', file=file)
    print(file=file)
    for i in range(len(appname_L)):
        print(appname_L[i] + ',', end='', file=file)
        for j in range(n):
            unsorted_idx = sorted_idx_to_unsorted_idx[j]
            print('%22.3f'%transform_speedup_table[i][unsorted_idx] + (',' if j < n-1 else ''), end='', file=file)
        print(file=file)
    print(file=file)


def main_args(args):
    global use_subprocess
    global verbose
    use_subprocess = args.subprocess
    
    if args.validate or args.validate_speedups or args.retime or args.validate_images:
        validate_dir = os.path.join(os.getcwd(), ('out_validate' if (args.validate or args.validate_speedups) else 'out_retime'))
        if args.out_dir is not None:
            validate_dir = os.path.abspath(args.out_dir)
        basedir = os.path.abspath(args.filename)
        
        if not os.path.exists(validate_dir):
            os.makedirs(validate_dir)

        print('Validating directory {}'.format(basedir))
        print('Validation output written to {}'.format(validate_dir))
        if args.validate_speedups:
            print('Transforms in reverse dependency order: {}'.format(transforms.transforms_reverse_dep_order))
        print()
        time_stats = {transform: [] for transform in transforms.transforms_reverse_dep_order}
        orig_time_stats = copy.deepcopy(time_stats)
        
        def print_summary_stats():
            print('Transform,Mean Speedup,StdDev Speedup,Max Speedup,Samples')
            for transform in sorted(transforms.transforms_reverse_dep_order):
                subL = time_stats[transform]
                mean_val = std_val = max_val = 'NA'
                if len(subL):
                    mean_val = numpy.nanmean(subL)
                    std_val = numpy.nanstd(subL)
                    max_val = numpy.nanmax(subL)
                print('{},{:.3},{:.3},{:.3},{}'.format(transform, mean_val, std_val, max_val, len(subL)))
            print()
            
            app_list = [subdir_info[i][0] for i in range(len(subdir_info))]
            print_transform_speedup_table(transform_speedup_table, app_list)

            subdir = 'all_stats'
            subdir_full = os.path.join(validate_dir, subdir, 'stats')
            if not os.path.exists(subdir_full):
                os.makedirs(subdir_full)
            validate_speedups_filename = os.path.join(validate_dir, subdir, 'stats', 'validate_speedups_after.txt')
            with open(validate_speedups_filename, 'wt') as f_v:
                print_transform_speedup_table(transform_speedup_table, app_list, file=f_v)
        
            #sorted_idx_to_unsorted_idx = [tup[1] for tup in sorted([(desc, j) for (j, desc) in enumerate(transforms.transforms_reverse_dep_order_descriptions)])]
            #n = len(transforms.transforms_reverse_dep_order_descriptions)
            #print('Application,', end='')
            #for (j, transform) in enumerate(transforms.transforms_reverse_dep_order):
            #    unsorted_idx = sorted_idx_to_unsorted_idx[j]
            #    print(transforms.transforms_reverse_dep_order_descriptions[unsorted_idx] + (',' if j < n-1 else ''), end='')
            #print()
            #for i in range(len(subdir_info)):
            #    print(subdir_info[i][0] + ',', end='')
            #    for j in range(n):
            #        unsorted_idx = sorted_idx_to_unsorted_idx[j]
            #        print('%22.3f'%transform_speedup_table[i][unsorted_idx] + (',' if j < n-1 else ''), end='')
            #    print()
            #print()
            
        subdirs = [subdir for subdir in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, subdir))]
        if args.validate_list is not None:
            subdirs = args.validate_list.split(',')
    
        print()
        is_validating = (args.validate or args.validate_speedups or args.validate_images)
        print('Applications to be {}:'.format('validated' if is_validating else 'retimed'))
        subdir_info = []
        for subdir in sorted(subdirs):
            final_program = os.path.join(basedir, subdir, 'final', 'program.py')
            tune_filename = os.path.join(basedir, subdir, 'stats', 'tune_filename.txt')
            stats_filename = os.path.join(basedir, subdir, 'stats', 'final_transform_stats.txt')
            all_exist = os.path.exists(final_program) and os.path.exists(tune_filename) and os.path.exists(stats_filename)
            subdir_info.append((subdir, final_program, tune_filename, stats_filename, all_exist))
            if not all_exist:
                print(' * Warning: will not validate {} because of missing files'.format(subdir))
            else:
                print(subdir)
        print()

        transform_speedup_table = [[numpy.nan for j in range(len(transforms.transforms_reverse_dep_order))] for i in range(len(subdir_info))]

        all_ok = True
        for (isubdir, (subdir, final_program, tune_filename, stats_filename, all_exist)) in enumerate(subdir_info):
            print(('Validating' if is_validating else 'Retiming'), final_program)
            
            if os.path.exists(final_program) and os.path.exists(tune_filename) and os.path.exists(stats_filename):
                final_program_source = open(final_program, 'rt').read()
                tune_filename = open(tune_filename, 'rt').read()

                # Convert absolute pathname for tune_filename into relative pathname
                if 'annotating_compiler' in tune_filename:
                    rest = tune_filename[tune_filename.index('annotating_compiler')+len('annotating_compiler')+1:]
                    tune_filename = os.path.abspath(os.path.join('../../', rest))

                tune_path = os.path.split(tune_filename)[0]
                final_orig_dir_filename = os.path.join(tune_path, '_compile_validate.py') if (args.validate or args.validate_speedups) else os.path.join(tune_path, tune_filename)
                with open(final_orig_dir_filename, 'wt') as final_file:
                    final_file.write(final_program_source)
                sub_name = os.path.split(subdir)[-1]
                out_filename = os.path.join(validate_dir, sub_name + '.pyx')
                out_time = '_compile_out_time.txt'
                
                if args.validate_speedups:
                    program_info = ProgramInfo(final_program_source, tune_path, types={}, preallocate=False)

                    allow_transforms = program_info.get_allow_transforms(stats_filename, final_program_source)
                    
                    print('Transforms for this program:', allow_transforms)
                else:
                    allow_transforms = []
            
                def get_status(errcode):
                    return 'Program Ran OK' if errcode == 0 else 'Error (status {})'.format(errcode)
                
                time_list = []
                run_cmdL = []
                for nremove_transforms in range(len(allow_transforms)+1):
                    remain_transforms = allow_transforms[nremove_transforms:]
                    extra_args = ''
                    if not args.python:
                        extra_args += ' --no-python'
                    if not args.comparisons:
                        extra_args += ' --no-comparisons'
                    if args.validate_speedups:
                        transform_str_L = ','.join(remain_transforms)
                        if len(transform_str_L) == 0:
                            transform_str_L = ','
                        extra_args = ' --ntests 50 --transforms ' + transform_str_L
                    if args.validate_images:
                        extra_args += ' --out-image ' + os.path.join(validate_dir, sub_name + '.png')
                    if args.retime:
                        ntests_args = ''
                        if args.ntests > 0:
                            ntests_args = ' --ntests {}'.format(args.ntests)
                        basedir_subdir = os.path.join(basedir, subdir)
                        cmd = 'python compiler.py {final_orig_dir_filename} --no-tune --quiet{ntests_args} --out-dir {basedir_subdir} --out-file {out_filename} --out-time {out_time}{extra_args}'.format(**locals())
                        errcode = util.system_verbose(cmd, exit_if_error=False)
                    else:
                        cmd = 'python compiler.py {final_orig_dir_filename} --no-tune-annotated --no-comparisons --quiet --no-python --out-file {out_filename} --out-time {out_time}{extra_args}'.format(**locals())
                        if args.validate_images:
                            cmd = cmd.replace('--no-python ', '')
                        errcode = util.system_verbose(cmd, exit_if_error=False)
                    run_cmdL.append(cmd)
                    status = get_status(errcode)
                    if errcode != 0:
                        all_ok = False
                    print(' *', subdir.ljust(30), status)
                    if args.retime and errcode == 0:
                        time_info = json.loads(open(out_time, 'rt').read())
                        validate_subdir = os.path.join(validate_dir, subdir)
                        py_time = time_info.get('python', 0.0)
                        ours_time = time_info.get('ours', 0.0)
                        pypy_time = time_info.get('pypy', 0.0)
                        numba_time = time_info.get('numba', 0.0)
                        unpython_time = time_info.get('unpython', 0.0)
                        speedup = py_time / ours_time if ours_time else 0.0
                        numba_speedup = numba_time / ours_time if ours_time else 0.0
                        pypy_speedup = pypy_time / ours_time if ours_time else 0.0
                        unpython_speedup = unpython_time / ours_time if ours_time else 0.0
                        if os.path.exists(validate_subdir):
                            shutil.rmtree(validate_subdir)
                        shutil.copytree(os.path.join(basedir, subdir), validate_subdir)
                        
                        def write_float(filename, value):
                            with open(filename, 'wt') as f_float:
                                f_float.write(str(value))
                    
                        write_float(os.path.join(validate_subdir, 'stats/final_py_time.txt'), py_time)
                        write_float(os.path.join(validate_subdir, 'stats/final_ours_time.txt'), ours_time)
                        write_float(os.path.join(validate_subdir, 'stats/final_numba_time.txt'), numba_time)
                        write_float(os.path.join(validate_subdir, 'stats/final_pypy_time.txt'), pypy_time)
                        write_float(os.path.join(validate_subdir, 'stats/final_unpython_time.txt'), unpython_time)

                        write_float(os.path.join(validate_subdir, 'stats/final_speedup.txt'), speedup)
                        write_float(os.path.join(validate_subdir, 'stats/numba_speedup.txt'), numba_speedup)
                        write_float(os.path.join(validate_subdir, 'stats/pypy_speedup.txt'), pypy_speedup)
                        write_float(os.path.join(validate_subdir, 'stats/unpython_speedup.txt'), unpython_speedup)
                    elif errcode == 0:
                        time_info = json.loads(open(out_time, 'rt').read())
                        ours_time = time_info.get('ours', 0.0)
                        try:
                            py_time = float(open(os.path.join(basedir, subdir, 'stats', 'final_py_time.txt'), 'rt').read())
                        except:
                            py_time = 0.0
                        try:
                            orig_speedup = float(open(os.path.join(basedir, subdir, 'stats', 'final_speedup.txt'), 'rt').read())
                        except:
                            orig_speedup = 0.0
                        
                        speedup = py_time / ours_time if ours_time else 0.0
                        print(' *', subdir.ljust(30), 'speedup:', '%.2f'%speedup, '(previous tuned speedup was %.2f)'%orig_speedup)
                        #if speedup < orig_speedup * 0.75 and nremove_transforms == 0:
                        #    print(' *', subdir.ljust(30), 'Error: speedup is significantly slower than original speedup of {}'.format(orig_speedup))
                        #    all_ok = False
                            
                    if args.validate_speedups or True:
                        if errcode == 0:
                            time_dict = json.loads(open(out_time, 'rt').read())
                            time_list.append(time_dict['ours'])
                        else:
                            time_list.append(numpy.nan)
                    print(' *', subdir.ljust(30), 'time list:', time_list)
                    print()

                if args.validate_speedups:
                    current_time_stats = copy.deepcopy(orig_time_stats)
                    for itransform in range(len(allow_transforms)):
                        current_transform = allow_transforms[itransform]
                        current_time = time_list[itransform]
                        next_time = time_list[itransform+1]
                        
                        if current_transform == 'VectorizeInnermost':
                            cmd = run_cmdL[itransform]
                            print('Special handling of VectorizeInnermost:')
                            cmd += ' --no-vectorize'
                            print(cmd)
                            errcode = util.system_verbose(cmd, exit_if_error=False)

                            status = get_status(errcode)
                            if errcode != 0:
                                all_ok = False
                            print(' *', subdir.ljust(30), status)

                            time_info = json.loads(open(out_time, 'rt').read())
                            next_time = time_info.get('ours', 0.0)
                            print(' * current_time={}, next_time={}'.format(current_time, next_time))
                        
                        speedup = next_time/current_time
                        time_stats[current_transform].append(speedup)
                        current_time_stats[current_transform].append(speedup)
                        idx_transform = transforms.transforms_reverse_dep_order.index(current_transform)
                        transform_speedup_table[isubdir][idx_transform] = speedup
                    print(' => Time list:', time_list)
                    print(' => Speedup current raw data:', current_time_stats)
                    print(' => Speedup cumulative raw data:', time_stats)
                    print(' => Speedup full raw table:', transform_speedup_table)
                    print()
                    print_summary_stats()
                    print()

        print()
        if all_ok:
            print('Validation: OK')
        else:
            print('Validation: Failed')
        sys.exit(0 if all_ok else 1)
    
    if args.profile:
        transforms.do_profile = True
    
    if args.once:
        args.comparisons = False
        args.tune_annotated = False
    
    if not args.comparisons:
        args.numba = False
        args.pypy = False
        args.unpython = False

    if args.in_image is not None:
        util.override_input_image = os.path.abspath(args.in_image)
    
    if args.use_4channel:
        util.use_4channel = True

    def capture_args(*args, **kw):
        return (args, kw)

    override_args = None
    override_kw_args = None
    if args.args is not None:
        (override_args, override_kw_args) = eval('capture_args(' + args.args + ')', locals(), locals())

    global quiet
    quiet = args.verbose == 0
    if args.verbose >= 2:
        verbose = True
    if args.verbose >= 3:
        transforms.verbose = True

    if not args.tune_annotated and not args.tune:
        print('Cannot specify both --no-tune-annotated and --no-tune.', file=sys.stderr)
        sys.exit(1)
    
    if args.all_transforms is not None:
        enable_transforms = args.all_transforms.split(',')
        set_enable_transforms(None, args, enable_transforms)

    if args.disable_transforms is not None:
        disable_transforms = args.disable_transforms.split(',')
        set_disable_transforms(None, args, disable_transforms)

    (path, filename) = os.path.split(args.filename)
    if len(path) == '':
        path = './'
    path = os.path.abspath(path)
    s_orig = open(os.path.join(path, filename), 'rt').read()
    full_filename = os.path.abspath(args.filename)

    if os.path.splitext(args.filename)[1].lower() == '.pyx':
        filename_prefix = os.path.splitext(full_filename)[0]
        (path, filename_prefix_no_path) = os.path.split(filename_prefix)
        #util.system_verbose('cython -3 {}'.format(args.filename))
        os.chdir(path)
        util.compile_cython_single_file(filename_prefix_no_path, c_edit_func)
#        util.compile_c_single_file(filename_prefix)
        if args.run:
            util.system_verbose('python -c "import {} as t; t.test({})"'.format(filename_prefix_no_path, args.args if args.args is not None else ''))
        sys.exit(0)
    
    program_info = ProgramInfo(
        preprocess.preprocess_input_python(s_orig),
        path, 
        types={} if not args.tune else None,    # Auto-detect types unless in --no-tune mode
        log_transforms=default_log_transforms, 
        compile_dir=args.out_dir, 
        is_verbose=args.run_verbose, 
        filename=args.filename, 
        max_types=args.max_types if args.max_types > 0 else None, 
        apply_macros=args.apply_macros,
        verbose_level=args.verbose,
        out_file=args.out_file,
        preallocate=args.preallocate,
        preallocate_verbose=args.tune_annotated and args.tune,
        full_filename=full_filename,
        quiet=quiet,
        use_4channel=args.use_4channel,
        safe=args.safe)
    
    transforms.profile.out_files.append(program_info.log_output_file)

    try:
        if not quiet and args.tune:
            util.print_header('Types:')
            pprint.pprint(program_info.types)
            print()

        extra_info = {}
        run_kw_args = dict(clean=True, verbose=args.run_verbose, cython=True, extra_info=extra_info, temp_dir=program_info.temp_compile_dir)

        all_run_info = {}

        def run_no_tune(pyx_filename, program_path):
            source = open(pyx_filename, 'rt').read()
            in_image = os.path.abspath(args.in_image) if args.in_image is not None else args.in_image
            ntests = args.ntests if args.ntests > 0 else 1
            if verbose:
                print('calling run_code:', program_path, '[source omitted]', 'override_input_image=', in_image, 'override_n_tests=', ntests)
                util.print_header('run_code source code:', source)
            return run_code(program_path, source, override_input_image=in_image, override_n_tests=ntests, override_args=override_args, override_kw_args=override_kw_args, use_4channel=args.use_4channel)

        def print_run_result(run_info):
            run_info_no_output = dict(run_info)
            run_info_no_output.pop('output', None)
            print('run result:', run_info_no_output)

        if args.ntests > 0:
            run_kw_args['override_n_tests'] = args.ntests
        if args.in_image is not None:
            run_kw_args['override_input_image'] = os.path.abspath(args.in_image)
        if args.args is not None:
            run_kw_args['override_args'] = override_args
            run_kw_args['override_kw_args'] = override_kw_args
        run_kw_args['use_4channel'] = False
        if not args.vectorize:
            run_kw_args['vectorize'] = False

        def handle_out_image(current_run_info, suffix=''):
            if len(suffix):
                suffix = '-' + suffix
            if args.out_image is not None:
                if 'output' in current_run_info:
                    (out_image_pre, out_image_ext) = os.path.splitext(args.out_image)
                    util.write_img(current_run_info['output'], out_image_pre + suffix + out_image_ext)
                else:
                    print('Warning: not saving output image (either tuner failed or test() function did not return dict with image output in key "output")')

        if args.python:
            start_time_python = time.time()
            initial_run_info = program_info.get_initial_run_info(run_kw_args=run_kw_args)
            all_run_info['python'] = initial_run_info
            handle_out_image(initial_run_info, 'python')
            if transforms.do_profile:
                transforms.profile['initial_run'] += time.time()-start_time_python
        else:
            initial_run_info = {'time': 1.0, 'error': 0.0}

        if args.use_4channel:
            run_kw_args['use_4channel'] = args.use_4channel

        if args.tune_annotated and args.tune:
            tune_info = program_info.tune(run_kw_args=run_kw_args, max_iters=args.max_iters)
            run_info = tune_info['run']
        elif not args.tune_annotated:       # --no-tune-annotated argument supplied
            s = program_info.s_orig
            L = program_info.s_orig.split('\n')
            Lp = [line for line in L if not line.strip().startswith(transforms.transform_str)]
            program_info.s_orig = s_orig = '\n'.join(Lp)
            s = transforms.move_transforms_to_end(program_info, s, s_orig)
            if verbose:
                util.print_header('s_orig:', s_orig, linenos=True)
                util.print_header('s:', s, linenos=True)

            program_info.transformL = transforms.parse_transforms(program_info, s)
            if verbose:
                util.print_header('transforms before filter:', program_info.transformL)
                util.print_header('all transforms:', transforms.all_transforms)
            program_info.transformL = [transform for transform in program_info.transformL if isinstance(transform, tuple(transforms.all_transforms))]
            if verbose:
                util.print_header('transforms before resolve_dependencies:', program_info.transformL)
            if args.resolve_deps:
                program_info.resolve_dependencies() #add_typespecialize=False)
            if verbose:
                util.print_header('transforms after resolve_dependencies:', program_info.transformL)
            if not args.vectorize:
                program_info.convert_vectorize_to_loop_implicit()

            run_kw_args['once'] = True
            run_kw_args['out_filename'] = program_info.out_file
            run_info = program_info.run(**run_kw_args)
            if not quiet:
                util.print_header('Cython source:', extra_info['source'])
                print_run_result(run_info)
    #        if program_info.out_file is not None:
    #            with open(program_info.out_file, 'wt') as out_file_f:
    #                out_file_f.write(extra_info['source'])
            speedup = initial_run_info['time'] / run_info['time'] if run_info['time'] and initial_run_info else 0.0
            if not quiet:
                print('speedup:', speedup)
        elif not args.tune:                 # --no-tune argument supplied
            run_info = run_no_tune(program_info.final_pyx_filename, program_info.path)
            if not quiet:
                print_run_result(run_info)
        else:
            raise ValueError

        if args.validate_speedups_after:
            program_info.validate_speedups_after(run_kw_args=run_kw_args)

        all_run_info['ours'] = run_info

        handle_out_image(run_info)

        # Compare with JITs: Numba and PyPy
        
        run_kw_args['cython'] = False
        run_kw_args['repeats'] = 2                      # Run twice to give JIT a chance to type-specialize and compile the first time
        if not args.tune and 'override_n_tests' not in run_kw_args:
            run_kw_args['override_n_tests'] = 1         # Test image only once in --no-tune mode

        if args.numba:
            try:
                run_info = run_numba(program_info, run_kw_args)
                numba_speedup = initial_run_info['time'] / run_info['time']
                print('numba speedup:', numba_speedup)

                with open(os.path.join(program_info.stats_dir, 'numba_speedup.txt'), 'wt') as numba_speedup_file:
                    numba_speedup_file.write(str(numba_speedup))

                with open(os.path.join(program_info.stats_dir, 'numba_time.txt'), 'wt') as numba_time_file:
                    numba_time_file.write(str(run_info['time']))

                all_run_info['numba'] = run_info
                handle_out_image(run_info, 'numba')
            except:
                traceback.print_exc()

        if args.pypy:
            try:
                modname = os.path.splitext(os.path.split(program_info.filename)[1])[0]
                compiler_path = sys.path[0]
                inject_overrides = ''
                if not args.tune:                           # Test image only once in --no-tune mode
                    
                    inject_overrides = 'import util; util.override_n_tests = {}'.format(args.ntests if args.ntests > 0 else 1)
                    if args.out_image is not None:
                        (out_image_pre, out_image_ext) = os.path.splitext(args.out_image)
                        pypy_out_image = os.path.abspath(out_image_pre + '-pypy' + out_image_ext)
                        inject_overrides += '\nutil.override_output_image = {!r}'.format(pypy_out_image)
                    if args.in_image is not None:
                        inject_overrides += '\nutil.override_input_image = {!r}'.format(os.path.abspath(args.in_image))
                    if args.use_4channel:
                        inject_overrides += '\nutil.use_4channel = True'
                args_str = args.args if args.args is not None else ''
                
                code_str = """
import sys; sys.path += [{program_info.path!r}, {compiler_path!r}]
import os
os.chdir({program_info.path!r})
{inject_overrides}
import {modname}
for j in range(2):
    result = {modname}.test({args_str})
if 'output' in result:              # Prevent bug due to pickling numpy array in PyPy and depickling in CPython
    del result['output']
    """.format(**locals())
    #        print(code_str)
                run_info = run_subprocess.run_subprocess(program_info.compile_dir, code_str, 'pypy')
                pypy_speedup = initial_run_info['time'] / run_info['time']
                print('pypy speedup:', pypy_speedup)

                with open(os.path.join(program_info.stats_dir, 'pypy_speedup.txt'), 'wt') as pypy_speedup_file:
                    pypy_speedup_file.write(str(pypy_speedup))

                with open(os.path.join(program_info.stats_dir, 'pypy_time.txt'), 'wt') as pypy_time_file:
                    pypy_time_file.write(str(run_info['time']))

                all_run_info['pypy'] = run_info
            except:
                traceback.print_exc()
            #handle_out_image(run_info, 'pypy')

        if args.unpython:
            try:
                if not args.tune:           # --no-tune argument supplied
                    unpython_filename = os.path.join(program_info.compile_dir, 'compare_unpython', program_info.final_subdir, 'program.pyx')
                    run_info = run_no_tune(unpython_filename, program_info.path)
                else:
                    unpython_out_dir = os.path.join(os.path.abspath(program_info.compile_dir), 'compare_unpython')
                    transforms_unpython = 'TypeSpecialize,Parallel'
                    if not args.parallel_unpython:
                        transforms_unpython = 'TypeSpecialize'
                    cmd = 'python compiler.py \"{args.filename}\" --no-comparisons --no-preallocate --transforms {transforms_unpython} --max-types 1 --no-macros --max-iters 10 --out-dir \"{unpython_out_dir}\"'.format(**locals())

                    print(cmd)
                    os.system(cmd)
                    unpython_speedup_filename = os.path.join(unpython_out_dir, program_info.default_stats_subdir, default_final_speedup_filename)
                    if os.path.exists(unpython_speedup_filename):
                        shutil.copyfile(unpython_speedup_filename, os.path.join(program_info.stats_dir, 'unpython_speedup.txt'))
                        try:
                            print('unpython speedup:', open(unpython_speedup_filename, 'rt').read())
                        except:
                            print('could not read unpython speedup')
                            traceback.print_exc()

                    unpython_time_filename = os.path.join(unpython_out_dir, program_info.default_stats_subdir, default_final_time_filename)
                    if os.path.exists(unpython_time_filename):
                        shutil.copyfile(unpython_time_filename, os.path.join(program_info.stats_dir, 'unpython_time.txt'))

                all_run_info['unpython'] = run_info
                handle_out_image(run_info, 'unpython')
            except:
                traceback.print_exc()

        all_times = {}
        for key in all_run_info:
            all_times[key] = all_run_info[key]['time']
        if args.out_time is not None:
            with open(args.out_time, 'wt') as out_time_file:
                out_time_file.write(json.dumps(all_times))
    finally:
        transforms.profile.close()
        program_info.log_output_file.flush()

def main():
    random.seed(3)
    
    parser = argparse_util.ArgumentParser(description='Compile Python program with tuner enabled by default.')
    parser.add_argument('filename', help='If .py extension, an input Python module with test() method.\nIf .pyx extension, compiles and runs the test() method of the given Cython module. If a directory and --validate is passed, check tuner output for that directory (see --validate).')
    parser.add_argument('--max-iters', dest='max_iters', type=int, default=ProgramInfo.default_max_iters, help='maximum number of tuner iterations')
    parser.add_argument('--run-verbose', dest='run_verbose', action='store_true', help='run code with verbose reporting')
    parser.add_argument('--no-run-verbose', dest='run_verbose', action='store_false', help='run code silently')
    parser.add_argument('--out-dir', dest='out_dir', default=None, help='compilation output directory: write log files here')
    parser.add_argument('--out-file', dest='out_file', default=None, help='final output file with .pyx extension (can be compiled to a module by running the compiler on it)')
    parser.add_argument('--out-time', dest='out_time', default=None, help='store output times of test routine to file with .json extension')
    parser.add_argument('--no-tune-annotated', dest='tune_annotated', action='store_false', help='evaluate speed of a single annotated Python source code without running tuner (this source code can be produced manually or by the tuner).')
    parser.add_argument('--in-image', dest='in_image', default=None, help='input image to run through program, if applicable. if a single filename is passed to the test routine then this overrides that.')
    parser.add_argument('--no-tune', dest='tune', action='store_false', help='given an original python program and --out-dir argument, evaluate the final tuned program previously output by the tuner (can be used with --in-image, --out-image, --out-time), plus any comparisons such as Numba, PyPy, etc.')
    parser.add_argument('--no-numba', dest='numba', action='store_false', help='do not compare speedup with result from Numba JIT')
    parser.add_argument('--no-pypy', dest='pypy', action='store_false', help='do not compare speedup with result from PyPy JIT')
    parser.add_argument('--no-unpython', dest='unpython', action='store_false', help='do not compare against unPython reimplementation')
    parser.add_argument('--no-comparisons', dest='comparisons', action='store_false', help='disable all comparisons against Numba, PyPy, unPython, etc')
    parser.add_argument('--no-python', dest='python', action='store_false', help='disable implicit comparison against pure Python (this is still enabled even when --no-comparisons is used)')
    parser.add_argument('--no-run', dest='run', action='store_false', help='do not run resulting output program when fed .pyx as input')
    parser.add_argument('--transforms', dest='all_transforms', help='list of all available transforms, e.g. --transforms TypeSpecialize,Parallel,LoopImplicit')
    parser.add_argument('--disable-transforms', dest='disable_transforms', help='list of transforms to disable, e.g. --disable-transforms Parallel')
    parser.add_argument('--max-types', dest='max_types', type=int, help='maximum number of type signatures for type specialization transformation')
    parser.add_argument('--ntests', dest='ntests', type=int, help='take minimum of exactly n executions of test function to determine program run time')
    parser.add_argument('--no-macros', dest='apply_macros', action='store_false', help='do not apply macros')
    parser.add_argument('--no-preallocate', dest='preallocate', action='store_false', help='do not preallocate arrays')
    parser.add_argument('--quiet', dest='verbose', action='store_false', help='suppress all output, same as --verbose 0')
    parser.add_argument('--once', dest='once', action='store_true', help='Run annotated Python file through compiler only once. Equivalent to --no-tune-annotated --no-comparisons')
    parser.add_argument('--no-resolve', dest='resolve_deps', action='store_false', help='Do not resolve dependencies in --no-tune-annotated mode (the default does resolve dependencies).')
    parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='set verbosity level. 0=quiet, 1=normal, 2=more compiler messages, 3=more compiler and transform messages')
    parser.add_argument('--out-image', dest='out_image', help='output image filename. by default, not saved. if comparisons are made against Numba, PyPy, etc, then the name of each comparison will be appended to the output filename')
    parser.add_argument('--profile', dest='profile', action='store_true', help='profile compiler, saving and printing timing information for functions')
    parser.add_argument('--no-subprocess', dest='subprocess', action='store_false', help='do not run program tests in a subprocess, which is the default (can be useful for obtaining debugging tracebacks)')
    parser.add_argument('--validate', dest='validate', action='store_true', help='Check (validate) previous tuner output, running each final tuned program through the compiler, and printing OK or else raising an error. The positional argument is the input validation directory. The --out-dir argument can be used to specify the validation output directory. Useful for unit testing without having to wait for the tuner to finish.')
    parser.add_argument('--validate-images', dest='validate_images', action='store_true', help='Alternative mode of --validate, where all output images are collected.')
    parser.add_argument('--validate-list', dest='validate_list', help='Optional comma-separated list of subdirectories of the input validation directory to be validated (by default, all subdirectories are validated).')
    parser.add_argument('--args', dest='args', help='Python string expression for additional arguments to be passed to the test() function of the Python program. For example: --args "1.0" or --args "time=1.0"')
    parser.add_argument('--validate-speedups', action='store_true', dest='validate_speedups', help='Deprecated due to subtle bugs. Similar to --validate: re-run each final tuned program. But report statistics for speedups "independently" due to each transform, by removing transforms in reversed dependency order and measuring each resulting slow-down.')
    parser.add_argument('--validate-speedups-after', action='store_true', dest='validate_speedups_after', help='A variant of --validate-speedups that runs a similar test immediately after the tuner is finished, rather than in a second independent run of the compiler.')
    parser.add_argument('--retime', action='store_true', dest='retime', help='Similar to --validate, given a positional argument that is a directory such as "out" containing tuner runs as subdirectories, retimes all the applications.')
    parser.add_argument('--4channel', action='store_true', dest='use_4channel', help='Force color images to load in 4 channel mode, which can improve SIMD vectorization.')
    parser.add_argument('--no-parallel-unpython', action='store_false', dest='parallel_unpython', help='Disable parallelism in unPython comparison.')
    parser.add_argument('--safe', action='store_true', dest='safe', help='Make sure all operations are safe: disable float64 => float32 conversion, and disable wrap-around indexing')
    parser.add_argument('--no-vectorize', action='store_false', dest='vectorize', help='Disable vectorization: convert vectorize transformation to a loop and disable vectorizer in C compiler.')
    parser.set_defaults(once=False)
    parser.set_defaults(numba=True)
    parser.set_defaults(python=True)
    parser.set_defaults(pypy=True)
    parser.set_defaults(unpython=True)
    parser.set_defaults(comparisons=True)
    parser.set_defaults(run_verbose=False)
    parser.set_defaults(apply_macros=True)
    parser.set_defaults(preallocate=True)
    parser.set_defaults(tune_annotated=True)
    parser.set_defaults(tune=True)
    parser.set_defaults(max_types=-1)
    parser.set_defaults(ntests=-1)
    parser.set_defaults(verbose=1)
    parser.set_defaults(resolve_deps=True)
    parser.set_defaults(out_image=None)
    parser.set_defaults(profile=False)
    parser.set_defaults(subprocess=True)
    parser.set_defaults(validate=False)
    parser.set_defaults(validate_images=False)
    parser.set_defaults(validate_speedups=False)
    parser.set_defaults(retime=False)
    parser.set_defaults(use_4channel=False)
    parser.set_defaults(parallel_unpython=True)
    parser.set_defaults(validate_speedups_after=False)
    parser.set_defaults(safe=False)
    parser.set_defaults(vectorize=True)
    parser.set_defaults(run=True)
    
    args = parser.parse_args()
#    if not args.profile:
    main_args(args)

#    else:
#        profile_filename = 'profile'
#        cProfile.runctx('main_args(args)', globals(), locals(), profile_filename)
#        p = pstats.Stats(profile_filename)
#        p.strip_dirs()
#        p.sort_stats('cumulative')
#        p.print_stats(30)

if __name__ == '__main__':
    main()
