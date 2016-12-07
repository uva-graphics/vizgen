
"""
Code transforms.

On the Python input source, transform classes can be applied with the syntax (line numbers shown at left):

    1 #transform(TypeSpecialize([{"n": "int", "i": "int", "ans": "numpy.ndarray[numpy.float64_t, ndim=1]"}]))
    2 def f(n):
    3     ans = numpy.zeros(n)
    4     for i in range(n):
    5         ans[i] += i
    6     return ans
    7
    8 #transform*(Parallel(3))       # Apply transform to given line number in original source

We call the latter transform a line-numbered transform, because it takes a first positional argument of the line number in the original source.

Each transform instantiates an instance of the relevant class in transforms.py and calls its apply() method. Since Cython code is not supported by the parser we use, any output Cython code lines should be wrapped in #cython() comments as follows:

    1 def f(n):
    2     #cython(cdef int n)
    3     #cython(cdef int i)
    4     #cython(numpy.ndarray[numpy.float64_t, ndim=1])
    5     #cython(for i in prange(n)):
    6         ans[i] += i
    7     return ans

Code transforms are applied to an arbitrary program by these two steps:

 1. Convert all transforms to non-line-numbered form by moving them into the body of the code.
 2. Repeatedly pop and apply transforms in order from top to bottom of the program.

In contrast, the stochastic search process used by the tuner does not make any modifications to the input Python source lines, and only uses line-numbered transforms which are placed after the end of the program. This is done to simplify rules that mutate or create transforms.
"""
import json
import string
import inspect
import traceback
import collections
import sys
import pprint

import preprocess

from transforms_util import *
from transforms_base import BaseTransform
from transforms_typespecialize import TypeSpecialize
from transforms_parallel import Parallel
from transforms_loopimplicit import LoopImplicit
from transforms_applymacros import ApplyMacros
from transforms_loopimplicit import VectorizeInnermost
from transforms_loopremoveconditionals import LoopRemoveConditionals
from transforms_loopfusion import LoopFusion
from transforms_arraystorage import ArrayStorage
from transforms_preallocate import *

class Profile(collections.defaultdict):
    def __init__(self):
        collections.defaultdict.__init__(self, lambda: 0.0)
        self.out_files = [sys.stdout]
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        if len(self):
            for out_file in self.out_files:
                max_t = 1.0
                if len(self):
                    max_t = max(self.values())
                print('Profile times [sec]:', file=out_file)
                for d_key in sorted(self, key=lambda current_key: current_key.lower()):
                    print((d_key + ':').ljust(70), '%10.5f'%self[d_key], '(%7.4f%%)'%(self[d_key]*100.0/max_t), file=out_file)
    
    def __del__(self):
        self.close()

profile = Profile()

def get_indentation(line):
    line_strip = line.lstrip()
    nindent = len(line) - len(line_strip)
    indentation = line[:nindent]
    return (nindent, indentation)

def get_next_indentation(lines, i):
    for j in range(i + 1, len(lines)):
        lines_j_strip = lines[j].strip()
        if len(lines_j_strip) and not lines_j_strip.startswith('#'):
            (nindent_j, indentation_j) = get_indentation(lines[j])
            return (j, indentation_j)
    raise ValueError('Could not get next indentation', i)

def finalize_cython(program_info, s):
    """
    Postprocess Python code turning it into Cython code.
    
    Strip special Cython identifiers (cython_str, cython_replace_str), returning Cython code str.
    
    Replace pointer_cast and type_cast with Cython syntax for pointer and type casting.
    
    Also add imports, and fix up Cython bugs.
    """
    if verbose:
        util.print_header('finalize_cython received:', s)
    
    if macros.workaround_cython_546:
        s = macros.fix_range_3_args(s)
    
    T_replace_node = 0.0
    T_rewrite_var = 0.0
    
    T0 = time.time()
    rootnode = py_ast.get_ast(s)
    py_ast.add_parent_info(rootnode)
    
    rewrite_vars = {}
    
    all_nodes = py_ast.find_all(rootnode, (ast.Str, ast.Subscript, ast.Call))
    comment_nodes = [tnode for tnode in all_nodes if isinstance(tnode, ast.Str)]
    
    T1 = time.time()
    lines = s.split('\n')
    
    for commentnode in comment_nodes:
        comment = py_ast.dump_ast(commentnode)
        if comment.startswith(cython_preallocate_intermediate):
            (prefix, varname, hexcode_ordinary, hexcode_float32) = comment.split()[:4]
            try:
                defnode = get_previous_node_func(rootnode, commentnode, ast.FunctionDef)
            except TransformError:
                warnings.warn('Could not extract defnode for preallocate intermediate node:', comment)
                continue
            
            if id(defnode) not in rewrite_vars:
                rewrite_vars.setdefault(id(defnode), get_all_rewrite_vars_py_ast(defnode))
                
            is_rewrite = varname in rewrite_vars[id(defnode)]
            
            if verbose:
                print('commentnode:', commentnode, 'prefix:', prefix, 'varname:', varname, 'hexcode_ordinary:', hexcode_ordinary, 'hexcode_float32:', hexcode_float32, 'is_rewrite:', is_rewrite)
                
            local_types = None
            try:
                local_types = chosen_typespec_loads_py_ast(program_info, defnode)
            except TransformError:
                pass
            
            if local_types is None:                 # In the non-type-specialized function, do nothing
                continue
            
            if is_rewrite:
                commentnode.s = ''
                if verbose:
                    print('  => commentnode after rewrite:', commentnode)
            else:
                try:
                    var_type = local_types[varname]
                except NameError:
                    continue
                
                if var_type.primitive_type() == 'float':
                    hexcode = hexcode_float32
                else:
                    hexcode = hexcode_ordinary
                start_time_replace_node = time.time()
                py_ast.replace_node(rootnode, commentnode, 
                                    py_ast.get_ast(cython_preallocate_intermediate + ' ' + varname + ' ' + hexcode))
                T_replace_node += time.time() - start_time_replace_node
    T2 = time.time()

    # Rewrite a[y,x] to a[y][x] for the variables that were rewritten to C array type
    subscript_nodes = [tnode for tnode in all_nodes if isinstance(tnode, ast.Subscript)]
    for node in subscript_nodes:
        try:
            varname = node.value.id
        except:
            continue
        try:
            defnode = get_previous_node_func(rootnode, node, ast.FunctionDef)
        except TransformError:
            continue

        if id(defnode) not in rewrite_vars:
            rewrite_vars.setdefault(id(defnode), get_all_rewrite_vars_py_ast(defnode))   
        is_rewrite = varname in rewrite_vars[id(defnode)]

        if is_rewrite:
            try:
                #not sure what it means here, is it used to check dictionary?
                do_continue = True
                if hasattr(node.slice, 'value'):
                    if isinstance(node.slice.value, ast.Str):
                        do_continue = False
                    elif isinstance(node.slice.value, ast.Tuple):
                        if isinstance(node.slice.value.elts[0], ast.Str):
                            do_continue = False
                elif hasattr(node.slice, 'dims'):
                    if isinstance(node.slice.dims[0], ast.Str):
                        do_continue = False
            except:
                do_continue = False
            if do_continue:
                if hasattr(node.slice, 'value'):
                    if not hasattr(node.slice.value, 'elts'):
                        args = py_ast.dump_ast(node.slice.value)
                        node_new_str = varname + ''.join('[' + args + ']')
                    else:
                        args = [py_ast.dump_ast(subnode) for subnode in node.slice.value.elts]
                        node_new_str = varname + ''.join('[' + arg + ']' for arg in args)
                else:
                    args = [py_ast.dump_ast(subnode for subnode in node.slice.dims)]
                    node_new_str = varname + ''.join('[' + arg + ']' for arg in args)
                if verbose:
                    print('node before replacement: ', py_ast.dump_ast(node), 'after replacement:', node_new_str)
                start_time_replace_node = time.time()
                py_ast.replace_node(rootnode, node, py_ast.get_ast(node_new_str).body[0].value)
                T_replace_node += time.time() - start_time_replace_node

    T3 = time.time()

    # Replace type_cast and pointer_cast with unique IDs that later get replaced with the Cython casting operation
    cast_d = {}
    
    atom_nodes = [tnode for tnode in all_nodes if isinstance(tnode, ast.Call) or isinstance(tnode, ast.Subscript)]
    for node in atom_nodes:
        #don't know what if hasattr(node, 'name') and isinstance(node.name, redbaron.NameNode): is used to check
        name = None
        try:
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.value.id
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Tuple) and len(node.slice.value.elts) == 2:
                    name = node.value.id
        except:
            name = None
                
        if name is not None:
            if verbose:
                print('finalize_cython, name:', name)
            if name in [pointer_cast, type_cast]:
                current_id = get_unique_id('cast')
                is_ptr = (name == pointer_cast)
                
                if isinstance(node, ast.Call):
                    v0 = py_ast.dump_ast(node.args[0])
                    v1 = py_ast.dump_ast(node.args[1])
                elif isinstance(node, ast.Subscript):
                    v0 = py_ast.dump_ast(node.slice.value.elts[0])
                    v1 = py_ast.dump_ast(node.slice.value.elts[1])
                else:
                    raise ValueError
                
                #rest = ','.join([py_ast.dump_ast(tnode) for tnode in node.args])
                rest = ''
                ptr = '*' if is_ptr else ''
                while current_id in cast_d:
                    current_id = get_unique_id('cast')
                cast_d[current_id] = '(<{} {}>({})){}'.format(v0, ptr, v1, rest)
                start_time_replace_node = time.time()
                py_ast.replace_node(rootnode, node, py_ast.get_ast(current_id).body[0].value)
                T_replace_node += time.time() - start_time_replace_node

    T4 = time.time()

    if verbose:
        util.print_header('finalize_cython, cast_d:', cast_d)
        
    T5 = time.time()
    s = py_ast.dump_ast(rootnode)
    T6 = time.time()

    # Replace cython_preallocate_intermediate with actual preallocation code
    lines = s.split('\n')
    i=0
    while i < len(lines):
        line = lines[i]
        line_strip = lines[i].lstrip()
        if line_strip.startswith(cython_preallocate_intermediate):
            (j, indentation_j) = get_next_indentation(lines, i)
            if verbose:
                print('get_next_indentation: {} => {}, {!r}'.format(i, j, indentation_j))
            (prefix, varname, hexcode) = line_strip.strip().split()[:3]
            
            #if varname doesn't exist in any other part of the code, ignore this comment
            namenodes = py_ast.find_all(rootnode, ast.Name)
            names = [node.id for node in namenodes if node.id == varname]
            if len(names):
                code_block = binascii.unhexlify(hexcode.encode('ascii')).decode('ascii')
                code_blockL = code_block.split('\n')
                lines[i:i+1] = [indentation_j + code_block_line for code_block_line in code_blockL]
            else:
                lines[i] = ''
            
        i += 1
    s = '\n'.join(lines)
    T7 = time.time()

    # Replace var.shape with (<object> var).shape. Works around Cython bug 302 (http://trac.cython.org/ticket/302)
    rootnode = py_ast.get_ast(s)
    py_ast.add_parent_info(rootnode)
    strip_id = get_unique_id('finalize_cython')
    dot_nodes = py_ast.find_all(rootnode, ast.Attribute)
    for dot_node in dot_nodes:
        if isinstance(dot_node.value, ast.Name) and dot_node.attr == 'shape' and (not isinstance(dot_node.parent, ast.Subscript)):
            dot_node.value.id = '(' + strip_id + dot_node.value.id + ')'
    s = py_ast.dump_ast(rootnode)
    s = s.replace(strip_id, '<object>')

    T8 = time.time()

    for (current_id, current_s) in cast_d.items():
        s = s.replace(current_id, current_s)
        
    lines = s.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        line_strip = lines[i].lstrip()
        (nindent, indentation) = get_indentation(lines[i])
        if line_strip.startswith(cython_replace_str):
            lines[i] = ''
            (j, indentation_j) = get_next_indentation(lines, i)
            lines[j] = indentation_j + line[nindent + len(cython_replace_str)+1:].rstrip()[:-1]
            #for j in range(i + 1, len(lines)):
            #    if lines[j].strip() != '':
            #        (nindent_j, indentation_j) = get_indentation(lines[j])
            #        lines[j] = indentation_j + line[nindent + len(cython_replace_str)+1:].rstrip()[:-1]
            #        break
            i = j
        elif line_strip.startswith(cython_str):
            (j, indentation_j) = get_next_indentation(lines, i)
            lines[i] = indentation_j + lines[i][nindent + len(cython_str)+1:].rstrip()[:-1]
        
        i += 1
    s = '\n'.join(lines)
    s = cython_headers + '\n'.join(list(macro_funcs_templated.templated_func.values())) + '\n' + s

    T9 = time.time()

    s = macros.fix_cpointer_in_final_cython(s)
    T10 = time.time()
    if do_profile:
        profile['transforms: finalize_cython_part0'] += T1-T0
        profile['transforms: finalize_cython_part1'] += T2-T1
        profile['transforms: finalize_cython_part2'] += T3-T2
        profile['transforms: finalize_cython_part3'] += T4-T3
        profile['transforms: finalize_cython_part4'] += T5-T4
        profile['transforms: finalize_cython_part5'] += T6-T5
        profile['transforms: finalize_cython_part6'] += T7-T6
        profile['transforms: finalize_cython_part7'] += T8-T7
        profile['transforms: finalize_cython_part8'] += T9-T8
        profile['transforms: finalize_cython_part9'] += T10-T9
        profile['transforms: finalize_cython: replace_node'] += T_replace_node
    if verbose:
        util.print_header('finalize_cython returned:', s)
    return s
            

def fix_comment_indentation(s):
    """
    Fixes indentation of #cython() code lines to match surrounding indentation.
    """
    is_tab = '\t' in s
    indent_char = '\t' if is_tab else ' '
    lines = s.split('\n')
    for (i, line) in enumerate(lines):
        if line.strip().startswith(cython_str):
            lines[i] = indent_char * indentation + line.lstrip()
        else:
            indentation = len(line) - len(line.lstrip())
            line_no_comment = line
            if '#' in line_no_comment:
                line_no_comment = line_no_comment[:line_no_comment.index('#')]
            if line_no_comment.strip().endswith(':'):
                for ip in range(i+1, len(lines)):
                    lines_ip_strip = lines[ip].strip()
                    if not lines_ip_strip.startswith('#') and len(lines_ip_strip):
                        indentation = len(lines[ip]) - len(lines[ip].lstrip())
                        break

    # Work around redbaron bug where insert_before() will break a line previous to it if it contains a semicolon (e.g. 'a; b')
    for i in range(len(lines)):
        if lines[i].strip() == ';':
            lines[i] = ''

    i = 0
    while i < len(lines)-1:
        if lines[i].strip().startswith(cython_str) and lines[i+1].strip() == '':
            del lines[i+1]
        i += 1

    return '\n'.join(lines)

def parse_transforms(program_info, s, apply_order=False):
    """
    Given Python source s, return list of transform instances in the order they were encountered when parsing lines.
    
    If apply_order is True then instead return them in order that they should be applied (based on their position in all_transforms).
    If a transform is not present in all_transforms then it is applied last.
    """
    """Same as parse_transforms, but using py_ast

    Args:
        program_info, ProgramInfo, the program info
        s, String, program source-code
        apply_order, Boolean, whether or not to return the list in the order 
            that the transforms were applied
    """

    if verbose:
        util.print_header('parse input:')
        print(s)

    program_info.s_current = s
    result = []
    transforms = py_ast.get_all_transform_comments_str(s)

    if verbose:
        print("Transform list:")
        print(", ".join([py_ast.dump_ast(t) for t in transforms]))    

    # now, unpack the transform from its string representation:
    for i in range(len(transforms)):
        full_comment_str = py_ast.dump_ast(transforms[i]).strip()

        # chop off '#transform' from string:
        comment_str = full_comment_str[full_comment_str.index('('):]

        # remove any trailing comment:
        if '#' in comment_str:
            comment_str = comment_str[:comment_str.index('#')]
            comment_str = comment_str.strip()

        assert comment_str[0] == '('
        assert comment_str[-1] == ')'

        comment_str = comment_str[1:-1]

        try:
            start_i = comment_str.index('(')
            end_i = comment_str.rindex(')')

            if start_i == end_i:
                raise ValueError

        except ValueError:
            raise ValueError('Error parsing transform %s' % comment_str)

        clss_name = comment_str[:start_i]

        try:
            clss = globals()[clss_name]
        except KeyError:
            raise ValueError('Could not find class %s' % clss_name)

        transform_args = comment_str[(start_i + 1):end_i]

        if not full_comment_str.startswith(transform_prefix_lineno):
            transform_args = "%d,%s" % (transforms[i].lineno - i, transform_args)
            #argList = transform_args.split(',')
            #print(str(get_orig_line_from_s_orig(program_info.s_orig, (int)(argList[0])))) 
            #transform_args = str(get_orig_line_from_s_orig(program_info.s_orig, (int)(argList[0])))+ ',' + transform_args       

        transform_args = 'program_info,' + transform_args

        if verbose:
            print('On line %d constructing %s %s' % (transforms[i].lineno, clss, 
                transform_args))
        """
        print("-----------")
        print(full_comment_str)
        print(transform_prefix_lineno)
        print(clss_name)
        print(transform_args)
        print("-----------")
        """
        transform = eval("%s(%s)" % (clss_name, transform_args))
        transform.annotated_line = transforms[i].lineno             # .annotated_line stores the line number encountered in the annotated input source string (s_current).
        result.append(transform)

    if verbose:
        print("parse transforms:")
        print(result)

    if apply_order:
        def sort_key(transform):
            transform_class = transform.__class__

            try:
                return (all_transforms.index(transform_class), 0)
            except ValueError:
                return (len(all_transforms), transform.line)

        result = sorted(result, key=sort_key)
    else:
        result = sorted(result, key=lambda transform: transform.line)

    program_info.s_current = None

    return result

def unparse_transforms(program_info, s_orig, transformL, end=True):
    """
    Given list of transforms, put them back in the original Python source (which has no transform annotations).
    
    If end is True, puts them at the end of the source, with line numbers. If False, puts them in the body without line numbers.
    """

    assert hasattr(s_orig, 'upper')
    if end:
        ans = s_orig + '\n' + '\n'.join(transform_str + '*(' + repr(x) + ')' for x in transformL)
    else:
        lines = s_orig.split('\n')
        for transform in transformL:
            line = transform.line-1
            extra_s = '     # applied to original line ' + str(transform.orig_num) if debug_extra_linenos else ''
            lines[line] = transform_str + '(' + transform.repr_no_line_number() + ')' + extra_s + '\n' + lines[line]
        ans = '\n'.join(lines)
    if verbose:
        util.print_header('unparse_transforms, input transforms:', transformL)
        util.print_header('unparse_transforms, input program:', s_orig, linenos=True)
        util.print_header('unparse_transforms, result:', ans, linenos=True)
    return ans

def move_transforms_to_body(program_info, s, s_orig):
    """
    Move transforms into the body of the code, without line numbers. Here s is source with transforms, and s_orig is original source.
    """
    return unparse_transforms(program_info, s_orig, parse_transforms(program_info, s), False)

def move_transforms_to_end(program_info, s, s_orig):
    """
    Move transforms to the end of the code, with line numbers. Here s is source with transforms, and s_orig is original source.
    """
    return unparse_transforms(program_info, s_orig, parse_transforms(program_info, s), True)

def str_transform_list(L):
    """
    Convert transform list into str() form (shorter, human-readable form).
    """
    return [str(x) for x in L]

def unparse_type(type_str):
        """
        parse type_str to annotations
        """
        if not type_str.startswith('array'):
            return type_str
        arg_dim = type_str.lstrip('array')[0]
        data_type = type_str.lstrip('array')[1:]
        arg_type = "vizgen.ndarray('" + data_type + "', " + arg_dim + ")"
        return arg_type

def resolve_dependencies(transformL, add_typespecialize=True, randomize=True, reverse=False):
    """
    Returns a modified list of transforms with all dependencies resolved (this potentially adds new transforms to the returned list).
    
    Existing transforms will not be modified. If randomize is True then randomize the resolved dependencies: either keep the order
    as it is originally, reverse the order, or shuffle the order. If reverse is True then reverse the order of resolved dependencies.
    """
    ans = []
    seen = []
    remain = list(transformL)
    random.shuffle(remain)
    is_randomized = random.random() <= (1.0/3.0)
    forward = (random.randrange(2) if not reverse else 0)
#    print()
#    util.print_header('begin resolve_dependencies, is_randomized: {}, forward: {}'.format(is_randomized, forward))
    while len(remain):
#        print('resolve_dependencies, remain:', str_transform_list(remain))
#        print('resolve_dependencies, ans:', str_transform_list(ans))
        transform = remain.pop(0)
        #if transform.line in seen:
#            print()
        #    continue
        found = False
        for transform_p in seen:
            if transform == transform_p:
                found = True
                break
        if found:
            continue
        if isinstance(transform, TypeSpecialize) and not add_typespecialize:
#            print()
            continue
        transform = copy.deepcopy(transform)
        seen.append(transform)
        ans.append(transform)
        deps_start_time = time.time()
        try:
            deps = transform.dependencies()
        finally:
            if transform.__class__.dependencies != BaseTransform.dependencies:
                if do_profile:
                    profile['ProgramInfo: tune: resolve dependencies: {}'.format(transform.__class__.__name__)] += time.time() - deps_start_time
#        print('resolve_dependencies, deps:', str_transform_list(deps))
        if randomize or reverse:
            #r = random.Random(seed)
            #r.shuffle(deps)
            if is_randomized and not reverse:
                random.shuffle(deps)
            else:
                if not forward or reverse:
                    deps = deps[::-1]
#            print('resolve_dependencies, deps after reorder:', str_transform_list(deps))
        if len(deps):
            remain.extend(deps)
            #random.shuffle(remain)     # Do not re-randomize list. This prevents existing transforms from being modified.
#        print()
    ans.sort(key=lambda transform: (transform.line, repr(transform.__class__)))
    return ans

def get_previous_node_func(rootnode, tnode, cls):
    """
    Helper function to get previous node matching string name nodename from RedBaron instance r, before int lineno (or raise TransformError).
    """
    """
    A faster alternative to get_previous_node_func_py_ast2, with different semantics: find parent node of given class
    """
    if not hasattr(tnode, 'parent'):
        py_ast.add_parent_info(rootnode)
    parent = tnode.parent
    while parent is not None:
        if isinstance(parent, cls):
            return parent
        parent = parent.parent
    raise TransformError('could not get previous node')

def get_all_rewrite_vars_py_ast(defnode):
    rewrite_vars = []
    defnode_s = py_ast.dump_ast(defnode)
    for line in defnode_s.split('\n'):
        line = line.lstrip()
        if line.startswith(cython_c_rewrite + '('):
            var_start = len(cython_c_rewrite + '(')
            var_end = line.index(')')
            varname_line = line[var_start:var_end].strip()
            rewrite_vars.append(varname_line)
    return rewrite_vars

# List of all transforms that should be searched over, and their application order. This can be modified by the compiler for e.g. comparisons.
# Excludes special transformation classes ApplyMacros and Preallocate which are always applied, unless specifically disabled.
all_transforms = [ArrayStorage, TypeSpecialize, Parallel, LoopImplicit, LoopRemoveConditionals, VectorizeInnermost]

if enable_loop_fusion:
    all_transforms = [LoopFusion] + all_transforms

# Original list of all transforms to be searched over, before modification by compiler.
orig_all_transforms = list(all_transforms)

# Transform string names in reverse dependency order and then alphabetical by their name in the paper if there are ties
transforms_reverse_dep_order = 'Parallel VectorizeInnermost ArrayStorage ApplyMacros LoopImplicit Preallocate LoopRemoveConditionals TypeSpecialize'.split()

transforms_reverse_dep_order_descriptions = ['Parallelize loop', 'Vectorize innermost variable', 'Array storage alteration', 'API call rewriting', 'Loop over implicit variables', 'Preallocate arrays', 'Remove loop conditionals', 'Type specialization']

if enable_loop_fusion:
    transforms_reverse_dep_order = transforms_reverse_dep_order[:-2] + ['LoopFusion'] + transforms_reverse_dep_order[-1:]
    transforms_reverse_dep_order_descriptions = transforms_reverse_dep_order_descriptions[:-2] + ['Fuse different stages'] + transforms_reverse_dep_order_descriptions[-1:]
