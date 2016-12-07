import ast
import py_ast
import copy
import util
import random
import time
import macros
import astor
import z3
import binascii
import hashlib
import numpy
import z3_util
import macro_funcs_templated

verbose = False                             # Whether this entire module is verbose. Set by compiler.py --verbose level, where level >= 3.

enable_loop_fusion  = False                  # Enable unfinished loop fusion transformation
enable_preallocate  = True                   # Whether to allow preallocation
enable_cache        = True                   # Enable caching of transform apply() method
cache_vectorize_innermost = True             # Enable caching for VectorizeInnermost class
do_profile = False                           # Print profiling information

##### py_ast flags: #####
use_py_ast_parallel_preallocate = True
use_py_ast_preallocate_init = True
use_py_ast_preallocate_grow_shrink = True

annotate_type_signature = True
accept_annotated_type = False

cython_str = '#cython'
cython_replace_str = '#cython_replace'
cython_c_rewrite = '#c_rewrite_cython'
cython_preallocate_intermediate = '#preallocate_intermediate_cython'
transform_str = '#transform'
openmp_add_private_str = '#openmp_add_private'
transform_prefix_paren = transform_str + '('
transform_prefix_lineno = transform_str + '*('
transform_suffix = ')'

pointer_cast = '_pointer_cast'
type_cast = '_type_cast'

typespecialize_header = '_0typespec_'
typespecialize_trailer = '_typespec_'
loopimplicit_trailer = '_loopimplicit_'
chosen_typespec = '#_chosen_typespecialize '
after_cdef_trailer = '#_after_cdefs_'

preallocate_prefix = '_prealloc_'
unknown_type_prefix = '_unknown_type'
fusion_prefix = '_fusion_'

debug_extra_linenos = True

python_headers = """
import util
import numpy
""" + '\n'

cython_imports = """
cimport numpy
cimport numpy as np
import numpy
cimport cython
cimport cython.parallel
from libcpp cimport bool as _cbool
""" + '\n' + python_headers

cython_headers = '\n' + open('macro_funcs.pyx', 'rt').read() + '\n'

parallel_mutate_tries = 20

all_transform_cache = {}
transform_cache = {}

def get_verbose():
    return verbose

def chosen_typespec_loads_py_ast(program_info, defnode):
    """
    Return chosen type specialization for a given redbaron.DefNode instance, assuming TypeSpecialize() has already been applied to the code.
    """
    """Same as chosen_typespec_loads, but for py_ast AST's
    """
    start_time = time.time()
    try:

        for comment_node in py_ast.get_comment_strings(defnode):
            comment_node_str = py_ast.dump_ast(comment_node).lstrip(" ")

            if comment_node_str.startswith(chosen_typespec):
                # chop off chose_typespec header:
                dict_str = comment_node_str[len(chosen_typespec):]
                type_dict = eval(dict_str)

                if type_dict is not None:
                    if verbose:
                        print('chosen_typespec_loads before conversion to CythonType:',
                            type_dict)

                    # convert types to CythonTypes
                    type_dict = {
                        key: util.CythonType.from_cython_type(value, program_info) \
                             for (key, value) in type_dict.items()
                    }

                    if verbose:
                        print('chosen_typespec_loads after conversion to CythonType:', 
                            type_dict)

                return type_dict
    finally:
        if do_profile:
            profile['transforms: chosen_typespec_loads_py_ast'] += time.time() - start_time
    
    raise TransformError(
        'did not find special string chosen_typespec={}'.format(
            chosen_typespec))

def chosen_typespec_dumps(typeconfig):
    """
    Add a comment about the chosen type specialization that was used for a given function.
    """
    ans = chosen_typespec + repr(typeconfig) + '\n'
    if typeconfig is not None and verbose:
        print('chosen_typespec_dumps:', ans)
        print('chosen_typespec_dumps, checking')
        for value in typeconfig.values():
            print('chosen_typespec_dumps, check', type(value), repr(value))
    #        value.check()
    return ans

def get_orig_line_from_s_orig(s_orig, line_no):
    if(line_no == None):
        return -1
    node = py_ast.get_ast(s_orig)
    nodeList = [i for i in ast.walk(node) if (hasattr(i, 'lineno') and hasattr(i, 'orig_lineno') and i.lineno == line_no)]
    if(len(nodeList) == 0):
        #print("------ get_orig_line_from_s_orig begin--------")
        #print(s_orig)
        #print(line_no)
        #print("------ get_orig_line_from_s_orig end--------")
        #print("______________")
        #print("cannot find lineno")
        #print("______________")
        #node = preprocess.add_str_node(node)
        #nodeList2 = [i for i in ast.walk(node) if (hasattr(i, 'lineno') and hasattr(i, 'orig_lineno') and i.lineno == line_no)]
        #if(len(nodeList2) == 0):
        return line_no
    
    return nodeList[0].orig_lineno

def repr_sort_keys(v, suppress_comma=False):
    if isinstance(v, (list, tuple)):
        extra_comma = ',' if len(v) == 1 else ''
        return '(' + ','.join([repr_sort_keys(x) for x in v]) + (extra_comma if not suppress_comma else '') + ')'
    elif isinstance(v, dict):
        return '{' + ','.join([repr(key) + ':' + repr(value) for (key, value) in sorted(v.items())]) + '}'
    else:
        return repr(v)

def transform_repr(self, args):
    """
    Returns repr() for a transformation that takes positional arguments 'args'.
    """
    return '{}({})'.format(self.__class__.__name__, repr_sort_keys(tuple(args), True)[1:-1])
#    return '{}({})'.format(self.__class__.__name__, json.dumps(tuple(args), sort_keys=True)[1:-1])

def is_array_rewrite_var_py_ast(defnode, varname):
    """
    Bool for whether given variable name (str) in given redbaron.DefNode defnode has been rewritten into a C array data structure.
    """
    varname = varname.strip()
    defnode_s = py_ast.dump_ast(defnode)
    for line in defnode_s.split('\n'):
        line = line.lstrip()
        if line.startswith(cython_c_rewrite + '('):
            var_start = len(cython_c_rewrite + '(')
            var_end = line.index(')')
            varname_line = line[var_start:var_end].strip()
            if varname_line == varname:
                return True
    return False

class TransformError(Exception):
    """
    Base error class: failed to apply or mutate a transform.
    """

class MutateError(TransformError):
    """
    Could not mutate transform.
    """
    
class WrongMacroArgsError(Exception):
    pass

class ArrayParseFailed(TransformError):
    pass

scalar_macros = [
    'numpy_clip_double',
    'numpy_clip_int',
    'square_double',
    'square_int',
    'libc.math.sqrt',
    'float_to_int',
    'int_to_int',
    'float_to_float',
    'int_to_float',
    'libc.stdlib.abs',
    'libc.math.fabs',
    'numpy_linalg_norm_vec3f_ptr',
    'numpy_linalg_norm_vec3g_ptr',
    'randrange_1arg',
    'randrange_2arg',
    'libc.math.pow']

class LoopRemoveConditionalsProofFailed(Exception):
    pass

def rewrite_expr_z3_py_ast(r, is_py_ast=True):
    if verbose and False:
        print('rewrite_expr_z3_py_ast:', r)
    # Rewrites py_ast expression to a str expression that could be used in z3
    # Return (z3_expr_str, z3_varnames)
    z3_expr_str = (py_ast.dump_ast(r) if is_py_ast else r).strip()
    z3_expr_str = z3_expr_str.replace('.', '_').replace('[', '_').replace(']', '_')
    rp = py_ast.get_ast(z3_expr_str).body[0].value
    nodes = py_ast.find_all(rp, ast.UnaryOp)
    while nodes != []:
        node = nodes[0]
        if isinstance(node.op, ast.Not):
            if rp == node:
                rp = py_ast.get_ast('z3.Not(' + py_ast.dump_ast(node.operand) + ')').body[0].value
            else:
                py_ast.replace_node(rp, node, py_ast.get_ast('z3.Not(' + py_ast.dump_ast(node.operand) + ')').body[0].value)
            nodes = py_ast.find_all(rp, ast.UnaryOp)
        else:
            nodes = nodes[1:]
    
    nodes = py_ast.find_all(rp, ast.BoolOp)
    while nodes != []:
        node = nodes[0]
        if isinstance(node.op, ast.And):
            rp_str = 'z3.And('
            for value in node.values:
                rp_str += py_ast.dump_ast(value) + ', '
            rp_str = rp_str.rstrip(', ')
            rp_str += ')'
            if rp == node:
                rp = py_ast.get_ast(rp_str).body[0].value
            else:
                py_ast.replace_node(rp, node, py_ast.get_ast(rp_str).body[0].value)
        elif isinstance(node.op, ast.Or):
            rp_str = 'z3.Or('
            for value in node.values:
                rp_str += py_ast.dump_ast(value) + ', '
            rp_str = rp_str.rstrip(', ')
            rp_str += ')'
            if rp == node:
                rp = py_ast.get_ast(rp_str).body[0].value
            else:
                py_ast.replace_node(rp, node, py_ast.get_ast(rp_str).body[0].value)
        nodes = py_ast.find_all(rp, ast.BoolOp)
        
    z3_expr_str = py_ast.dump_ast(rp)
            
    z3_vars = set()
    for node in py_ast.find_all(rp, ast.Name):
        z3_vars.add(node.id)
    if 'z3' in z3_vars:
        z3_vars.remove('z3')
    return (z3_expr_str, z3_vars)

fusion_info_cache = {}

def loop_fusion_program_analysis(rootnode):
    """
    analyze on the given code s about loop fusion
    1. find array nodes
    2. for each array, check whether each pixel is only assigned once
    """
    
    class find_array_visitor(ast.NodeVisitor):
        def __init__(self):
            self.arrays = {}
            self.current_def = None
        
        def visit_FunctionDef(self, node):
            self.arrays[node.name] = []
            self.current_def = node.name
            self.generic_visit(node)
        
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name):
                if node.value.id not in self.arrays[self.current_def]:
                    self.arrays[self.current_def].append(node.value.id)
                    self.generic_visit(node)
    
    array_visitor = find_array_visitor()
    array_visitor.visit(rootnode)
    arrays = array_visitor.arrays
    
    class loop_fusion_visitor(ast.NodeVisitor):
        def __init__(self):
            self.fusion_info = {}
            self.current_def = None
            self.current_defnode = None
        
        def visit_FunctionDef(self, node):
            self.fusion_info[node.name] = {}
            self.current_def = node.name
            self.current_defnode = node
            self.generic_visit(node)
        
        def visit_Assign(self, node):
            targets = node.targets
            array_name = None
            array_idx = None
            array_tuple = None
            if self.current_def is None:
                return
            if len(targets) == 1:
                if isinstance(targets[0], ast.Name):
                    if targets[0].id in arrays[self.current_def]:
                        array_name = targets[0].id
                        array_idx = []
                        array_tuple = []
                        if array_name not in self.fusion_info[self.current_def]:
                            self.fusion_info[self.current_def][array_name] = {'shape': None, 'dims': None, 'idx': []}
                elif isinstance(targets[0], ast.Subscript):
                    if isinstance(targets[0].value, ast.Name) and targets[0].value.id in arrays[self.current_def]:
                        (getitem_tuple, getitem_strs) = parse_array_slice_py_ast(targets[0])
                        array_name = targets[0].value.id
                        array_idx = getitem_strs
                        array_tuple = getitem_tuple
                        if array_idx == [':']:
                            array_idx = []
                            array_tuple = []
                        if array_name not in self.fusion_info[self.current_def]:
                            self.fusion_info[self.current_def][array_name] = {'shape': None, 'dims': None, 'idx': []}
            
            if array_name is not None:
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        assign_modname = node.value.func.value.id
                        assign_funcname = node.value.func.attr
                        if assign_modname in macros.numpy_modules and assign_funcname in preallocate_numpy_funcs + ['ones']:
                            if self.fusion_info[self.current_def][array_name]['shape'] is not None:
                                """
                                if it's declared using numpy functions more than once, remove the array from fusion_info
                                """
                                del self.fusion_info[self.current_def][array_name]
                                del arrays[array_name]
                                return
                            elif array_idx == []:
                                call_args = py_ast.dump_ast(node.value.args[0])
                                if call_args.endswith('\n'):
                                    call_args = call_args[0 : -1]
                                dimensions = parse_call_args(self.current_defnode, array_name, call_args, [])
                                self.fusion_info[self.current_def][array_name]['dims'] = dimensions
                                shape = []
                                for i in range(len(dimensions)):
                                    for j in range(len(dimensions[i])):
                                        if j < len(shape):
                                            if not isinstance(shape[j], int) and py_ast.is_int_constant_py_ast(dimensions[i][j]):
                                                shape[j] = eval(dimensions[i][j])
                                        else:
                                            if py_ast.is_int_constant_py_ast(dimensions[i][j]):
                                                shape.append(eval(dimensions[i][j]))
                                            else:
                                                shape.append(dimensions[i][j])
                                self.fusion_info[self.current_def][array_name]['shape'] = shape
                                return
                    
                namenodes = py_ast.find_all(node, ast.Name)
                names = []
                for namenode in namenodes:
                    if namenode.id not in names:
                        names.append(namenode.id)
                    
                iter_lookup = {}
                if_conditions = []
                parent = node
                while parent.parent is not None and parent.parent != self.current_defnode:
                    if isinstance(parent.parent, ast.For):
                        targetnode = parent.parent.target
                        if isinstance(targetnode, ast.Name):
                            if targetnode.id in names:
                                iter_lookup[targetnode.id] = parent.parent.iter
                    if isinstance(parent.parent, ast.If):
                        if parent in parent.parent.orelse:
                            if_conditions.append('not ' + py_ast.dump_ast(parent.parent.test))
                        else:
                            if_conditions.append(py_ast.dump_ast(parent.parent.test))
                    parent = parent.parent
                if_condition = ' and '.join(if_conditions)
                self.fusion_info[self.current_def][array_name]['idx'].append((array_idx, array_tuple, node.value, iter_lookup, if_condition, None))
                    
        def visit_AugAssign(self, node):
            target = node.target
            array_name = None
            array_idx = None
            array_tuple = None
            if isinstance(target, ast.Name):
                if target.id in arrays[self.current_def]:
                    array_name = target.id
                    array_idx = []
                    array_tuple = []
                    if array_name not in self.fusion_info[self.current_def]:
                        self.fusion_info[self.current_def][array_name] = {'shape': None, 'dims': None, 'idx': []}
            elif isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Name) and target.value.id in arrays[self.current_def]:
                    (getitem_tuple, getitem_strs) = parse_array_slice_py_ast(target)
                    array_name = target.value.id
                    array_idx = getitem_strs
                    array_tuple = getitem_tuple
                    if array_idx == [':']:
                        array_idx = []
                        array_tuple = []
                    if array_name not in self.fusion_info[self.current_def]:
                        self.fusion_info[self.current_def][array_name] = {'shape': None, 'dims': None, 'idx': []}
            
            if array_name is not None:
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        assign_modname = node.value.func.value.id
                        assign_funcname = node.value.func.attr
                        if assign_modname in macros.numpy_modules and assign_funcname in preallocate_numpy_funcs + ['ones']:
                            if self.fusion_info[self.current_def][array_name]['shape'] is not None:
                                """
                                if it's declared using numpy functions more than once, remove the array from fusion_info
                                """
                                del self.fusion_info[self.current_def][array_name]
                                del arrays[array_name]
                                return
                            elif array_idx == []:
                                call_args = py_ast.dump_ast(node.value.args[0])
                                if call_args.endswith('\n'):
                                    call_args = call_args[0 : -1]
                                dimensions = parse_call_args(self.current_defnode, array_name, call_args, [])
                                self.fusion_info[self.current_def][array_name]['dims'] = dimensions
                                shape = []
                                for i in range(len(dimensions)):
                                    for j in range(len(dimensions[i])):
                                        if j < len(shape):
                                            if not isinstance(shape[j], int) and py_ast.is_int_constant_py_ast(dimensions[i][j]):
                                                shape[j] = eval(dimensions[i][j])
                                        else:
                                            if py_ast.is_int_constant_py_ast(dimensions[i][j]):
                                                shape.append(eval(dimensions[i][j]))
                                            else:
                                                shape.append(dimensions[i][j])
                                self.fusion_info[self.current_def][array_name]['shape'] = shape
                                return
                    
                namenodes = py_ast.find_all(node, ast.Name)
                names = []
                for namenode in namenodes:
                    if namenode.id not in names:
                        names.append(namenode.id)
                    
                iter_lookup = {}
                if_conditions = []
                parent = node
                while parent.parent is not None and parent.parent != self.current_defnode:
                    if isinstance(parent.parent, ast.For):
                        targetnode = parent.parent.target
                        if isinstance(targetnode, ast.Name):
                            if targetnode.id in names:
                                iter_lookup[targetnode.id] = parent.parent.iter
                    if isinstance(parent.parent, ast.If):
                        if parent in parent.parent.orelse:
                            if_conditions.append('not ' + py_ast.dump_ast(parent.parent.test))
                        else:
                            if_conditions.append(py_ast.dump_ast(parent.parent.test))
                    parent = parent.parent
                if_condition = ' and '.join(if_conditions)
                self.fusion_info[self.current_def][array_name]['idx'].append((array_idx, array_tuple, node.value, iter_lookup, if_condition, node.op))
                   
    fusion_visitor = loop_fusion_visitor()
    fusion_visitor.visit(rootnode)
    return fusion_visitor.fusion_info

class PreallocateNotFound(Exception):
    pass

preallocate_numpy_funcs = ['zeros', 'empty', 'zeros_like', 'empty_like']
preallocate_numpy_funcs_array_arg = ['zeros_like', 'empty_like']
preallocate_empty_funcs = ['empty', 'empty_like']

def parse_call_args(defnode, array_var, call_args, prev_dimensions):
    """
    try split call_args into multiple dimensions
    """
    
    if isinstance(py_ast.get_ast(call_args).body[0].value, (ast.List, ast.Tuple)):
        call_args = call_args[1 : -1]
        dimensions = call_args.split(',')
        for i in range(len(dimensions)):
            dimension = dimensions[i]
            try:
                d_node = py_ast.get_ast(dimension.strip()).body[0].value
                if isinstance(d_node, ast.BinOp):
                    if isinstance(d_node.op, (ast.Add, ast.Sub)):
                        if isinstance(d_node.left, ast.Num) and not isinstance(d_node.right, ast.Num):
                            dimensions[i] = py_ast.dump_ast(d_node.right)
                        elif isinstance(d_node.right, ast.Num) and not isinstance(d_node.left, ast.Num):
                            dimensions[i] = py_ast.dump_ast(d_node.left)
                        else:
                            dimensions[i] = py_ast.dump_ast(d_node)
                    else:
                        dimensions[i] = py_ast.dump_ast(d_node)
                else:
                    dimensions[i] = py_ast.dump_ast(d_node)
            except:
                pass
        prev_dimensions.append(dimensions)
    else:
        arg_node = py_ast.get_ast(call_args).body[0].value
        if isinstance(arg_node, ast.Attribute):
            if arg_node.attr == 'shape':
                arg_shape_var = py_ast.dump_ast(arg_node.value)
                try:
                    (shape_assignnode, shape_call_args, shape_assign_funcname) = preallocate_find_assignnode_py_ast(defnode, defnode.name, arg_shape_var)
                    prev_dimensions.append([call_args])
                    prev_dimensions = parse_call_args(defnode, arg_shape_var, shape_call_args, prev_dimensions)
                except PreallocateNotFound:
                    dimensions = [call_args]
                    prev_dimensions.append(dimensions)
        else:
            dimensions = [call_args]
            prev_dimensions.append(dimensions)
    return prev_dimensions

def preallocate_find_assignnode_simple_py_ast(r, func_name, array_var):
    """
    Find (assignnode, callnode, numpy_func) in given function name for given array variable string, or raise PreallocateNotFound.
    also accepts numpy.ones
    """
    defvisitor = py_ast.FindDefVisitor(func_name)
    defvisitor.visit(r)
    defnodeL = defvisitor.defnode
    if len(defnodeL) == 0:
        warnings.warn('preallocate_find_arrays: found 0 copies of defnode for function {}, skipping'.format(func_name))
        raise PreallocateNotFound
    defnode = defnodeL[-1]
    
    assignvisitor = py_ast.FindAssignVisitor()
    assignvisitor.visit(defnode)
    assignnodes = assignvisitor.assignnode
    
    for assignnode in assignnodes:
        if len(assignnode.targets) == 1:
            if isinstance(assignnode.targets[0], ast.Name) and assignnode.targets[0].id == array_var:
                if isinstance(assignnode.value, ast.Call):
                    if isinstance(assignnode.value.func, ast.Attribute):
                        assign_modname = assignnode.value.func.value.id
                        assign_funcname = assignnode.value.func.attr
                        if assign_modname in macros.numpy_modules and assign_funcname in preallocate_numpy_funcs + ['ones']:
                            call_args = py_ast.dump_ast(assignnode.value.args[0])
                            #call_args = ''
                            #for i in range(len(assignnode.value.args)):
                                #call_args = call_args + py_ast.dump_ast(assignnode.value.args[i])
                                #call_args = call_args + ','
                            if call_args.endswith('\n'):
                                call_args = call_args[0 : -1]
                            return (assignnode, call_args, assign_funcname)
    raise PreallocateNotFound

def preallocate_find_assignnode_py_ast(r, func_name, array_var):
    """
    Find (assignnode, callnode, numpy_func) in given function name for given array variable string, or raise PreallocateNotFound.
    """
    defvisitor = py_ast.FindDefVisitor(func_name)
    defvisitor.visit(r)
    defnodeL = defvisitor.defnode
    if len(defnodeL) == 0:
        warnings.warn('preallocate_find_arrays: found 0 copies of defnode for function {}, skipping'.format(func_name))
        raise PreallocateNotFound
    defnode = defnodeL[-1]
    
    assignvisitor = py_ast.FindAssignVisitor()
    assignvisitor.visit(defnode)
    assignnodes = assignvisitor.assignnode
    
    for assignnode in assignnodes:
        if len(assignnode.targets) == 1:
            if isinstance(assignnode.targets[0], ast.Name) and assignnode.targets[0].id == array_var:
                if isinstance(assignnode.value, ast.Call):
                    if isinstance(assignnode.value.func, ast.Attribute):
                        assign_modname = assignnode.value.func.value.id
                        assign_funcname = assignnode.value.func.attr
                        if assign_modname in macros.numpy_modules and assign_funcname in preallocate_numpy_funcs:
                            call_args = py_ast.dump_ast(assignnode.value.args[0])
                            #call_args = ''
                            #for i in range(len(assignnode.value.args)):
                                #call_args = call_args + py_ast.dump_ast(assignnode.value.args[i])
                                #call_args = call_args + ','
                            if call_args.endswith('\n'):
                                call_args = call_args[0 : -1]
                            return (assignnode, call_args, assign_funcname)
    raise PreallocateNotFound

def parse_array_slice_py_ast(array_node):
        
    if hasattr(array_node.slice, 'value'):
        getitem_tuple = array_node.slice.value
    elif hasattr(array_node.slice, 'dims'):
        getitem_tuple = array_node.slice.dims
    elif isinstance(array_node.slice, ast.Slice):
        getitem_tuple = array_node.slice
    if hasattr(getitem_tuple, 'elts'):
        getitem_tuple = getitem_tuple.elts
    if isinstance(getitem_tuple, ast.Name) or isinstance(getitem_tuple, ast.Slice) or isinstance(getitem_tuple, ast.Num):
        getitem_tuple = [getitem_tuple]
    try:
        getitem_strs = [py_ast.dump_ast(x) for x in getitem_tuple]
    except:
        getitem_strs = [py_ast.dump_ast(x) for x in getitem_tuple.elts] 
    
    return (getitem_tuple, getitem_strs)

def is_before(node1, node2):
    """
    checks if definately appears before node2
    """
    
    current = node2
    
    while current is not None:
        try:
            if current.parent == node1.parent:
                for field, value in ast.iter_fields(node1.parent):
                    if value == current or value == node1:
                        return False
                    elif isinstance(value, list) and current in value and node1 in value:
                        list_index1 = value.index(node1)
                        list_index2 = value.index(current)
                        if list_index2 > list_index1:
                            return True
        except:
            pass
        
        current = current.parent
    return False

def is_wider(area, namenode):
    """
    checks if the area initialized contains the area indexed by namenode.parent
    """
    if area == []:
        return True
    return False

def replace_scalar_assign(defnode, namenode):
    """
    search in defnode to see if namenode assigned as scalar before
    if it's true, return the assigned str
    """
    potential_assignnodes = py_ast.find_all(defnode, ast.Assign)
    assignnodes = []
    for potential_assignnode in potential_assignnodes:
        try:
            if len(potential_assignnode.targets) == 1:
                if isinstance(potential_assignnode.targets[0], ast.Name):
                    if potential_assignnode.targets[0].id == namenode.id:
                        if is_before(potential_assignnode, namenode):
                            assignnodes.append(potential_assignnode)
        except:
            pass
    if len(assignnodes) == 1:
        return py_ast.dump_ast(assignnodes[0].value)
    else:
        raise ValueError
    
def find_parent_field(parent, node):
    """
    assume parent is the ancestor of node, return the field that derives to node in parent
    """
    current = node
    while current.parent != parent:
        current = current.parent
    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            if current in value:
                return field
        elif current == value:
            return field
    return None

def fix_comment_indentation_no_deletion(s):
    """
    Fixes indentation of #cython() code lines to match surrounding indentation.
    """
    if get_verbose():
        util.print_header("Fixing comment indentation")
        print("Input string:")
        print(s)

    is_tab = '\t' in s
    indent_char = '\t' if is_tab else ' '
    lines = s.split('\n')
    for (i, line) in enumerate(lines):
        if line.strip().startswith('#') and not line.strip().startswith(transform_str):
            try:
                lines[i] = indent_char * indentation + line.lstrip()
            except:
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
        else:
            if len(line):
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
    
    return '\n'.join(lines)

def fix_comment_indentation_no_deletion(s):
    """
    Fixes indentation of #cython() code lines to match surrounding indentation.
    """
    if get_verbose():
        util.print_header("Fixing comment indentation")
        print("Input string:")
        print(s)

    is_tab = '\t' in s
    indent_char = '\t' if is_tab else ' '
    lines = s.split('\n')
    for (i, line) in enumerate(lines):
        if line.strip().startswith('#') and not line.strip().startswith(transform_str):
            try:
                lines[i] = indent_char * indentation + line.lstrip()
            except:
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
        else:
            if len(line):
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
    
    return '\n'.join(lines)

def fix_comment_indentation_no_deletion(s):
    """
    Fixes indentation of #cython() code lines to match surrounding indentation.
    """
    if get_verbose():
        util.print_header("Fixing comment indentation")
        print("Input string:")
        print(s)

    is_tab = '\t' in s
    indent_char = '\t' if is_tab else ' '
    lines = s.split('\n')
    for (i, line) in enumerate(lines):
        if line.strip().startswith('#') and not line.strip().startswith(transform_str):
            try:
                lines[i] = indent_char * indentation + line.lstrip()
            except:
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
        else:
            if len(line):
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
    
    return '\n'.join(lines)

def introduce_new_variables_py_ast(defnode, initialized_var_strL):
    """
    Given a redbaron.DefNode instance and a list of new lines to introduce, introduce them after the string after_cdef_trailer.
    """
    after_cdef_node = get_after_cdef_node_py_ast(defnode)
    if not hasattr(after_cdef_node, 'parent'):
        py_ast.add_parent_info(defnode)
    for i in range(len(initialized_var_strL)):
        py_ast.add_before_node(defnode, after_cdef_node.parent, py_ast.get_ast(initialized_var_strL[i]).body[0])
        
def get_after_cdef_node_py_ast(defnode):
    """
    Get the node corresponding to after_cdef_trailer.
    """
    after_cdef_nodeL = [tnode for tnode in py_ast.get_comment_strings(defnode) if py_ast.dump_ast(tnode) == after_cdef_trailer]
    if len(after_cdef_nodeL) == 0:
        raise TransformError('no after_cdef_trailer found ("{}"): must apply TypeSpecialize first'.format(after_cdef_trailer))
    after_cdef_node = after_cdef_nodeL[0]
    return after_cdef_node

def get_unique_id(prefix):
    return prefix + '_' + hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()

def find_const_value(defnode, arg_str, seen_names):
    """
    given arg_str, which usually represents a dimension size of an array
    eg: a // 4 + 8
    try replace variables with constants
    """
    try:
        value = eval(arg_str)
        return value
    except:
        dimension_node = py_ast.get_ast(arg_str).body[0].value
        namenodes = py_ast.find_all(dimension_node, ast.Name)
        names = []
        for namenode in namenodes:
            if namenode.id not in names:
                names.append(namenode.id)
        
        assignnodes = py_ast.find_all(defnode, ast.Assign)
        aug_assignnodes = py_ast.find_all(defnode, ast.AugAssign)
        
        for name in names:
            if name in seen_names:
                raise TransformError('could not replace variable to const')
            
            potential_assignnodes = [assignnode for assignnode in assignnodes if len(assignnode.targets) == 1 and isinstance(assignnode.targets[0], ast.Name) and assignnode.targets[0].id == name]
            potential_augassigns = [assignnode for assignnode in aug_assignnodes if isinstance(assignnode.target, ast.Name) and assignnode.target.id == name]
            
            if len(potential_assignnodes) == 1 and len(potential_augassigns) == 0:
                seen_names.append(name)
                for namenode in namenodes:
                    if namenode.id == name:
                        py_ast.replace_node(dimension_node, namenode, potential_assignnodes[0].value)
                return find_const_value(defnode, py_ast.dump_ast(dimension_node), seen_names)
            else:
                raise TransformError('could not replace variable to const')
