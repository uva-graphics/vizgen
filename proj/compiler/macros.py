
"""
Typed macros which perform textual substitutions on Python source as it is read.

A macro is a three-tuple (source_pattern, arg_types, dest_pattern).

For example:

   ('numpy.clip(x, a, b)', (Float, Float, Float), 'numpy_clip_double(x, a, b)')

The source pattern matches text directly (with spaces stripped), except for items in parentheses, which match
sub-expressions as macro variables (there should be exactly one set of parentheses). The types are passed as
util.CythonType instances to arg_types, which can either be a single function (which accepts all types as a
tuple), or a tuple of functions. Each type function returns (success, transformed_arg_str). If all type functions
have a True value for success then the macro matches and the text is replaced with the destination pattern. If
the destination pattern is a function then it is called with all types as an argument tuple, and the return value
from this function is used as the destination pattern.
"""

import string
import util
import ast
import py_ast
import astor
import re

import numpy
import time

workaround_cython_546 = True      # Enable workaround for Cython bug #546: http://trac.cython.org/ticket/546
use_py_ast_fix_range_3_args = True      # Use ast module for fix_range_3_args()

verbose = False
cpointer_str = '__cpointer['

# Aliases for numpy library, starting with its original name 'numpy'
numpy_modules = ['numpy', 'np']

numpy_array_storage_funcs = """
zeros
zeros_like
ones
ones_like
empty
empty_like
asarray
asarray_chkfinite
ascontiguousarray
asfarray
asfortranarray
asmatrix
array
""".split()

def fix_cpointer_in_final_cython(s):
    """
    Given final Cython program containing cpointer_str(x), replace it with &x[0].
    """
    while True:
        try:
            i = s.index(cpointer_str)
        except ValueError:
            return s
        start = i+len(cpointer_str)
        subexpr_len = 0
        paren_count = 1
        while True:
            subexpr_len += 1
            if start+subexpr_len-1 >= len(s):
                return s
            c = s[start+subexpr_len-1]
            if c == '[':
                paren_count += 1
            elif c == ']':
                paren_count -= 1
            if paren_count == 0:
                break

        s = s[:i] + '&(' + s[start:start+subexpr_len-1] + ')[0]' + s[start+subexpr_len:]

    return s

def make_cpointer(s):
    return cpointer_str + s + ']'

def Float(program_info, t, arg_str):
    return (t.cython_type in ['float', 'double'], arg_str)

def Int(program_info, t, arg_str):
    return (t.cython_type in ['int'], arg_str)

def FloatOrInt(program_info, t, arg_str):
    return (Float(program_info, t, arg_str) or Int(t, arg_str), arg_str)

def VecXf_ptr_same(program_info, t, arg_str, n=None):
    if n is None:
        if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'float' and isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    else:
        if t.shape == (n,) and t.primitive_type() == 'float' and isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    return (ans, make_cpointer(arg_str))

def VecXg_ptr_same(program_info, t, arg_str, n=None):
    if n is None:
        if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'double' and isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    else:
        if t.shape == (n,) and t.primitive_type() == 'double' and isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    return (ans, make_cpointer(arg_str))

def VecXf_same(program_info, t, arg_str, n=None):
    if n is None:
        if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'float' and not isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    else:
        if t.shape == (n,) and t.primitive_type() == 'float' and not isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    return (ans, arg_str)

def VecXg_same(program_info, t, arg_str, n=None):
    if n is None:
        if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'double' and not isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    else:
        if t.shape == (n,) and t.primitive_type() == 'double' and not isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
            ans = t.shape[0]
        else:
            ans = False
    return (ans, arg_str)

def VecXf_ptr(program_info, t, arg_str):
    if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'float' and isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
        ans = t.shape[0]
    else:
        ans = False
    return (ans, make_cpointer(arg_str))

def VecXg_ptr(program_info, t, arg_str):
    if  isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'double' and isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
        ans = t.shape[0]
    else:
        ans = False
    return (ans, make_cpointer(arg_str))

def VecXf(program_info, t, arg_str):
    if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'float' and not isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
        ans = t.shape[0]
    else:
        ans = False
    return (ans, arg_str)

def VecXg(program_info, t, arg_str):
    if isinstance(t.shape, tuple) and len(t.shape) == 1 and isinstance(t.shape[0], int) and t.primitive_type() == 'double' and not isinstance(py_ast.get_ast(arg_str).body[0].value, ast.Name):
        ans = t.shape[0]
    else:
        ans = False
    return (ans, arg_str)

def ConstantLengthVec(program_info, t, arg_str):
    return (len(t.shape) == 1 and isinstance(t.shape[0], int), arg_str)

def ArrayStorageRewriteFloat32(numpy_func):
    """
    Matches float64 arrays only if ArrayStorage macro has been applied with the intent that float64 arrays be rewritten to type float32.
    """
    def type_check(program_info, t, arg_str):
#        print('ArrayStorageRewriteFloat32:', 'numpy_func:', numpy_func, 'transformL:', [str(x) for x in program_info.transformL], 't:', t, 'arg_str:', arg_str, 'use_float32:', any(getattr(t, 'use_float32', False) for t in program_info.transformL))
        return (program_info.is_rewrite_float32(), arg_str)
    return type_check

def IsCythonType(cython_type):
    def type_check(program_info, t, arg_str):
        return (t == cython_type, arg_str)
    return type_check

# Mappings from vector macro name to scalar macro name
macro_to_scalar = {}

# List of all macros
macros = [
    ('numpy.clip(x, a, b)',     (Float, FloatOrInt, FloatOrInt),      'numpy_clip_double(x, a, b)'),
    ('numpy.clip(x, a, b)',     (Int, Int, Int),            'numpy_clip_int(x, a, b)'),
    ('numpy.clip(x, a, b)',     (VecXf_ptr, Float, Float),  'numpy_clip_vecXf_ptr(x, a, b)'),
    ('numpy.clip(x, a, b)',     (VecXg_ptr, Float, Float),  'numpy_clip_vecXg_ptr(x, a, b)'),
    ('numpy.clip(x, a, b)',     (VecXf, Float, Float),      'numpy_clip_vecXf(x, a, b)'),
    ('numpy.clip(x, a, b)',     (VecXg, Float, Float),      'numpy_clip_vecXg(x, a, b)'),
    ('(x)**2',                  (Float,),                   'square_double(x)'),
    ('numpy.square(x)',         (Float,),                   'square_double(x)'),
    ('(x)**2',                  (Int,),                     'square_int(x)'),
    ('numpy.square(x)',         (Int,),                     'square_int(x)'),
    ('(x)**0.5',                (FloatOrInt,),              'libc.math.sqrt(x)'),
    ('int(x)',                  (Float,),                   'float_to_int(x)'),
    ('int(x)',                  (Int,),                     'int_to_int(x)'),
    ('float(x)',                (Float,),                   'float_to_float(x)'),
    ('float(x)',                (Int,),                     'int_to_float(x)'),
    ('abs(x)',                  (Int,),                     'libc.stdlib.abs(x)'),
    ('numpy.abs(x)',            (Int,),                     'libc.stdlib.abs(x)'),
    ('abs(x)',                  (Float,),                   'libc.math.fabs(x)'),
    ('numpy.abs(x)',            (Float,),                   'libc.math.fabs(x)'),
    ('numpy.linalg.norm(x)',    (VecXf_ptr,),               'numpy_linalg_norm_vecXf_ptr(x)'),
    ('numpy.linalg.norm(x)',    (VecXg_ptr,),               'numpy_linalg_norm_vecXg_ptr(x)'),
    ('numpy.dot(x, y)',         (VecXf_ptr_same, VecXf_ptr_same),     'numpy_dot_vecXf_ptr(x, y)'),
    ('numpy.dot(x, y)',         (VecXg_ptr_same, VecXg_ptr_same),     'numpy_dot_vecXg_ptr(x, y)'),
    ('random.randrange(x)',     (Int,),                     'randrange_1arg(x)'),
    ('random.randrange(x, y)',  (Int, Int),                 'randrange_2arg(x, y)'),
    ('len(x)',                  (ConstantLengthVec,),       lambda t: str(t.shape[0])),
    ('pow(x, y)',               (FloatOrInt, FloatOrInt),   'libc.math.pow(x, y)'),
    ('util.randrange(x, a, b)', (Int, Int, Int),            'randrange_seed(x, a, b)'),
]

for _numpy_func in numpy_array_storage_funcs:
    macros.append(('numpy.' + _numpy_func + '(x)', (ArrayStorageRewriteFloat32(_numpy_func),), 'numpy.' + _numpy_func + '(x, "float32")'))

# Missing math functions:
#  - Due to requiring tuple return values: frexp, modf
#  - Constants: e, pi
#  - Conversion functions: radians, degrees
#  - log() with 2 arguments

math_funcs = {'acos': (1, None, 'arccos'), 'acosh': (1, None, 'arccosh'), 'asin': (1, None, 'arcsin'), 'asinh': (1, None, 'arcsinh'), 'atan': (1, None, 'arctan'), 'atan2': (2, None, 'arctan2'), 'atanh': (1, None, 'arctanh'), 'ceil': 1, 'copysign': 1, 'cos': 1, 'cosh': 1, 'erf': 1, 'erfc': 1, 'exp': 1, 'expm1': 1, 'fabs': 1, 'floor': 1, 'fmod': 2, 'tgamma': (1, 'gamma', ''), 'hypot': 2, 'isinf': 1, 'isnan': 1, 'ldexp': 2, 'lgamma': 1, 'log': 1, 'log10': 1, 'log1p': 1, 'modf': 2, 'pow': 2, 'sin': 1, 'sinh': 1, 'sqrt': 1, 'tan': 1, 'tanh': 1, 'trunc': 1}

def unpack_math_funcs():
    for (_math_func, _math_nargs) in math_funcs.items():
        _math_source_func = _numpy_source_func = _math_func
        if isinstance(_math_nargs, tuple):
            (_math_nargs, _math_source_func, _numpy_source_func) = _math_nargs
            if _math_source_func is None:
                _math_source_func = _math_func
            if _numpy_source_func is None:
                _numpy_source_func = _math_func
        yield (_math_func, _math_nargs, _math_source_func, _numpy_source_func)

for (_math_func, _math_nargs, _math_source_func, _numpy_source_func) in unpack_math_funcs():
    _math_args = '(x)' if _math_nargs == 1 else '(x, y)'
    if len(_math_source_func):
        macros.append(('math.{}{}'.format(_math_source_func, _math_args), (FloatOrInt,)*_math_nargs, 'libc.math.{}{}'.format(_math_func, _math_args)))
    if hasattr(numpy, _numpy_source_func):
        macros.append(('numpy.{}{}'.format(_numpy_source_func, _math_args), (FloatOrInt,)*_math_nargs, 'libc.math.{}{}'.format(_math_func, _math_args)))

if workaround_cython_546 and use_py_ast_fix_range_3_args:
    def fix_range_3_args(s):
        """
        Work around Cython bug #546.
        
        With Cython, 'for i in range(a, b, c)' for ints i, a, b, c generates inefficient Python code. Our workaround is to when the
        Cython code is finalized (the postprocess), replace range function calls with old Pyrex style looping:
        "for i from a <= i < b by c". Currently we assume step is positive (TODO: fix this in Cython, and/or fix this the same way
        that cython.parallel.prange() does, by finding the number of total iterations and then calculating the index variable with
        respect to the iteration count).
        """
        import transforms
        start_time = time.time()
        
#        print('fix_range_3_args')
        lines = s.split('\n')
        r = py_ast.get_ast(s)
        fornode_to_replace_line = {}
        for fornode in py_ast.find_all(r, ast.For):
            node = fornode.iter
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['range', 'xrange']:
            #if len(node.value) == 2 and isinstance(node.value[0], redbaron.NameNode) and node.value[0].name.value in ['range', 'xrange']:
                if isinstance(fornode.target, ast.Name):
                    range_args = node.args #node.value[1]
                    if len(range_args) == 3:
                        line = py_ast.get_line(r, fornode) - 1
                        
                        fornode_s = astor.to_source(fornode).strip().split('\n')[0]     # TODO: Parsing hack: this breaks on "for a in range(1, 10, 1): pass"
                        i = fornode.target.id
                        a = astor.to_source(range_args[0])
                        b = astor.to_source(range_args[1])
                        c = astor.to_source(range_args[2])
                        
                        start = transforms.get_unique_id('start')
                        stop = transforms.get_unique_id('stop')
                        step = transforms.get_unique_id('step')
                        current_count = transforms.get_unique_id('current_count')
                        total_count = transforms.get_unique_id('total_count')

                        # Add cdef variable statements to top of function
                        cdefL = []
                        for count in [start, stop, step, current_count, total_count]:
                            cdefL.append('{}(cdef int {})'.format(transforms.cython_str, count))
                        defnode = transforms.get_previous_node_func(r, fornode, ast.FunctionDef)
                        #py_ast.add_before_node(r, defnode.body[0], py_ast.get_ast('\n'.join(cdefL)))
                        cdef_nodes = py_ast.get_ast('\n'.join(cdefL)).body
                        for cdef_node in cdef_nodes[::-1]:
                            py_ast.add_before_node(r, defnode.body[0], cdef_node)
                        (subindent_line, subindent) = transforms.get_next_indentation(lines, line)
                        (for_nindent, forindent) = transforms.get_indentation(lines[line])
                        assert len(forindent) <= len(subindent)
                        subindent = ' ' * (len(subindent) - len(forindent))
#                        print('subindent_line={}, subindent={!r}'.format(subindent_line, subindent))
#                        with open('test_bugfix.pyx', 'wt') as f:
#                            f.write(s)
                        replace_s = """
{start} = {a}
{stop} = {b}
{step} = {c}
if {step} > 0:
    if {stop} > {start}:
        {total_count} = ({stop}-{start}+{step}-1)//{step}
    else:
        {total_count} = ({stop}-{start})//{step}
elif {step} < 0:
    if {stop} > {start}:
        {total_count} = ({start}-{stop})//(-{step})
    else:
        {total_count} = ({start}-{stop}-{step}-1)//(-{step})
else:
    {total_count} = 0
for {current_count} in range({total_count}):
{subindent}{i} = {start} + {step}*{current_count}
""".strip().format(**locals())

                        # In the else case above one could raise ValueError('step argument must not be zero') to more closely emulate Python,
                        # but then this will not work in nogil sections.
                        
                        fornode_to_replace_line[fornode_s] = replace_s
#        print('fix_range_3_args:', fornode_to_replace_line)

        s = py_ast.dump_ast(r)

        L = s.split('\n')
        i = 0
        while i < len(L):
            line = L[i].strip()
            if line in fornode_to_replace_line:
                replace_str = fornode_to_replace_line[line]
                replaceL = replace_str.split('\n')
                subL = [(transforms.cython_str if j < len(replaceL) - 1 else transforms.cython_replace_str) + '(' + replace_line + ')' for (j, replace_line) in enumerate(replaceL)]
                L[i:i] = subL
#                print('fix_range_3_args: replacing line {}: {}'.format(i, line))
                i += len(subL)
            i += 1
        
        ans = '\n'.join(L)
#        util.print_header('fix_range_3_args result:', ans)

        if transforms.do_profile:
            transforms.profile['apply_transforms_and_finalize: finalize_cython: fix_range_3_args'] += time.time() - start_time
        return ans

def find_func_calls_ast(r):
    """
    Given ast.AST instance r, return list of all nodes which are function calls (excludes keyword-argument function calls).
    """
    return [node for node in py_ast.find_all(r, ast.Call) if
            len(node.keywords) == 0 and node.starargs is None and node.kwargs is None]

def all_source_patterns(source_pattern):
    """
    Given macro source pattern, return a list of the original and all alternate variations of that pattern.
    """
    if source_pattern.startswith(numpy_modules[0] + '.') and len(numpy_modules) > 1:
        return [source_pattern] + [mod + '.' + source_pattern[len(numpy_modules[0]) + 1:] for mod in numpy_modules[1:]]
    return [source_pattern]

def remove_whitespace(s):
    while ' ' in s:
        s = s.replace(' ', '')
    while '\t' in s:
        s = s.replace('\t', '')
    return s

class MacroMatch:
    """
    Stores:
         - node:      redbaron.RedBaron or ast.AST node instance for a node matching a macro (type not checked yet though).
         - node_str:  code string associated with node after whitespace has been stripped.
         - macroL:    candidate list of macro tuples (source_pattern, arg_types, dest_pattern)
         - arg_nodes: list of arguments to the macro, sub-nodes of node
         - is_func:   bool, if True, is an ordinary function call, if False, is a binary operator (matched as a macro taking a single argument)
         - line:      line number of node
    """
    def __init__(self, root, node, node_str, macroL, arg_nodes, is_func, line):
        self.node = node
        self.node_str = node_str
        self.macroL = macroL
        self.arg_nodes = arg_nodes
        self.is_func = is_func
        if line is None:
            if isinstance(node, ast.AST):
                line = py_ast.get_line(root, node)
            else:
                line = redbaron_util.line(node)
        self.line = line

    def __repr__(self):
        return 'MacroMatch({!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(self.node, self.node_str, self.macroL, self.arg_nodes, self.is_func, self.line)

def find_macros_ast(r, macros, all_func_calls=False, match_return=True):
    """
    Given ast.AST instance r and list of macros, return list of nodes which are candidate macro calls (as MacroMatch instances).
    
    The list has not yet been filtered to see if the types match.
    
    If all_func_calls is True then also return all non-macro function calls.
    """
    def func_prefix(s):
        return remove_whitespace(s[:s.index('(')])
    
    node_to_str_cache = {}

    def get_call_args(node):
        if isinstance(node, ast.Return):
            return [node.value]
        else:
            assert isinstance(node, ast.Call)
            return node.args
    
    def node_to_str(node):
        if id(node) in node_to_str_cache:
            return node_to_str_cache[id(node)]
        if isinstance(node, ast.Return):
            s = 'return'
        else:
#            util.print_header('node_to_str in find_macros_ast:', astor.dump(node))
            s = remove_whitespace(py_ast.to_source_any(node).strip()) #astor.to_source(node))
            if s.startswith('(') and s.endswith(')'):
                s = s[1:-1]
#        print('node_to_str:', s)
        node_to_str_cache[id(node)] = s
        return s

    ans_id_to_node = {}
    remove_parents = set()

    # Match all function calls
    
    func_calls = find_func_calls_ast(r)
    
    if match_return:
        for node in py_ast.find_all(r, ast.Return):
            func_calls.append(node)
#            print(' => find_macros_ast: return {}'.format(node))

    func_call_prefix_to_nodes = {}
    for func_call in func_calls:
        if isinstance(func_call, ast.Return):
            func_call_prefix = 'return'
        else:
            func_call_prefix = astor.to_source(func_call.func)
        func_call_prefix_to_nodes.setdefault(func_call_prefix, [])
        func_call_prefix_to_nodes[func_call_prefix].append(func_call)

        if all_func_calls or (match_return and isinstance(func_call, ast.Return)):
            match_obj = MacroMatch(r, func_call, node_to_str(func_call), [], get_call_args(func_call), True, None)
            ans_id_to_node[id(func_call)] = match_obj
            if verbose:
                if isinstance(func_call, ast.Return):
                    print(' => find_macros_ast: return match object is {}'.format(match_obj))

    def filter_parents(node):
        node_str = node_to_str(node)
        if verbose:
            print('filter_parents:', repr(node_str))

        for parent in py_ast.parent_list(r, node):
            parent_str = node_to_str(parent)
            if verbose:
                print('filter_parents:', repr(node_str), repr(parent_str))
            if parent_str == node_str:
                remove_parents.add((id(parent), parent_str))
            else:
                break

    # Match function call macros
    for (orig_source_pattern, arg_types, dest_pattern) in macros:
        for source_pattern in all_source_patterns(orig_source_pattern):
            source_pattern_prefix = func_prefix(source_pattern)
            for node in func_call_prefix_to_nodes.get(source_pattern_prefix, []):
                if id(node) not in ans_id_to_node:
                    if verbose:
                        print('node, func-call:', repr(astor.to_source(node)), id(node), type(node), (orig_source_pattern, arg_types, dest_pattern))
                    ans_id_to_node[id(node)] = MacroMatch(r, node, node_to_str(node), [(source_pattern, arg_types, dest_pattern)], get_call_args(node), True, None)
                    filter_parents(node)
                else:
                    ans_id_to_node[id(node)].macroL.append((source_pattern, arg_types, dest_pattern))
#                    ans_strs.add(node_str)

    # Match non-function call patterns that are binary operators, e.g. '(x)**2' or '3*(x)', although as macros these take a single argument.
    # We intentionally only capture the variable part (x) as an argument, assuming the macro handles replacing the rest of the expression.
    all_nodes = list(ast.walk(r))
    all_nodes_with_strs = [(node, node_to_str(node)) for node in all_nodes]

    letters_underscore = string.ascii_letters + '_'
    digits_letters_underscore = string.digits + string.ascii_letters + '_.'
    for (source_pattern, arg_types, dest_pattern) in macros:
        prefix = func_prefix(source_pattern)
        suffix = remove_whitespace(source_pattern[source_pattern.index(')')+1:])
        prefix_identifier_len = 0
        if len(prefix) and prefix[0] in (string.ascii_letters + '_'):
            prefix_identifier_len = 1
            while prefix_identifier_len+1 <= len(prefix) and prefix[prefix_identifier_len] in digits_letters_underscore:
                prefix_identifier_len += 1
        prefix_after_identifier = prefix[prefix_identifier_len:]
        if len(suffix) or len(prefix_after_identifier):
            r_macro = ast.parse(source_pattern).body[0].value
            if isinstance(r_macro, ast.BinOp):
                op = r_macro.op
            else:
                raise ValueError('could not parse non-function macro with non-binary operator form: {}'.format(source_pattern))

            # Handle a non-function call macro
            if verbose:
                print('non function call macro:', repr(source_pattern), repr(prefix), repr(suffix))

            for (node, node_str) in all_nodes_with_strs:
#                if verbose:
#                    print('  candidate match node_str={}'.format(node_str))
#                prefix_ok = (node_str.startswith(prefix) and len(prefix)) or (not len(prefix) and node_str.startswith(letters_underscore+'('))
#                suffix_ok = (node_str.endswith(suffix) and len(suffix)) or (not len(suffix) and node_str.endswith(
                if node_str.startswith(prefix) and node_str.endswith(suffix):
                    if isinstance(node, ast.BinOp) and type(node.op) == type(op):
                        if verbose:
                            print('node, non-func call:', repr(node_str), 'macro:', (source_pattern, arg_types, dest_pattern))
                        if id(node) not in ans_id_to_node:
                            #if node_str not in ans_strs:
                            #    for parent in redbaron_util.parent_list(node):
                            #        remove_nonfunc_ids.append(id(parent))
        #                        ans_strs.add(node_str)
                            arg_idx = 1 if len(prefix) else 0
                            call_args = [node.left if arg_idx == 0 else node.right]
                            ans_id_to_node[id(node)] = MacroMatch(r, node, node_to_str(node), [(source_pattern, arg_types, dest_pattern)], call_args, False, None)
                            
                            filter_parents(node)
                            #print('match:', node_str)
                        else:
                            ans_id_to_node[id(node)].macroL.append((source_pattern, arg_types, dest_pattern))
#            print('done nodes')
    if verbose:
        print('remove_parents:', remove_parents)

    ans_filtered = [node for (node_id, node) in ans_id_to_node.items() if (node_id, node.node_str) not in remove_parents]
    ans_sorted = sorted(ans_filtered, key=lambda node: (py_ast.get_line(r, node.node), len(node.node_str), node.node_str))
#    print('macros result:', ans_sorted)
    return ans_sorted
