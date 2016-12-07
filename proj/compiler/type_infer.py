
"""
Type inference. Requires Python 3+.
"""

from ast import *
import copy
import astor
import util
import types
import pprint
import type_funcs
import type_utils
import macros
import warnings
import os
import functools
#import functools
import py_ast
import numpy
import numpy as np
import math
from py_ast import BottomUpVisitor, TopDownVisitor, add_parent_info, parent_function

checks = False                      # Additional checks for debugging (should be turned off for checked-in code)

return_value = '_return_value'      # Key to get type associated with return value
return_value_tuple = util.types_non_variable_prefix + 'return'

accept_annotated_type = True

def parse_statements(s):
    """
    Parse source code containing one or more statements, returning list of ast nodes.
    """
    return parse(s).body

class TypeContext:
    def __init__(self, program_info, arg_tuple, type_signature, is_call):
        self.program_info = program_info
        self.arg_tuple = arg_tuple
        self.type_signature = type_signature
        self.is_call = is_call

def dump_ast_types(m0):
    """
    Same as astor.dump(m0), but include type information for each node (from the node.type field).
    """
    return astor.dump(m0, iter_node=lambda node: [(node.type, 'type'), (node.aliases, 'aliases')] + list(astor.iter_node(node)) if hasattr(node, '_fields') else
                                                     astor.iter_node(node))

def list_union(a, b):
    return sorted(set(a) | set(b))

class ApplyTypeFuncNotFound(Exception):
    pass

def type_infer(program_info, verbose=False, both_formats=False, get_macros=False, prealloc=True, use_4channel=False):
    """
    Infers types given a module's source code.
    
    Given input module source code, run the program, which may result in
    calling one or more functions. All callables beginning with 'test'
    are also run. Infer input and return types for all functions, as well
    as local variable types. Types are CythonType instances. Returns a dict
    mapping function name to a list of type signatures* for the function,
    each of which is a dict mapping variable name to variable type.
    
    Returns a dict with keys:
     * 'types' => type signature in above-described dict format.
     * 'types_internal' => type signature in internal representation.
     * 'prealloc_internal' => dict mapping from function name to dict mapping from variable name to whether
                              alias analysis concludes that the variable can be preallocated+.
     * 'prealloc' => dict mapping from function name to list of variables that alias analysis concluded can be preallocated+.

    * We assume that global functions and modules are not dynamically rebound
      (e.g. f = g for functions f and g), and types in the global and local
      scope are also not dynamically modified by constructs such as exec/eval.
    + But one still needs to verify whether the RHS is allocated using a suitable numpy command.
    """
    
    annotated_type = {}
    
    def parse_type(type_str):
        """
        parse a str annotated in source code into CythonType
        """
        if not type_str.startswith('vizgen'):
            return util.CythonType.from_cython_type(type_str, program_info)
        
        node = py_ast.get_ast(type_str).body[0].value
        
        assert(isinstance(node.args[0], Str))
        dtype_str = node.args[0].s
        
        if(isinstance(node.args[1], Num)):
            dim = node.args[1].n
            shape_str = '(' + 'None, ' * (dim - 1) + 'None)'
            shape = eval(shape_str)
        else:
            shape = eval(py_ast.dump_ast(node.args[1]))
            dim = len(shape)
        
        if not isinstance(shape, tuple):
            shape = tuple(shape,)
            
        cython_type_str = 'numpy.ndarray[numpy.{}_t, ndim={}](shape={},shape_list=[])'.format(dtype_str, str(dim), str(shape))
        
        return util.CythonType.from_cython_type(cython_type_str, program_info)
    
    annotated_type = {}
    
    if accept_annotated_type:
        
        rootnode = py_ast.get_ast(program_info.s_orig)
        defnodes = py_ast.find_all(rootnode, FunctionDef)
        for defnode in defnodes:
            type_to_add = {}
            if defnode.returns is not None:
                type_to_add['_return_value'] = parse_type(py_ast.dump_ast(defnode.returns))
            for arg in defnode.args.args:
                if arg.annotation is not None:
                    type_to_add[arg.arg] = parse_type(py_ast.dump_ast(arg.annotation))
            
            defnode_dump = py_ast.dump_ast(defnode)
            
            lines = defnode_dump.split('\n')
            
            for line in lines:
                while '# ' in line:
                    line = line.replace('# ', '#')
#                print('LINE:', line)
                if '#type:' in line:
                    annotated_nodes = py_ast.get_ast(line.strip()).body
                    ind = line.find('#type:')
                    type_str = line[ind : len(line.rstrip())].lstrip('#type:').strip()
                    
                    if isinstance(annotated_nodes[0], Assign) and len(annotated_nodes[0].targets) == 1 and isinstance(annotated_nodes[0].targets[0], Name):
                        type_to_add[annotated_nodes[0].targets[0].id] = parse_type(type_str)
                    elif isinstance(annotated_nodes[0], For) and isinstance(annotated_nodes[0].target, Name):
                        type_to_add[annotated_nodes[0].target.id] = parse_type(type_str)
            
            annotated_type[defnode.name] = [type_to_add]

        if verbose:
            print('annotated_type:', annotated_type)
        
    ObjectType = util.CythonType.from_value(object(), program_info)         # An unknown (uninferred) type
    
    # Log function argument types
    
    type_infer_prefix = '_type_infer_arg_types_'
    
    def func_arg_names(node):
        args = node.args.args
        argnames = [arg.arg for arg in args]
        return argnames
    
    module_source = program_info.s_orig
    module_source = astor.to_source(parse(module_source))                   # Round trip once to merge line continuations into a single line
    m0 = parse(module_source)
    m = copy.deepcopy(m0)
    logged_func_names = []
    class LogFunctionTypes(NodeTransformer):
        def visit_FunctionDef(self, node, type_infer_prefix=type_infer_prefix):
            node = copy.deepcopy(node)
#            args = node.args.args
            func_name = node.name
            logged_func_names.append(func_name)
            argnames = func_arg_names(node) #[arg.arg for arg in args]
            argtype_vals = '[' + ','.join("('" + argname + "', util.CythonType.from_value(" + argname + ", _type_infer_program_info))" for argname in argnames) + ']'
            node.body = parse_statements("""
global {type_infer_prefix}{func_name}
try:
    {type_infer_prefix}{func_name}
except:
    {type_infer_prefix}{func_name} = util.TypeSignatureSet({argnames})
{type_infer_prefix}{func_name}.add({argtype_vals})
#print({type_infer_prefix}{func_name})
""".format(**locals())) + node.body

#            '''
#_argtype_values = {{}}
#def _log_argtype_value(id_num, v):
#    try:
#        if type(v) == type(_argtype_values[id_num]):
#            return v
#    except KeyError:
#        _argtype_values[id_num] = v
#        return v
#    _argtype_values[id_num] = util.promote_numeric(_argtype_values[id_num], v)
#    return v
#
#global _log_func_types
#try:
#    _log_func_types
#except:
#    _log_func_types = {{}}
#_log_func_types.setdefault('{funcname}', {})
#_log_func_types['{funcname}'].setdefault('{argname}',
#')
#            '''
            return node
    LogFunctionTypes().visit(m)
    module_source = astor.to_source(m)
    module_source += """
_globals_names = list(globals().keys())
for _callable in _globals_names:
    if callable(globals()[_callable]) and _callable.startswith('test'):
        globals()[_callable]()
"""
#    print('-'*70)
#    print(module_source)
#    print('-'*70)
    module = compile(module_source, '<unknown>', 'exec')
    
    m_globals = {'_type_infer_program_info': program_info, 'util': util}
#    print('before exec')
    always_verbose_run = False
    orig_initial_run = util.is_initial_run
    orig_path = os.getcwd()
    orig_use_4channel = util.use_4channel
    try:
        util.is_initial_run = True
        util.use_4channel = use_4channel
        os.chdir(program_info.path)
        with util.SuppressOutput(verbose=verbose or always_verbose_run):
            exec(module, m_globals, m_globals)
    finally:
        util.use_4channel = orig_use_4channel
        util.is_initial_run = orig_initial_run
        os.chdir(orig_path)
#    print('done')
#    print(m_globals)

    can_prealloc = {}

    def constrain_types_by_annotation():
        for func_name in type_signature:
            if func_name in annotated_type:
#                print('annotated_type, func_name={}:'.format(func_name), annotated_type[func_name])
                for annotated_type_d in annotated_type[func_name]:
                    for annotated_type_varname in annotated_type_d:
#                        print('setting {}.{} type: {}'.format(func_name, annotated_type_varname, annotated_type_d[annotated_type_varname]))
                        for arg_tuple in type_signature[func_name]:
                            type_signature[func_name][arg_tuple][annotated_type_varname] = annotated_type_d[annotated_type_varname]
#            print('local_types_d:', local_types_d)

    def reset_function_types(func_name):
        key = type_infer_prefix + func_name
        if key in m_globals:
            # A full type signature maps an argument type tuple to a dictionary mapping local variable
            # names to CythonType instance or ObjectType.
            # The special key return_value in these local variables stores the function's return value type.
            # An argument type tuple is a tuple of argument types sorted by name: (name, typeval), name is a str and typeval is a CythonType instance
            full_sig = {}
            for arg_sig in m_globals[key]:
                arg_tuple = tuple(sorted(arg_sig.items()))
                local_types_d = dict(arg_tuple)
                full_sig[arg_tuple] = local_types_d
            type_signature[func_name] = full_sig
        else:
            type_signature[func_name] = {}
#        print('reset_function_types, func_name={}, type_signature:'.format(func_name), type_signature)
        constrain_types_by_annotation()
        
        # can_prealloc maps a function name to a dict mapping variable name to True or False, for whether the variable can be preallocated.
        # Variables are assumed to have a default value of True until they are set.
        can_prealloc[func_name] = {}

    type_signature = {}                                     # Full type signature for each function
    for func_name in logged_func_names:
        reset_function_types(func_name)
    if checks:
        type_signature0 = copy.deepcopy(type_signature)
    
    if verbose:
        print('Initial type signatures:')
        print(type_signature)

    # Infer function local types and return types given argument types

    if verbose:
        print(astor.dump(m0))
    """
    class InferTypes(NodeVisitor):
        def visit_Assign(self, node):
            print('Assign:', node.lineno, dump(node))
            NodeVisitor.generic_visit(self, node)
        def visit_Return(self, node):
            print('Return:', node.lineno, dump(node))
            NodeVisitor.generic_visit(self, node)
        def visit_UnaryOp(self, node):
            print('UnaryOp:', node.lineno, dump(node))
            NodeVisitor.generic_visit(self, node)
        def visit_BinOp(self, node):
            print('BinOp:', node.lineno, dump(node))
            NodeVisitor.generic_visit(self, node)
    """

    add_parent_info(m0)

    function_name_to_node = {}
    class MapFunctionNameToNode(TopDownVisitor):
        def visit_FunctionDef(self, node):
            function_name_to_node[node.name] = node

    MapFunctionNameToNode().visit(m0)

    class AddParentFunctionDef(TopDownVisitor):
        def generic_visit(self, node):
            if isinstance(node, FunctionDef):
                node.parent_functiondef = node
                node.parent_functionname = node.name
            elif getattr(node.parent, 'parent_functiondef', None) is not None:
                node.parent_functiondef = node.parent.parent_functiondef
                node.parent_functionname = node.parent_functiondef.name
            elif not hasattr(node, 'parent_functiondef'):
                node.parent_functiondef = None
                node.parent_functionname = None

    AddParentFunctionDef().visit(m0)

    def arg_tuples(node):
        arg_tuples_ans = sorted(type_signature.get(node.parent_functionname, {}).keys())
        if checks:
            orig_ans = sorted(type_signature0.get(node.parent_functionname, {}).keys())
            assert arg_tuples_ans == orig_ans, (arg_tuples_ans, orig_ans)
        return arg_tuples_ans

    # The node's type attribute maps an argument type tuple (see above for definition) to a CythonType instance or ObjectType.
    class AssignUnknownType(TopDownVisitor):
        def generic_visit(self, node):
            arg_tuplesL = arg_tuples(node)
            node.type = {arg_tuple: ObjectType for arg_tuple in arg_tuplesL}
            node.aliases = {arg_tuple: [] for arg_tuple in arg_tuplesL}
    AssignUnknownType().visit(m0)

    if verbose:
        util.print_header('Annotated AST before inference:')
        print(dump_ast_types(m0))

    def type_from_value(val):
        return util.CythonType.from_value(val, program_info)
    
    def type_from_known_value(val):
        return util.CythonType.from_known_value(val, program_info)

    def add_local_type(local_types_d, varname, vartype, node, apply_constraints=True):
        if varname not in local_types_d:
            if verbose:
                print('*** Add:', varname, vartype)
            local_types_d[varname] = copy.deepcopy(vartype)
        else:
            if verbose:
                print('*** Union:', varname, vartype, local_types_d[varname])
            local_types_d[varname] = util.union_cython_types(local_types_d[varname], vartype, numeric_promotion=True)   # TODO: numeric_promotion=False is safer, but numeric_promotion=True is faster. Figure out a way to resolve this...
        if apply_constraints:
            constrain_types_by_annotation()

    def update_aliases(target_node, source_node):
        for arg_tuple in arg_tuples(target_node):
            target_node.aliases[arg_tuple] = list_union(target_node.aliases[arg_tuple], source_node.aliases[arg_tuple])

    class InferTypes(BottomUpVisitor):
        if checks:
            def visit_one_node(self, node, lineno=None):
                BottomUpVisitor.visit_one_node(self, node, lineno)
                try:
                    arg_tuples(node)
                except:
                    print('InferTypes: exception raised at node:', node)
                    raise
        
        def __init__(self):
            BottomUpVisitor.__init__(self, strict_line_order=True)
        
        def print_info(self, node):
            """Print some information about a given node."""
            if verbose:
                print(node.__class__.__name__, ', parent:', node.parent, 'parent_functionname:', node.parent_functionname, 'lineno:', getattr(node, 'lineno', None), 'type:', node.type, dump(node))
        
        def add_type(self, node, varname: str, vartype_node, vartype_func=lambda _arg: _arg, alias_node=None):
            """Add a type for a variable of given name varname to surrounding function scope, if it exists (if no function scope then do nothing)."""
            if verbose:
                print('add_type:\n  node={}\n  varname={}\n  vartype_node={}\n  alias_node={}\n'.format(dump_ast_types(node), varname, dump_ast_types(vartype_node), dump_ast_types(alias_node)))
            vartype = vartype_node.type
            vartype_aliases = vartype_node.aliases
            if node.parent_functionname is None:
                return
            assert alias_node is not None
            for arg_tuple in vartype:
                d = type_signature[node.parent_functionname][arg_tuple]
                add_local_type(d, varname, vartype_func(vartype[arg_tuple]), node, apply_constraints=False)
                alias_node.aliases[arg_tuple] = list_union(alias_node.aliases[arg_tuple], vartype_aliases[arg_tuple])
                #if varname not in d:
                #    print('*** Add:', varname, vartype)
                #    d[varname] = copy.deepcopy(vartype[arg_tuple])
                #else:
                #    print('*** Union:', varname, vartype, d[varname])
                #    d[varname] = util.union_cython_types(d[varname], vartype[arg_tuple], numeric_promotion=False)
            #d = local_types[node.parent_functionname]
            #if varname not in d:
            #    d[varname] = vartype
            #else:
            #    d[varname] = util.union_cython_types(d[varname], vartype)
            constrain_types_by_annotation()
            
        def check_node_type(node):
            assert isinstance(arg_tuples(node), list), arg_tuples(node)
        
        def apply_typefunc(self, node, node_args, func_fullname, is_call=True, method=False, method_selfnode=None, stop_if_not_found=False, no_prealloc_not_found=False):
            if verbose:
                print('*** apply_typefunc, node:', node, 'method:', method)
            if func_fullname.startswith('np.'):
                func_fullname = 'numpy.' + func_fullname[len('np.'):]
            if method:
                assert method_selfnode is not None
                if node_args == []:
                    try:
                        eval_num = eval(py_ast.dump_ast(node))
                        if isinstance(eval_num, (float, int)):
                            ans = type_from_value(eval_num)
                            res = {}
                            for arg_tuple in arg_tuples(node):
                                res[arg_tuple] = ans
                            node.type = res
                            node.aliases = {}
                            return
                    except:
                        pass
            mod_fullname = 'type_funcs.{}_'.format('typefunc' if not method else 'typemethod') + func_fullname
            if verbose:
                print('*** apply_typefunc, mod_fullname={}'.format(mod_fullname))
            try:
                mod_func = eval(mod_fullname, globals())
            except:
                mod_func = None
            if stop_if_not_found and mod_func is None:
                raise ApplyTypeFuncNotFound

            if verbose:
                print('Original arg_tuples:', arg_tuples(node))
            
            res = {}
            res_aliases = {}
            for arg_tuple in arg_tuples(node):
                ok = False
                if mod_func is not None:
                    # Apply given type function
                    if verbose:
                        print('*** apply_typefunc, located type function {}'.format(mod_fullname))
                        print('*** apply_typefunc, arg_tuple before mod_func is {}'.format(arg_tuple))
                    ctx = TypeContext(program_info, arg_tuple, type_signature, is_call)
                    all_args = node_args if not method else ([method_selfnode] + list(node_args))
                    if verbose:
                        print('*** apply_typefunc, all_args is {}'.format(astor.dump(all_args)))
                    try:
                        # TODO: workaround for asarray
                        new_args = [arg.type[ctx.arg_tuple] for arg in all_args]
                        if mod_fullname == 'type_funcs.typefunc_numpy.asarray':
                            for i in range(len(new_args)):
                                new_args[i].aliases = all_args[i].aliases[ctx.arg_tuple]
                        ans = mod_func(ctx, *new_args)
                        ok = True
                    except TypeError:
                        pass
                    if ok:
                        if isinstance(ans, util.CythonType):
                            ans = type_funcs.TypeWithAliases(ans, [])
                        elif isinstance(ans, type_funcs.TypeWithAliases):
                            pass
                        else:
                            raise ValueError('invalid return value from type func: {}'.format(ans))
                        if verbose:
                            print('*** apply_typefunc, arg_tuple after mod_func is {}'.format(arg_tuple))
                            print('   => '.format(arg_tuple), ans)
                        res[arg_tuple] = ans.type
                        res_aliases[arg_tuple] = ans.aliases
                        if verbose:
                            print('*** apply_typefunc, return type is: {}, aliases are: {}'.format(res[arg_tuple], res_aliases[arg_tuple]))
                elif func_fullname in type_signature and func_fullname in function_name_to_node:
                    # Resolve types using a user function in the global scope
                    funcnode = function_name_to_node[func_fullname]
                    target_arg_types = [node_arg.type[arg_tuple] for node_arg in node_args]     # Arguments in positional order
                    argnames = func_arg_names(funcnode)                                         # Arguments in positional order
                    if verbose:
                        print('*** Calling Python function', func_fullname, 'target_arg_types:', target_arg_types)
                    
                    if len(target_arg_types) < len(argnames):
                        # Add default types for missing arguments
                        defaults = funcnode.args.defaults[:len(argnames)-len(target_arg_types)]
                        target_arg_types.extend([default.type.get(arg_tuple, ObjectType) for default in defaults])

                    if verbose:
                        print('*** After adding default args, target_arg_types:', target_arg_types)
                    
                    if len(target_arg_types) == len(argnames):
                        target_arg_tuple = tuple(sorted([(argnames[j], target_arg_types[j]) for j in range(len(argnames))]))
                        if verbose:
                            print('*** After encoding target_arg_tuple:', target_arg_tuple)
                        # TODO: Expand target function with given types even if it is not in type_signature
                        if target_arg_tuple in type_signature[func_fullname]:
                            res[arg_tuple] = type_signature[func_fullname][target_arg_tuple].get(return_value, ObjectType)
                            res_aliases[arg_tuple] = funcnode.aliases.get(target_arg_tuple, [])
                            if verbose:
                                print('*** Got return type and aliases:', res[arg_tuple], res_aliases[arg_tuple])
                            ok = True
                else:
                    if no_prealloc_not_found:
                        parent_functionname = node.parent_functionname
                        if verbose:
                            print('*** Preallocate alias analysis: function {} not handled, called in {}, disabling preallocation for aliases passed in to it'.format(func_fullname, parent_functionname))
#                        if parent_functionname in can_prealloc:
                        for node_arg in node_args:
                            for arg_tuple in arg_tuples(node_arg):
                                for (alias_func, alias_var) in node_arg.aliases[arg_tuple]:
                                    if verbose:
                                        print('   => Disabling preallocation for {}'.format(alias_var))
                                    if alias_func in can_prealloc:
                                        can_prealloc[alias_func][alias_var] = False
                if not ok:
                    res[arg_tuple] = ObjectType
                    res_aliases[arg_tuple] = []
            if verbose:
                print('** apply_typefunc, result: type={}, aliases={}'.format(res, res_aliases))
            node.type = res
            node.aliases = res_aliases
#            if len(node.aliases) == 0:
#                raise ValueError((node.type, node.aliases))
        
        def map_type(self, func, target_nodes, basenode=None, pass_arg_tuple=False, use_node_types=True):
            in_type_d_L = [(target_node.type if use_node_types else target_node) for target_node in target_nodes]
            if basenode is None:
                basenode = target_nodes[0]
            if verbose:
                print('in_type_d_L:', in_type_d_L)
                print('arg_tuples:', arg_tuples(basenode))
            return {arg_tuple: func(*(([] if not pass_arg_tuple else [arg_tuple]) +
                                       [(in_type_d[arg_tuple] if use_node_types else in_type_d) for in_type_d in in_type_d_L])) for arg_tuple in arg_tuples(basenode)}
        def visit_FunctionDef(self, node):
            reset_function_types(node.name)
        
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], Name):
                def vartype_func(_arg):
                    _new_arg = copy.deepcopy(_arg)
                    _new_arg.known_value = None
                    return _new_arg
                self.add_type(node, node.targets[0].id, node.value, vartype_func=vartype_func, alias_node=node.targets[0])
            self.print_info(node)
        def visit_Return(self, node):
            # TODO: Handle case where one branch might return a known type whereas another branch does not call return, thus implicitly returning None
            self.add_type(node, return_value, node.value, alias_node=node.value)
            if node.parent_functionname in function_name_to_node:
                update_aliases(function_name_to_node[node.parent_functionname], node.value)
            self.print_info(node)
        def visit_UnaryOp(self, node):
            if isinstance(node.op, Invert):
                node.type = self.map_type(lambda in_type: (type_from_known_value(~in_type.known_value) if
                             in_type.known_value is not None
                             else (type_from_value(1) if
                                (in_type.is_object() or in_type.shape == () and in_type.cython_type in ['int', 'bool'])
                                else type_from_value(object()))), [node.operand])
            elif isinstance(node.op, Not):
                node.type = self.map_type(lambda in_type: (type_from_known_value(not in_type.known_value) if
                             in_type.known_value is not None
                             else( type_from_value(True) if
                                (in_type.is_object() or in_type.shape == () and in_type.cython_type in ['int', 'float', 'double', 'bool', 'str'])
                                else type_from_value(object()))), [node.operand])
            elif isinstance(node.op, (USub, UAdd)):
                def type_func(in_type):
                    if in_type.cython_type in ['int', 'float', 'double']:
                        if in_type.known_value is not None and isinstance(node.op, USub):
                            return type_from_known_value(-in_type.known_value)
                        else:
                            return in_type
                    elif in_type.cython_type == 'bool':
                        if in_type.shape == ():
                            return type_from_value(1)
                        else:
                            return in_type
                    else:
                        return type_from_value(object())
                node.type = self.map_type(type_func, [node.operand])
            else:
                raise ValueError('unimplemented UnaryOp node type: {}'.format(node.op))
            self.print_info(node)
        def visit_BoolOp(self, node):
            if isinstance(node.op, (And, Or)):
                def type_func(*in_type_L):
#                    return type_from_value(False)
                    if all(in_type_L[i+1] == in_type_L[i] for i in range(len(in_type_L)-1)):
                        if all(in_type_L[i].known_value is not None for i in range(len(in_type_L))):
                            if isinstance(node.op, And):
                                return type_from_known_value(all([in_type_L[i].known_value for i in range(len(in_type_L))]))
                            else:
                                return type_from_known_value(any([in_type_L[i].known_value for i in range(len(in_type_L))]))
                        else:
                            return in_type_L[0]
                    return ObjectType
#                print('node.values:', node.values)
                node.type = self.map_type(type_func, node.values)
#                print('*** Type after type map:', node.type)
            else:
                raise ValueError('unimplemented BoolOp node type: {}'.format(node.op))
            self.print_info(node)
        def visit_BinOp(self, node):
            def type_func(left, right):
                if isinstance(node.op, (Add, Sub, Mult, Div, Mod, FloorDiv, LShift, RShift, BitOr, BitXor, BitAnd)):
                    if verbose:
                        print('BinOp, types:', left, right)
                    ans_type = util.union_cython_types(left, right, numpy_promotion=True)
                    if isinstance(node.op, Div) and ans_type.is_scalar() and ans_type.cython_type == 'int':
                        ans_type = type_from_value(1.0)
                    if left.known_value is not None and right.known_value is not None:
                        new_node = copy.deepcopy(node)
                        if isinstance(left.known_value, str):
                            left_str = "'" + left.known_value + "'"
                        else:
                            left_str = left.known_value.__str__()
                        if isinstance(right.known_value, str):
                            right_str = "'" + right.known_value + "'"
                        else:
                            right_str = right.known_value.__str__()
                        py_ast.replace_node(new_node, new_node.left, py_ast.get_ast(left_str).body[0].value)
                        py_ast.replace_node(new_node, new_node.right, py_ast.get_ast(right_str).body[0].value)
                        try:
                            ans_type.known_value = eval(py_ast.dump_ast(new_node))
                        except:
                            pass
                    return ans_type
                else:
                    raise ValueError('unimplemented BoolOp node type: {}'.format(node.op))  # TODO: Support MatMult
#                print('node.values:', node.values)
            if isinstance(node.op, Pow):
                self.apply_typefunc(node, [node.left, node.right], 'pow')
            else:
                node.type = self.map_type(type_func, [node.left, node.right])
#                print('*** Type after type map:', node.type)
            self.print_info(node)
        def visit_Call(self, node):
            if len(node.keywords) == 0 and node.starargs is None and node.kwargs is None:
                if verbose:
                    print('Call node, apply_typefunc to node: {}'.format(astor.dump(node)))
                fullname0 = fullname = astor.to_source(node.func)
                if '.' in fullname:
                    try:
                        self.apply_typefunc(node, node.args, fullname, stop_if_not_found=True, no_prealloc_not_found=False)
                    except ApplyTypeFuncNotFound:
                        fullname = fullname.split('.')[1]
                        self.apply_typefunc(node, node.args, fullname, method=True, method_selfnode=node.func.value, no_prealloc_not_found=True)
                else:
                    self.apply_typefunc(node, node.args, fullname)
            self.print_info(node)
        def visit_Attribute(self, node):
            try:
                self.apply_typefunc(node, [], node.attr, method=True, method_selfnode=node.value, stop_if_not_found=True, no_prealloc_not_found=True)
            except ApplyTypeFuncNotFound:
                fullname = astor.to_source(node)
                self.apply_typefunc(node, [], fullname, False)
            """
            if node.attr == 'shape' and isinstance(node.value, Name):
                name = node.value.id
                res = {}
                for arg_tuple in arg_tuples(node):
                    if node.type[arg_tuple] == ObjectType:
                        d = type_signature[node.parent_functionname][arg_tuple]
                        if name in d:
                            node.type[arg_tuple] = type_from_known_value(d[name].shape)
                            """
                
        def visit_Name(self, node):
            res = {}
            res_aliases = {}
            for arg_tuple in arg_tuples(node):
                d = type_signature[node.parent_functionname][arg_tuple]
                if node.id in ['False', 'True']:
                    res[arg_tuple] = type_from_value(False)
                elif node.id in d:
                    res[arg_tuple] = d[node.id]
                else:
                    res[arg_tuple] = ObjectType
                if res[arg_tuple].is_array():
                    res_aliases[arg_tuple] = [(node.parent_functionname, node.id)]
                else:
                    res_aliases[arg_tuple] = []
            node.type = res
            node.aliases = res_aliases
            self.print_info(node)
        def visit_NameConstant(self, node):     # The NameConstant class is new in Python 3.4, and represents None/True/False
            res = {}
            for arg_tuple in arg_tuples(node):
                d = type_signature[node.parent_functionname][arg_tuple]
                if node.value in [False, True]:
                    res[arg_tuple] = type_from_known_value(node.value)
                else:
                    res[arg_tuple] = ObjectType
            node.type = res
            self.print_info(node)
        def visit_Expr(self, node):
            node.type = node.value.type
            self.print_info(node)
        def visit_Compare(self, node):
            # left, comparators
            def type_func(*in_type_L):
                if any(in_type.is_object() for in_type in in_type_L):
                    return ObjectType
                if all(not in_type.is_array() for in_type in in_type_L):
                    if all(in_type.known_value is not None for in_type in in_type_L):
                        new_node = copy.deepcopy(node)
                        new_known_values = [in_type.known_value.__str__() if not isinstance(in_type.known_value, str) 
                                            else "'" + in_type.known_value + "'"
                                            for in_type in in_type_L]
                        py_ast.replace_node(new_node, new_node.left, py_ast.get_ast(new_known_values[0]).body[0].value)
                        new_comparators = [py_ast.get_ast(new_known_value).body[0].value 
                                           for new_known_value in new_known_values]
                        new_node.comparators = new_comparators
                        return type_from_known_value(eval(py_ast.dump_ast(new_node)))
                    else:
                        return type_from_value(False)
                else:
                    array_types = [in_type for in_type in in_type_L if in_type.is_array()]
                    union_type = util.union_cython_types_list(array_types, numpy_promotion=True) #functools.reduce(lambda a, b: util.union_cython_types(a, b, numpy_promotion=True), array_types)
                    union_type.set_primitive_type('bool')
                    if all(in_type.known_value is not None for in_type in in_type_L):
                        new_node = copy.deepcopy(node)
                        new_known_values = [in_type.known_value.__str__() if not isinstance(in_type.known_value, str) 
                                            else "'" + in_type.known_value + "'"
                                            for in_type in in_type_L]
                        py_ast.replace_node(new_node, new_node.left, py_ast.get_ast(new_known_values[0]).body[0].value)
                        new_comparators = [py_ast.get_ast(new_known_value).body[0].value 
                                           for new_known_value in new_known_values]
                        new_node.comparators = new_comparators
                        union_type.known_value = eval(py_ast.dump_ast(new_node))
                    return union_type
            node.type = self.map_type(type_func, [node.left] + node.comparators)
            self.print_info(node)
        def visit_Num(self, node):
            typeval = util.CythonType.from_known_value(node.n, program_info)
#            print('*** Num:', node.n, typeval)
            node.type = {arg_tuple: typeval for arg_tuple in arg_tuples(node)}
            self.print_info(node)
        def visit_Str(self, node):
            typeval = util.CythonType.from_known_value(node.s, program_info)
            node.type = {arg_tuple: typeval for arg_tuple in arg_tuples(node)}
        def visit_For(self, node):
            if isinstance(node.target, Name):
                self.add_type(node, node.target.id, node.iter, lambda _type: _type.primitive_type(allow_complex=False, cython_type=True), alias_node=node.target)
            self.print_info(node)
        def visit_Subscript(self, node):
            if isinstance(node.slice, (Index, ExtSlice, Slice)):
                ok = True
                ext_slice = False
                if isinstance(node.slice, ExtSlice):
                    children = node.slice.dims
                    ext_slice = True
                elif isinstance(node.slice, Slice):
                    children = [node.slice]
                    ext_slice = True
                elif isinstance(node.slice, Index):
                    if isinstance(node.slice.value, (Name, Num)):
                        children = [node.slice.value]
                    elif isinstance(node.slice.value, Tuple):
                        children = node.slice.value.elts
                    else:
                        ok = False
                else:
                    raise ValueError('unknown node.slice type')
                if ok:
                    is_primitive = [False]
                    def type_func(arg_tuple, *in_node_L):
                        ans_type = None
                        target = node.value
                        if isinstance(target, Attribute) and target.attr == 'shape':
                            target_arr = target.value
                            if target_arr.type[arg_tuple].is_array():
                                ans_type =  type_from_value(1)
                        elif target.type[arg_tuple].is_array():
                            is_slice = [isinstance(in_node, Slice) for in_node in in_node_L]
                            is_full_slice = [is_slice[i] and in_node_L[i].lower is None and in_node_L[i].upper is None and in_node_L[i].step is None for i in range(len(in_node_L))]
                            index_count = len(in_node_L) - sum(is_slice)
                            n = len(target.type[arg_tuple].shape)
                            if index_count == n:
                                is_primitive[0] = True
                                ans_type =  target.type[arg_tuple].primitive_type(cython_type=True)
                            elif index_count < n:
                                t = copy.deepcopy(target.type[arg_tuple])
                                sub_shape = []
                                for i in range(len(t.shape)):
                                    if i < len(in_node_L):
                                        if is_slice[i]:
                                            if is_full_slice[i]:
                                                sub_shape.append(t.shape[i])
                                            else:
                                                try:
                                                    if in_node_L[i].step is None:
                                                        step_shape = abs(in_node_L[i].upper.n - in_node_L[i].lower.n)
                                                        sub_shape.append(step_shape)
                                                    else:
                                                        sub_shape.append(step_shape)
                                                except:
                                                    sub_shape.append(None)
                                    else:
                                        sub_shape.append(t.shape[i])
#                                print('index_count:', index_count)
#                                print('n:', n)
#                                print('in_node_L:', in_node_L)
#                                print('t:', t)
#                                print('sub_shape:', sub_shape)
                                t.set_shape(sub_shape) #t.shape[len(in_node_L):])
#                                if ext_slice:
#                                    sys.exit(1)
                                ans_type = t
                        if ans_type is not None:
                            if target.type[arg_tuple].known_value is not None and all([arg.type[arg_tuple].known_value is not None for arg in in_node_L]):
                                try:
                                    if len(in_node_L) == 1:
                                        index_tuple = in_node_L[0].type[arg_tuple].known_value
                                    else:
                                        index_tuple = tuple(arg.type[arg_tuple].known_value for arg in in_node_L)
                                    ans_type.known_value = target.type[arg_tuple].known_value[index_tuple]
                                except:
                                    pass
                            return ans_type
                        return ObjectType
                    node.type = self.map_type(type_func, children, pass_arg_tuple=True, use_node_types=False)
        
                    if verbose:
                        print('Subscript, node {}, is_primitive={}'.format(astor.dump(node), is_primitive[0]))
                    if not is_primitive[0]:
                        if verbose:
                            print('Subscript, not primitive result for node {}'.format(astor.dump(node)))
                        if isinstance(node.value, Name):
                            for arg_tuple in arg_tuples(node):
                                node.aliases[arg_tuple] = node.value.aliases[arg_tuple]#functools.reduce(list_union, [child.aliases[arg_tuple] for child in children])
            self.print_info(node)
        def visit_Index(self, node):
            self.print_info(node)
        def visit_If(self, node):
            self.print_info(node)           # TODO: Is this method necessary?
        def visit_While(self, node):
            self.print_info(node)           # TODO: Is this method necessary?
        def visit_Tuple(self, node):
            for arg_tuple in arg_tuples(node):
                node.type[arg_tuple] = util.CythonType.from_cython_type(tuple([sub.type[arg_tuple] for sub in node.elts]), program_info)
                node.type[arg_tuple].known_value = tuple(sub.type[arg_tuple].known_value for sub in node.elts)
            self.print_info(node)
        def visit_List(self, node):
            res_type = {}
            success = True
            for arg_tuple in arg_tuples(node):
                if len(node.elts):
                    res_type[arg_tuple] = util.CythonType.from_cython_type([util.union_cython_types_list([sub.type[arg_tuple] for sub in node.elts])]*len(node.elts), program_info)
                    res_type[arg_tuple].known_value = [sub.type[arg_tuple].known_value for sub in node.elts]
                else:
                    success = False
                    break
            if success:
                node.type = res_type
            self.print_info(node)
#        def generic_visit(self, node):
#            print('generic_visit:', dump(node))
#            RecursiveVisitor.generic_visit(self, node)

    def walk_type_signature(ts):
        for funcname in ts:
            for arg_tuple in ts[funcname]:
                d = ts[funcname][arg_tuple]
                yield (funcname, arg_tuple, d)

    ninferences = 0
    while True:
        if verbose:
            util.print_header('Performing inference {}'.format(ninferences+1))
        last_type_signature = copy.deepcopy(type_signature)
        InferTypes().visit(m0)
        ninferences += 1
        if verbose:
            util.print_header('Inference {} type signature:'.format(ninferences))
            pprint.pprint(type_signature)
            util.print_header('Inference {} last type signature:'.format(ninferences))
            pprint.pprint(last_type_signature)
        
        if type_signature == last_type_signature:
            break
        
        # Early terminate if all types are inferred
        success = True
        for (funcname, arg_tuple, d) in walk_type_signature(type_signature):
            for value in d.values():
                if value == ObjectType:
                    success = False
        if success:
            break

    if prealloc:                # Take an extra pass to track aliasing information
        InferTypes().visit(m0)

    for (funcname, arg_tuple, d) in walk_type_signature(type_signature):
        if return_value in d:
            d[return_value_tuple] = util.CythonType.from_cython_type((d[return_value],), program_info)

    if verbose:
        util.print_header('Annotated AST after inference:')
        print(dump_ast_types(m0))
        util.print_header('Inferred type signatures:')
        pprint.pprint(type_signature)
#    print(type_signature)

    if verbose:
        util.print_header('can_prealloc before additional checks:')
        pprint.pprint(can_prealloc)

    name_nodes_aliased_against = {}        # Alias function name => alias var name => Name node funcname => set of name node strs aliased against in funcname.
                                           # Where a => b indicates a dictionary mapping from key a to value b.
    call_nodes_aliased_against = {}        # Alias function name => alias var name => Function name that call is made from => List of function calls aliased against

    class InferPrealloc(BottomUpVisitor):
        def visit_Name(self, node):
            varname = node.id
            funcname = node.parent_functionname
            
            for arg_tuple in arg_tuples(node):
                for (alias_func, alias_var) in node.aliases[arg_tuple]:
                    name_nodes_aliased_against.setdefault(alias_func, {})
                    name_nodes_aliased_against[alias_func].setdefault(alias_var, {})
                    name_nodes_aliased_against[alias_func][alias_var].setdefault(funcname, set())
                    name_nodes_aliased_against[alias_func][alias_var][funcname].add(node.id)
                    #name_nodes_aliased_against[alias_func][alias_var].append(node)

        def visit_Call(self, node):
            if len(node.keywords) == 0 and node.starargs is None and node.kwargs is None:
                fullname = astor.to_source(node.func)
                parent_functionname = node.parent_functionname
                if '.' not in fullname:
                    for arg_tuple in arg_tuples(node):
                        for (alias_func, alias_var) in node.aliases[arg_tuple]:
                            call_nodes_aliased_against.setdefault(alias_func, {})
                            call_nodes_aliased_against[alias_func].setdefault(alias_var, {})
                            call_nodes_aliased_against[alias_func][alias_var].setdefault(parent_functionname, [])
                            call_nodes_aliased_against[alias_func][alias_var][parent_functionname].append(fullname)

    if prealloc:
        InferPrealloc().visit(m0)
        if verbose:
            util.print_header('name_nodes_aliased_against (before removal of spurious aliases):')
            pprint.pprint(name_nodes_aliased_against)
            util.print_header('call_nodes_aliased_against:')
            pprint.pprint(call_nodes_aliased_against)

        func_varname_to_call_nodes = {}
        class RemoveSpuriousAliasVisitor(BottomUpVisitor):
            def visit_Call(self, node):
                funcname = node.parent_functionname
                name_nodes_inside = py_ast.find_all(node, Name)
                for name_node in name_nodes_inside:
                    name_node_name = name_node.id
                    func_varname_to_call_nodes.setdefault((funcname, name_node_name), [])
                    func_varname_to_call_nodes[(funcname, name_node_name)].append(node)
#                    if not hasattr(name_node, 'call_nodes'):
#                        name_node.call_nodes = []
#                    name_node.call_nodes.append(node)
        
        RemoveSpuriousAliasVisitor().visit(m0)
        if verbose:
            util.print_header('func_varname_to_call_nodes:')
            pprint.pprint(func_varname_to_call_nodes)
        
        # If a variable v is aliased against a different variable v', check if a second variable v' is always used in side-effect free
        # operations (in our case, just math operations), and never passed to any functions. If so then the alias can be removed.
        # TODO: also check that not assigned to any global variable.
        for alias_func in name_nodes_aliased_against:
            for alias_var in name_nodes_aliased_against[alias_func]:
                for funcname_p in name_nodes_aliased_against[alias_func][alias_var]:
                    varnames_remove = []
                    for varname_p in name_nodes_aliased_against[alias_func][alias_var][funcname_p]:
                        if (alias_func, alias_var) == (funcname_p, varname_p):
                            continue
                        if len(func_varname_to_call_nodes.get((funcname_p, varname_p), [])) == 0:
                            varnames_remove.append(varname_p)
                    for varname_p in varnames_remove:
                        name_nodes_aliased_against[alias_func][alias_var][funcname_p].remove(varname_p)

        if verbose:
            util.print_header('name_nodes_aliased_against (after removal of spurious aliases):')
            pprint.pprint(name_nodes_aliased_against)

        for alias_func in name_nodes_aliased_against:
            for alias_var in name_nodes_aliased_against[alias_func]:
                for funcname in name_nodes_aliased_against[alias_func][alias_var]:
                    nodeL = name_nodes_aliased_against[alias_func][alias_var][funcname]
                    if len(nodeL) > 1:
                        if alias_func in can_prealloc:
                            can_prealloc[alias_func][alias_var] = False

        for alias_func in call_nodes_aliased_against:
            for alias_var in call_nodes_aliased_against[alias_func]:
                for parent_funcname in call_nodes_aliased_against[alias_func][alias_var]:
                    callL = call_nodes_aliased_against[alias_func][alias_var][parent_funcname]
                    if len(callL) > 1:
                        if alias_func in can_prealloc:
                            can_prealloc[alias_func][alias_var] = False

        for (funcname, arg_tuple, d) in walk_type_signature(type_signature):
            for (varname, vartype) in d.items():
                if not vartype.is_array():
                    can_prealloc[funcname][varname] = False

        for (funcname, arg_tuple, d) in walk_type_signature(type_signature):
            for (varname, vartype) in d.items():
                if vartype.is_array():
                    can_prealloc[funcname].setdefault(varname, True)

    prealloc_return = {}
    for funcname in can_prealloc:
        prealloc_return.setdefault(funcname, [])
        for (varname, var_success) in can_prealloc[funcname].items():
            if var_success:
                prealloc_return[funcname].append(varname)
    
    if verbose:
        util.print_header('can_prealloc (final):')
        pprint.pprint(can_prealloc)
    
    # Convert return type to same format as compiler.get_types():
    # a dict mapping function name to a list of type configurations used for each function, where a type
    # configuration is a dict mapping local variable name => type.

    type_signature = {typesig_key: typesig_value for (typesig_key, typesig_value) in type_signature.items() if len(typesig_value)}

    type_config_d = {}
    for funcname in type_signature:
        type_config_d[funcname] = [item[1] for item in sorted(type_signature[funcname].items(), key=lambda item: item[0])]

    if verbose:
        util.print_header('Inferred type signatures (get_types() format):', '')
        pprint.pprint(type_config_d)

    if verbose:
        print('Number of inferences:', ninferences)

    if get_macros:
        macroL = macros.find_macros_ast(m0, macros.macros, True)
        for macro in macroL:
            if verbose:
                print('Macro:', macro.line, macro.node_str)
            try:
                parent_function_name = parent_function(m0, macro.node).name
            except KeyError:
                if verbose:
                    warnings.warn('macro {} could not find parent function'.format(macro))
                continue
            for (funcname, arg_tuple, d) in walk_type_signature(type_signature):
                if funcname != parent_function_name:
                    continue
                macro_key = util.types_non_variable_prefix + macro.node_str
                macro_argtypes = []
                for arg_node in macro.arg_nodes:
                    if verbose:
                        print(astor.to_source(arg_node))
                    t = arg_node.type[arg_tuple]
                    macro_argtypes.append(t)
                macro_argtypes = tuple(macro_argtypes)
                if verbose:
                    print(macro_argtypes)
                if len(macro_argtypes):
                    if verbose:
                        print(type(macro_argtypes[0]))
                d[macro_key] = util.CythonType.from_cython_type(macro_argtypes, program_info)

    return {'types': type_config_d, 'types_internal': type_signature, 'prealloc_internal': can_prealloc, 'prealloc': prealloc_return, 'annotated_type': annotated_type}

simple_test_source = """
import numpy
import numpy as np
import math
def h():
    return ~1
def g(a):
    return -a * 2
def f(a, b=3.5, *arg, **kw):
  i = 0
  i = 1
  q = 4.0
  q = 4.5
  bcopy = -b
  pow_val = 2 ** 2
  pow_val2 = 3 ** -1
#  pow_val3 = 3 ** i
  z = numpy.zeros(3)
  z2 = numpy.zeros((3,1))
  z3 = numpy.array([0.0, 0.0, 0.0])
  z4 = numpy.array([[0.0], [0.0], [0.0]])
  z4_3by2 = numpy.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
  z3_b = numpy.array(z3)
  z4_b = numpy.array(z4)
  z4_c = numpy.array(z4, 'float32')
  z4_d = numpy.array(z4, 'float')
  z5 = np.zeros([5,4,z4_d.shape[0]//2+1])
  bool_val = True or 1 > 2 or False
  pow_val4 = z**2
  pow_val5 = numpy.square(z)
  abs_arr = abs(z4)
  sum_arr = z3 + 1.0
  comp_arr = z3 < 2.0
  dot = numpy.dot(z3, [0.0, 0.0, 0.0])
  norm = numpy.linalg.norm(z2)
#  norm_object_type = numpy.linalg.norm(z2, 1, 0)
  cos_val = math.cos(1.5)
  tan_val = numpy.tan(cos_val)
  clip_arr = numpy.clip(sum_arr, 0.0, 1.0)
  A = numpy.ones((3,3)) < numpy.ones((3,3))
  L = numpy.ones((10,11))
  pi_e = math.sin(math.pi) + math.pi + numpy.e + math.e + numpy.pi + np.e
  for j in range(10):
    g_float = g(float(j))
    scalar_val = L[j, j]
    slice_val = L[j]
    L[j, j] = 2.0
  while False and True and 1 < 2:
    if True:
      g(1)
  ans = g(int(a)) + b + h()
  i
  q
  return ans
def test():
  f(1, 2)
  f(1.0, 2.0)
if __name__ == '__main__':
  test()
"""

simple_test_source2 = """
import numpy
def f():
  z3 = numpy.array([0.0, 0.0, 0.0])
  z3 < 2.0
def test():
  f()
if __name__ == '__main__':
  test()
"""

simple_test_source3 = """
import numpy
def copymat(a):
    return numpy.zeros(a.shape)
def f():
    Ad = numpy.ones(3)
    Af = numpy.ones(3, 'float32')
    Bd = copymat(Ad)
    Bd2 = copymat(Af)
    C = numpy.array([160.0, 240.0, (- 800.0)])
def test():
  f()
if __name__ == '__main__':
  test()
"""

def test_type_infer_simple(verbose=False, prealloc=False, source_list=None):
    import compiler
    if source_list is None:
        source_list = [simple_test_source, simple_test_source3]
        if prealloc:
            source_list = [open('test_programs/prealloc_test.py','rt').read()]
    else:
        source_list = [open(x, 'rt').read() for x in source_list]
    for s in source_list:
        program_info = compiler.ProgramInfo(s, preallocate=False)
        type_infer_d = type_infer(program_info, verbose=verbose, get_macros=True)
        
        if prealloc:
            prealloc = type_infer_d['prealloc_internal']
            for funcname in prealloc:
                for varname in prealloc[funcname]:
                    status = prealloc[funcname][varname]
                    if status and '_noprealloc' in varname:
                        assert False, (funcname, varname, status, prealloc)
                    if not status and '_prealloc' in varname:
                        assert False, (funcname, varname, status, prealloc)
                    if status:
                        assert varname in type_infer_d['prealloc'][funcname]
                    if not status:
                        assert varname not in type_infer_d['prealloc'][funcname]

#    print(str(main_types['A']))
#    assert(str(main_types['A']) == "'numpy.ndarray[numpy.float32_t, ndim=2](shape=(2, 2),shape_list=[])'")
#    pprint.pprint(type_infer_d['types'])
#    pprint.pprint(main_types)

if __name__ == '__main__':
    test_type_infer_simple(True)
    #test_type_infer_simple(True, True)
#    test_type_infer_simple(True, source_list=['test_programs/simple_pm.py'])
#    test_type_infer_annotations()
