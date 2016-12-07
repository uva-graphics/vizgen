from transforms_util import *
from transforms_base import BaseTransform

class Preallocate(BaseTransform):
    def apply(self, s, extra_info=None):
        """
        Applies array preallocation to input program s. If extra_info is not None then stores number of arrays pre-allocated in extra_info['preallocate_count'].
        """
        return preallocate_array_variables(self.program_info, s, self.program_info.preallocate_arrays, extra_info=extra_info, intermediate=True)
    
    def mutate(self):
        self.line = 1
        self.orig_num = 1
        
def preallocate_array_variables(program_info, s, array_vars, extra_info=None, intermediate=False):
    """
    Preallocate given array variables.
    
    Given program string s and array_vars, which maps function name string to a list of variables in each function
    to be preallocated (the same as the return value of preallocate_find_arrays()), returns a modified program string
    that preallocates the given variables. If r is not None then assumes redbaron.RedBaron instance r was constructed
    from s. If intermediate is True then replaces each preallocated variable's assignment line with an "intermediate"
    string that is later converted when finalizing Cython code (postprocessing) to the target line.
    """
    r = py_ast.get_ast(s)
    py_ast.add_parent_info(r)
    defnodes = py_ast.find_all(r, ast.FunctionDef)
    
    assign_count = 0
    
    assert isinstance(array_vars, dict)
    global_lines = []
    for func_name in array_vars:
        defnode = [tnode for tnode in defnodes if tnode.name == func_name][-1]
        assert isinstance(array_vars[func_name], (list, tuple))
        for array_var in array_vars[func_name]:
            (assignnode, call_args, numpy_func) = preallocate_find_assignnode_py_ast(r, func_name, array_var)
            if numpy_func not in preallocate_empty_funcs:
                init_str = preallocate_find_init(defnode, array_var, assignnode, call_args, r) if use_py_ast_preallocate_init else ''
            else:
                init_str = ''
            #print('*** Preallocate: assignnode=', assignnode, 'call_args=', call_args, 'numpy_func=', numpy_func, 'init_str=', init_str)
            rhs_str = py_ast.dump_ast(assignnode.value)
            
            if numpy_func in preallocate_numpy_funcs_array_arg:
                current_shape = call_args + '.shape'
            else:
                current_shape = call_args
                if not isinstance(py_ast.get_ast(current_shape).body[0].value, (ast.Tuple, ast.List)) and not current_shape.endswith('.shape'):
                    current_shape = '(' + current_shape + ',)'
            
            array_var_init = preallocate_prefix + array_var + '_init'
            array_var_global = preallocate_prefix + array_var + '_global'
            
            global_lines.append("{array_var_init} = False\n{array_var_global} = None\n".format(**locals()))
            iter_var = preallocate_prefix + 'j'
            
            shape_is_larger = 'any([{}[{}] > {}.shape[{}] for {} in range(len({}.shape))])'.format(current_shape, iter_var, array_var_global, iter_var, iter_var, array_var_global)
            
            if use_py_ast_preallocate_grow_shrink:
                array_length = len(program_info.types[defnode.name][0][array_var].shape)
                grow_shrink_index = []
                for i in range(array_length):
                    grow_shrink_index.append(':' + current_shape + '[' + str(i) + ']')
                grow_shrink_str = ','.join(grow_shrink_index)
            
            rhs_str_float32 = rhs_str.strip()
            if rhs_str_float32.endswith(')') and not rhs_str_float32.endswith("'float32')") and not rhs_str_float32.endswith('"float32")'):
                rhs_str_float32 = rhs_str[:-1] + ', "float32")'
               
            assign_newL = []   
            for rhs_str_current in [rhs_str, rhs_str_float32]:
                if not use_py_ast_preallocate_grow_shrink:
                    assign_newL.append("""
global {array_var_global}, {array_var_init}
if not {array_var_init} or {shape_is_larger}:
    {array_var_init} = True
    {array_var_global} = {rhs_str_current}
{array_var} = {array_var_global}
        """.format(**locals()))
                else:
                    assign_newL.append("""
global {array_var_global}, {array_var_init}
if not {array_var_init} or {shape_is_larger}:
    {array_var_init} = True
    {array_var_global} = {rhs_str_current}
{array_var} = {array_var_global}[{grow_shrink_str}]
        """.format(**locals()))
                
            if intermediate:
                assign_new = cython_preallocate_intermediate + ' ' + array_var + ' ' + ' '.join(binascii.hexlify((assign_new_str + init_str).encode('ascii')).decode('ascii') for assign_new_str in assign_newL)
            else:
                assign_new = assign_newL[0] + init_str
                
            """replace_node = py_ast.get_ast('pass')
            replace_node = replace_node.body[0]
            py_ast.replace_node(r, assignnode, replace_node)
            
            add_node = py_ast.dump_ast(assign_new)
            add_node = add_node.body[0]
            py_ast.add_before_node(r, assignnode, add_node)
            assign_count += 1"""
            
            """
            if there exist vectorize transforms right before preallocation, delete it
            this causes a bug in interpolate.py
            where VectorizeInnermost() is tagged on an assignnode where it should been preallocated
            """
            if hasattr(assignnode.parent, 'body'):
                assignnode_index = assignnode.parent.body.index(assignnode)
                if assignnode_index > 0:
                    node_prev = assignnode.parent.body[assignnode_index - 1]
                    if isinstance(node_prev, ast.Expr):
                        if py_ast.dump_ast(node_prev).startswith('#transform(VectorizeInnermost())'):
                            del assignnode.parent.body[assignnode_index - 1]
                            
            replace_node = py_ast.get_ast(assign_new)
            if len(replace_node.body) == 1:
                py_ast.replace_node(r, assignnode, replace_node.body[0])
            elif len(replace_node.body) > 1:
                for i in range(len(replace_node.body)):
                    py_ast.add_before_node(r, assignnode, replace_node.body[i])
                py_ast.replace_node(r, assignnode, py_ast.get_ast('pass').body[0])
            #py_ast.replace_node(r, assignnode, replace_node)
            assign_count += 1
            
    if extra_info is not None:
        extra_info['preallocate_count'] = assign_count
            
    sp = py_ast.dump_ast(r)
    L = sp.split('\n')
            
    last_import = 0
    for (i, line) in enumerate(L):
        if line.startswith('import'):
            last_import = i
    L[last_import+1:last_import+1] = global_lines

    sp = '\n'.join(L)

    return sp

def preallocate_find_init(defnode, array_var, assignnode, call_args, rootnode):
    """
    use pattern matching to determine if array_var has to be initialized to be all 0s each time
    use call_args to infer shape of the array
    """
    potential_assignnodes = py_ast.find_all(defnode, ast.Assign)
    assignnodes = []
    
    if assignnode is not None:
        dimensions = parse_call_args(defnode, array_var, call_args, [])
        is_constant = []
        constant_dim = []
        for i in range(len(dimensions)):
            for j in range(len(dimensions[i])):
                if len(is_constant) <= j:
                    if py_ast.is_int_constant_py_ast(dimensions[i][j]):
                        is_constant.append(True)
                        constant_dim.append(eval(dimensions[i][j]))
                    else:
                        is_constant.append(False)
                        constant_dim.append(-1)
                elif is_constant[j] == False and py_ast.is_int_constant_py_ast(dimensions[i][j]):
                    is_constant[j] = True
                    constant_dim.append(eval(dimensions[i][j]))
    else:
        dimensions = [call_args]
        constant_dim = call_args
        is_constant = [dim >=0 for dim in constant_dim]
        
    assign_tuples = []
    assign_strs = []
    assign_areas = []
    assign_skip = []
    
    for potential_assignnode in potential_assignnodes:
        try:
            if len(potential_assignnode.targets) == 1:
                if isinstance(potential_assignnode.targets[0], ast.Subscript) and isinstance(potential_assignnode.targets[0].value, ast.Name) and potential_assignnode.targets[0].value.id == array_var and potential_assignnode != assignnode:
                    assignnodes.append(potential_assignnode)
                    (getitem_tuple, getitem_strs) = parse_array_slice_py_ast(potential_assignnode.targets[0])
                    assign_tuples.append(getitem_tuple)
                    assign_strs.append(getitem_strs)
                    assign_skip.append(False)
                elif isinstance(potential_assignnode.targets[0], ast.Name) and potential_assignnode.targets[0].id == array_var and potential_assignnode != assignnode:
                    assignnodes.append(potential_assignnode)
                    assign_tuples.append([])
                    assign_strs.append([])
                    assign_skip.append(False)
        except:
            pass
    
    """
    is ther any more assignment for this array?
    """
    #if len(assignnodes) == 0:
     #   return '\n' + array_var + '[:] = 0.0'
    
    """
    find array reads so as to check if they ever use the initial value
    """
    namenodes = py_ast.find_all(defnode, ast.Name)
    resolve_reads = []
    for namenode in namenodes:
        if namenode.id == array_var:
            if isinstance(namenode.parent, ast.Subscript):
                if isinstance(namenode.parent.ctx, ast.Load):
                    resolve_reads.append(namenode)
                elif isinstance(namenode.parent.parent, ast.AugAssign):
                    if namenode.parent.parent.target == namenode.parent:
                        resolve_reads.append(namenode)
            elif isinstance(namenode.parent, ast.Return):
                resolve_reads.append(namenode)
            elif isinstance(namenode.parent, ast.AugAssign):
                if namenode.parent.target == namenode:
                    resolve_reads.append(namenode)
            elif isinstance(namenode.ctx, ast.Load):
                if not isinstance(namenode.parent, ast.Attribute):
                    resolve_reads.append(namenode)
                else:
                    if not namenode.parent.attr == 'shape':
                        resolve_reads.append(namenode)
    
    if len(resolve_reads) == 0:
        return ''
    
    defnodes = py_ast.find_all(rootnode, ast.FunctionDef)
    if len(assignnodes) == 0:
        ok = False
        for namenode in resolve_reads:
            try:
                call_field = namenode.parent.parent.arg
                func_name = namenode.parent.parent.parent.func.attr
                mod_name = namenode.parent.parent.parent.func.value.id
                ind = namenode.parent.elts.index(namenode)
                if call_field == 'input_imgL' and func_name == 'test_image_pipeline' and mod_name == 'util':
                    keywords = namenode.parent.parent.parent.keywords
                    call_defname = [key.value.id for key in keywords if key.arg == 'image_func'][-1]
                    defnode_call = [tnode for tnode in defnodes if tnode.name == call_defname][-1]
                    array_var_call = defnode_call.args.args[ind].arg
                    init_str = preallocate_find_init(defnode_call, array_var_call, None, constant_dim, rootnode)
                    if init_str == '':
                        ok = True
                    else:
                        break
            except AttributeError:
                pass

        if not ok:
            return '\n' + array_var + '[:] = 0.0'
        else:
            return ''
    
    """
    try match assigned_dimensions to dimensions acquired from call_args
    """
    
    """
    pattern 1: find pattern a[x, y, :] where x, y are iterators that loops through a.shape
    pattern 2: find pattern a = b
    """
    for i in range(len(assign_tuples)):
        
        if assign_skip[i]:
            continue
        
        getitem_tuple = assign_tuples[i]
        getitem_strs = assign_strs[i]
        
        if getitem_tuple == [] and getitem_strs == []:
            assign_areas.append({'start':assignnodes[i], 'area':[], 'assignnode':assignnodes[i], 'assignstr':[], 'if_list':[]})
            continue
        
        outer_forloop = None
        if_list = []
        
        ok = False
        for j in range(len(getitem_tuple)):
            if isinstance(getitem_tuple[j], (ast.Index, ast.Name, ast.Num)):
                
                index_name = getitem_tuple[j].value if isinstance(getitem_tuple[j], ast.Index) else getitem_tuple[j]
                try:
                    replace_int = None
                    if isinstance(index_name, ast.Name):
                        replace_str = replace_scalar_assign(defnode, index_name)
                        if py_ast.is_int_constant_py_ast(replace_str):
                            replace_int = eval(replace_str)
                    else:
                        replace_int = index_name.n
                    if replace_int is not None:
                        if is_constant[j]:
                            dimension_match = [replace_int]
                            for k in range(len(assign_tuples)):
                                if i != k and assignnodes[i].parent == assignnodes[k].parent:
                                    try:
                                        if not isinstance(assign_tuples[k][j], (ast.Num)):
                                            replace_str = replace_scalar_assign(defnode, assign_tuples[k][j].value if isinstance(assign_tuples[k][j], ast.Index) else assign_tuples[k][j])
                                        else:
                                            replace_str = str(assign_tuples[k][j].n)
                                        if py_ast.is_int_constant_py_ast(replace_str):
                                            dimension_match.append(eval(replace_str))
                                            assign_skip[k] = True
                                    except:
                                        pass
                                    
                            constant_ok = True
                            for k in range(constant_dim[j]):
                                if not k in dimension_match:
                                    constant_ok = False
                                    break
                            if not constant_ok:
                                ok = False
                                break
                            else:
                                ok = True
                                continue
                except AttributeError: 
                    pass
                except ValueError:
                    pass
                """
                start match ind with possible dimensions
                """
                parent = assignnodes[i].parent
                while parent is not None:
                    try:
                        if isinstance(parent, ast.For):
                            if parent.target.id == index_name.id and parent.iter.func.id == 'range':
                                """
                                checks if the variable in range(w) has been assigned before as
                                w = input.shape[0]
                                """
                                if len(parent.iter.args) == 1:
                                    arg_cmp = 0
                                else:
                                    arg_cmp = 0
                                    for arg_ind in range(len(parent.iter.args)):
                                        if not py_ast.is_int_constant_py_ast(py_ast.dump_ast(parent.iter.args[arg_ind])):
                                            arg_cmp = arg_ind
                                            break
                                        
                                arg_strs = [py_ast.dump_ast(parent.iter.args[arg_cmp])]
                                
                                """
                                try to find pattern w = a, for i in range(w):
                                """
                                namenodes = py_ast.find_all(parent.iter.args[arg_cmp], ast.Name)
                                for namenode in namenodes:
                                    try:
                                        replace_str = replace_scalar_assign(defnode, namenode)
                                        current_args_len = len(arg_strs)
                                        for arg_ind in range(current_args_len):
                                            replace_node = py_ast.get_ast(replace_str).body[0].value
                                            py_ast.replace_node(parent, namenode, replace_node)
                                            arg_strs.append(py_ast.dump_ast(parent.iter.args[arg_cmp]))
                                            py_ast.add_parent_info(parent)
                                            py_ast.replace_node(parent, replace_node, namenode)
                                    except ValueError:
                                        pass
                                
                                """
                                try to find pattern output=np.zeros(input.shape), for i in range(out.shape):
                                """
                                current_args_len = len(arg_strs)
                                for arg_ind in range(current_args_len):
                                    a_node = py_ast.get_ast(arg_strs[arg_ind])
                                    py_ast.add_parent_info(a_node)
                                    attrnodes = py_ast.find_all(a_node, ast.Attribute)
                                    for attrnode in attrnodes:
                                        if attrnode.attr == 'shape' and isinstance(attrnode.value, ast.Name) and isinstance(attrnode.parent, ast.Subscript):
                                            if attrnode.value.id != array_var and isinstance(attrnode.parent.slice, ast.Index):
                                                if isinstance(attrnode.parent.slice.value, ast.Num):
                                                    arg_recorded = False
                                                    for dimension in dimensions:
                                                        if len(dimension) == 1:
                                                            if dimension[0] + '[' + str(attrnode.parent.slice.value.n) + ']' in arg_strs[arg_ind]:
                                                                arg_recorded = True
                                                                break
                                                        elif len(dimension) > attrnode.parent.slice.value.n:
                                                            if dimension[attrnode.parent.slice.value.n] in arg_strs[arg_ind]:
                                                                arg_recorded = True
                                                                break
                                                    if not arg_recorded:
                                                        try:
                                                            (arg_assignnode, arg_call_args, arg_assign_funcname) = preallocate_find_assignnode_simple_py_ast(defnode, defnode.name, attrnode.value.id)
                                                            extra_dims = parse_call_args(defnode, attrnode.value.id, arg_call_args, [])
                                                            arg_to_replace = py_ast.dump_ast(attrnode.parent)
                                                            for extra_dim in extra_dims:
                                                                if len(extra_dim) == 1:
                                                                    arg_strs.append(arg_strs[arg_ind].replace(arg_to_replace, extra_dim[0] + '[' + str(attrnode.parent.slice.value.n) + ']'))
                                                                elif len(extra_dim) > attrnode.parent.slice.value.n:
                                                                    arg_strs.append(arg_strs[arg_ind].replace(arg_to_replace, extra_dim[attrnode.parent.slice.value.n]))
                                                        except:
                                                            pass
                                for dimension in dimensions:
                                    for arg_str in arg_strs:
                                        if assignnode is not None:
                                            if len(dimension) == 1 and dimension[0] + '[' + str(j) + ']' in arg_str:
                                                if dimension[0] + '[' + str(j) + ']' == arg_str:
                                                    ok = True
                                                    if outer_forloop is None:
                                                        outer_forloop = parent
                                                    break
                                                else:
                                                    try:
                                                        eval(arg_str.replace(dimension[0] + '[' + str(j) + ']', ''))
                                                        ok = True
                                                        if outer_forloop is None:
                                                            outer_forloop = parent
                                                        break
                                                    except SyntaxError:
                                                        pass
                                            if len(dimension) > j and dimension[j] in arg_str:
                                                if dimension[j] == arg_str:
                                                    ok = True
                                                    if outer_forloop is None:
                                                        outer_forloop = parent
                                                    break
                                                else:
                                                    try:
                                                        eval(arg_str.replace(dimension[j], ''))
                                                        ok = True
                                                        if outer_forloop is None:
                                                            outer_forloop = parent
                                                        break
                                                    except SyntaxError:
                                                        pass

                                        if array_var + '.shape[' + str(j) + ']' in arg_str:
                                            if array_var + '.shape[' + str(j) + ']' == arg_str:
                                                ok = True
                                                if outer_forloop is None:
                                                    outer_forloop = parent
                                                break
                                            else:
                                                try:
                                                    eval(arg_str.replace(array_var + '.shape[' + str(j) + ']', ''))
                                                    ok = True
                                                    if outer_forloop is None:
                                                        outer_forloop = parent
                                                    break
                                                except:
                                                    pass
                                    
                                    if ok:
                                        break
                                if ok:
                                    break
                            elif parent == outer_forloop:
                                outer_forloop = None
                        elif isinstance(parent, ast.If):
                            if_list.append(parent)
                        parent = parent.parent
                    except:
                        parent = parent.parent
                if parent is None:
                    ok = False
                if not ok:
                    break
            elif getitem_strs[j] == ':':
                ok = True
            else:
                ok = False
                break
        if ok:
            assign_areas.append({'start':outer_forloop if outer_forloop is not None else assignnodes[i], 'area':[], 'assignnode':assignnodes[i], 'assignstr':assign_strs[i], 'if_list':if_list})
            
    """
    case 1:if the assignment is inside if statements, move start up one level when two branches of if are both reached
    case 2"if the between assignnode and fornode exist if statement, resolve it before proceeed
    """
    assign_area_to_remove = []
    for i in range(len(assign_areas)):
        start_i = assign_areas[i]['start']
        if isinstance(start_i.parent, ast.If):
            for j in range(len(assign_areas)):
                start_j = assign_areas[j]['start']
                if i != j and start_i.parent == start_j.parent:
                    if start_i in start_i.parent.body and start_j in start_i.parent.orelse:
                        assign_areas[i]['start'] = start_i.parent
                        assign_areas[j]['start'] = start_j.parent
                        break
                    elif start_i in start_i.parent.orelse and start_j in start_i.parent.body:
                        assign_areas[i]['start'] = start_i.parent
                        assign_areas[j]['start'] = start_j.parent
                        break
        if_list_i = assign_areas[i]['if_list']
        if len(if_list_i) > 0:
            for k in range(len(if_list_i)):
                if_node = if_list_i[k]
                ok = False
                for j in range(len(assign_areas)):
                    if_list_j = assign_areas[j]['if_list']
                    if if_node in if_list_j:
                        parent_field_i = find_parent_field(if_node, assign_areas[i]['assignnode'])
                        parent_field_j = find_parent_field(if_node, assign_areas[j]['assignnode'])
                        if parent_field_i == 'body' and parent_field_j == 'orelse':
                            assign_areas[j]['if_list'].remove(if_node)
                            ok = True
                            break
                        elif parent_field_j == 'body' and parent_field_i == 'orelse':
                            assign_areas[j]['if_list'].remove(if_node)
                            ok = True
                            break
                if not ok:
                    break
            if not ok:
                assign_area_to_remove.append(i)
            else:
                assign_areas[i]['if_list'] = []
    
    for ind in assign_area_to_remove[::-1]:
        del assign_area_to_remove[ind]
                        
    not_resolved = []
    
    for namenode in resolve_reads:
        ok = False
        if isinstance(namenode.parent, ast.Subscript):
            (getitem_tuple, getitem_strs) = parse_array_slice_py_ast(namenode.parent)
        else:
            getitem_tuple = []
            getitem_strs = []
        for assign_dict in assign_areas:
            start = assign_dict['start']
            area = assign_dict['area']
            assign_node = assign_dict['assignnode']
            assignstr = assign_dict['assignstr']
            if is_before(start, namenode) and is_wider(area, namenode):
                ok = True
                break
            if not is_before(assign_node, namenode) or len(assignstr) > len(getitem_strs):
                continue
            ok = True
            for i in range(len(assignstr)):
                if not (assignstr[i] == getitem_strs[i] or assignstr[i] == ':'):
                    ok = False
                    break
            if ok:
                break
        if not ok:
            """
            checks if it's a function call
            """
            if isinstance(namenode.parent, ast.Call):
                if isinstance(namenode.parent.parent, ast.Assign):
                    for assign_dict in assign_areas:
                        assign_node = assign_dict['assignnode']
                        if assign_node == namenode.parent.parent:
                            ok = resolve_function_call(rootnode, namenode.parent, namenode, constant_dim)
                            break
        if not ok:
            return '\n' + array_var + '[:] = 0.0'
    
    return ''

def resolve_function_call(rootnode, callnode, namenode, constant_dim):
    """
    if there exist namenode = callnode(namenode)
    resolve if namenode should be filled with 0 by checking inside callnode
    """
    defnodes = py_ast.find_all(rootnode, ast.FunctionDef)
    if isinstance(callnode.func, ast.Name):
        func_name = callnode.func.id
    else:
        return False
    
    defnode_list = [tnode for tnode in defnodes if tnode.name == func_name]
    defnode = defnode_list[-1]
    
    ind = namenode.parent.args.index(namenode)
    
    try:
        array_var_call = defnode.args.args[ind].arg
    except:
        return False
    
    try:
        init_str = preallocate_find_init(defnode, array_var_call, None, constant_dim, rootnode)
        if init_str == '':
            return True
    except:
        pass
    
    return False

def intersect_prealloc_vars(A, B):
    """
    Intersect two dicts that map function name to list of variable names that can be preallocated.
    """
    ans = {}
    for funcname in set(A.keys()) & set(B.keys()):
        ans[funcname] = sorted(set(A[funcname]) & set(B[funcname]))
#    print(A)
#    print(B)
#    print(ans)
#    raise ValueError
    return ans

def preallocate_find_arrays(program_info, s, verbose=None):
    """
    Find arrays that can be pre-allocated by running unit tests.
    
    Given compiler.ProgramInfo instance program_info, and input program string s, returns dict mapping from
    function name string to list of array name strings that can be pre-allocated in the given function.
    """
    if verbose is None:
        verbose = get_verbose()
    extra_verbose = False
    import compiler
    r = py_ast.get_ast(s)
    
    candidates = {}

    # Find array variable candidates that are allocated using preallocate_numpy_funcs (numpy.zeros(), numpy.empty(), etc).
    for func_name in program_info.types:
        array_vars = set()
        for typeconfig in program_info.types[func_name]:
            for (varname, vartype) in typeconfig.items():
                if vartype.is_array():
                    array_vars.add(varname)
        array_alloc_vars = []
        for array_var in sorted(array_vars):
            try:
                preallocate_find_assignnode_py_ast(r, func_name, array_var)
                array_alloc_vars.append(array_var)
            except PreallocateNotFound:
                pass
        candidates[func_name] = array_alloc_vars

    if compiler.use_type_infer and compiler.use_prealloc_alias_analysis and hasattr(program_info, 'type_infer'):
        ans = intersect_prealloc_vars(candidates, program_info.type_infer['prealloc'])
        if verbose:
            util.print_header('Final preallocated arrays:')
            pprint.pprint(ans)
        return ans

    if extra_verbose:
        print('preallocate_find_arrays, candidate array variables:', candidates)

    def test_prealloc(array_vars_map):
        s_prime = preallocate_array_variables(program_info, s, array_vars_map)
        if extra_verbose:
            util.print_header('preallocate_find_arrays, func_name={}, array_vars_map={}, resulting code:'.format(func_name, array_vars_map), s_prime)
        program_info.run_count += 1
        try:
            result = compiler.run_code_subprocess(program_info.path, s_prime, cython=False, verbose=False, clean=True)
            success = True
        except compiler.RunFailed:
            success = False
        return (s_prime, success)

    total_count = 0
    for func_name in sorted(candidates.keys()):
        for array_var in candidates[func_name]:
            total_count += 1

    # Make a modified program that preallocates each candidate using numpy.random.random() and checks the validity of each.
    ans = {}
    current_index = 0
    for func_name in sorted(candidates.keys()):
        for array_var in candidates[func_name]:
            if not program_info.quiet:
                print('Program analysis: checking whether array is suitable for preallocation ({}/{}, variable: {}.{})'.format(current_index, total_count, func_name, array_var))
            (s_prime, success) = test_prealloc({func_name: [array_var]})
            if extra_verbose:
                print('* preallocate_find_arrays: func_name={}, array_var={}, success={}'.format(func_name, array_var, success))
            if success:
                ans.setdefault(func_name, [])
                ans[func_name].append(array_var)
            if verbose:
                print()
            current_index += 1
    if extra_verbose:
        print('* preallocate_find_arrays: ans={}'.format(ans))

    (s_prime, success) = test_prealloc(ans)
    if not success:
        if program_info.preallocate_verbose:
            print('Program analysis: preallocated arrays are: (None)')
        return {}
    else:
        if verbose:
            util.print_header('preallocate_find_arrays, resulting program after preallocations is:', s_prime)
        if program_info.preallocate_verbose:
            print('Program analysis: preallocated arrays are: {}'.format(ans))
        return ans
#        print(func_name, array_alloc_vars)
