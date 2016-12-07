from transforms_util import *
from transforms_base import BaseTransform

class LoopFusion(BaseTransform):
    def __init__(self, program_info, line=None, function_name=None, var_name=None, nstages=1):
        """
        given var_name in function_name, try to fuse its previous stages with nstages times
        """
        BaseTransform.__init__(self, program_info, line)
        self.function_name = function_name
        self.var_name = var_name
        self.nstages = nstages
        self.line = line
        if self.line is None:
            self.mutate()
        self.orig_num = get_orig_line_from_s_orig(program_info.s_orig, line)
    
    def __str__(self):
        args = list(self.args())
        return transform_repr(self, tuple(args))
    
    def args(self):
        return(self.line, self.function_name, self.var_name, self.nstages)
        
    def apply(self, s):
        rootnode = py_ast.get_ast(s)
        caller_fornode = self.get_next_node_py_ast(rootnode, ast.For)
        caller_fornode_str = py_ast.dump_ast(caller_fornode)
        py_ast.add_parent_info(rootnode)
        caller_if_conditions = []
        
        if self.function_name is None or self.var_name is None:
            defnode = self.get_previous_node_py_ast(rootnode, ast.FunctionDef)
            self.function_name = defnode.name
            assignnode = self.get_next_node_py_ast(rootnode, ast.Assign)
            assert(len(assignnode.targets) == 1)
            if isinstance(assignnode.targets[0], ast.Name):
                self.var_name = assignnode.targets[0].id
            else:
                assert(isinstance(assignnode.targets[0], ast.Subscript))
                assert(isinstance(assignnode.targets[0].value, ast.Name))
                self.var_name = assignnode.targets[0].value.id
        else:
            defnodes = py_ast.find_all(rootnode, ast.FunctionDef)
            defnode = [node for node in defnodes if node.name == self.function_name][0]
        
        loop_fusion_info = loop_fusion_program_analysis(defnode)

        var_info = loop_fusion_info[self.function_name][self.var_name]['idx']
        
        for (array_idx, array_tuple, value, iter_lookup, if_condition, op) in var_info:
            max_add_if_flag = -1
            callee_check_vars = []
            assignnode = value.parent
            i_to_replace_info = []
            i_to_replace_node = []
            i_to_namenode = []
            replaced_names = {}
            lo_callers = {}
            hi_callers = {}
            """
            if it's a 1-1 replacement, replace it directly.
            otherwise, store the information needed, and replace them all at the end of the method
            """
            namenodes = py_ast.find_all(value, ast.Name)
            for namenode in namenodes:
                if namenode.id in iter_lookup:
                    continue
                if namenode.id not in loop_fusion_info[self.function_name]:
                    continue
                node_to_replace_info = loop_fusion_info[self.function_name][namenode.id]['idx']
                
                if isinstance(namenode.parent, ast.Subscript):
                    if not len(lo_callers):
                        for caller_key, caller_condition in iter_lookup.items():
                            try:
                                assert(caller_condition.func.id == 'range')
                                if len(caller_condition.args) == 1:
                                    lo_callers[caller_key] = '0'
                                    hi_callers[caller_key] = py_ast.dump_ast(caller_condition.args[0])
                                elif len(caller_condition.args) == 2:
                                    lo_callers[caller_key] = py_ast.dump_ast(caller_condition.args[0])
                                    hi_callers[caller_key] = py_ast.dump_ast(caller_condition.args[1])
                                elif len(caller_condition.args) == 3 and caller_condition.args[2].n == -1:
                                    lo_callers[caller_key] = py_ast.dump_ast(caller_condition.args[1]) + ' + 1'
                                    hi_callers[caller_key] = py_ast.dump_ast(caller_condition.args[0]) + ' + 1'
                                elif len(caller_condition.args) == 3 and caller_condition.args[2].n == 1:
                                    lo_callers[caller_key] = py_ast.dump_ast(caller_condition.args[0])
                                    hi_callers[caller_key] = py_ast.dump_ast(caller_condition.args[1])
                                else:
                                    raise TransformError('cannot process {}'.format(namenode.id))
                                    return s
                            except:
                                raise TransformError('cannot process {}'.format(namenode.id))
                                return s

                #if len(node_to_replace_info) != 1:
                if namenode.id not in replaced_names:
                    """
                    try prove each assignment node are exclusive
                    or, for single callee, simply collect information needed
                    """
                    callee_z3_expr = []
                    dims = []
                    for i in range(len(node_to_replace_info[0][0])):
                        dims.append(fusion_prefix + str(i))
                    for i in range(len(node_to_replace_info)):
                        (replace_idx, replace_tuple, replace_value, replace_iter, replace_if, replace_op) = node_to_replace_info[i]
                        iter_keys = replace_iter.keys()
                        current_z3_vars = {}
                        current_z3_exprs = []
                        bounds = []
                        if len(dims) != len(replace_tuple):
                            raise TransformError('cannot process variable {} because its assignnodes has different dims'.format(namenode.id))
                        
                        if len(replace_if):
                            (z3_expr_str, if_vars) = rewrite_expr_z3_py_ast(replace_if, False)
                            has_if = False
                            if_lookup = {}
                        else:
                            if_vars = None
                        
                        for j in range(len(replace_tuple)):
                            replace_idx_node = replace_tuple[j]
                            """
                            find array.shape and give it a constrain to be larger than constant_shape_max_size
                            """
                            attrnodes = py_ast.find_all(replace_idx_node, ast.Attribute)
                            for attrnode in attrnodes:
                                if attrnode.attr == 'shape' and isinstance(attrnode.parent, ast.Subscript):
                                    shape_str = py_ast.dump_ast(attrnode.parent) + ' > ' + str(util.CythonType.constant_shape_max_size)
                                    (shape_expr, shape_vars) = rewrite_expr_z3_py_ast(shape_str, False)
                                    for var in shape_vars:
                                        if var not in current_z3_vars:
                                            current_z3_vars[var] = z3.Int(var)
                                    current_z3_exprs.append(shape_expr)
                            
                            if(py_ast.dump_ast(replace_idx_node) == ':'):
                                continue
                            (z3_expr_str, z3_vars) = rewrite_expr_z3_py_ast(replace_idx_node)
                            replace_idx_node = py_ast.get_ast(py_ast.dump_ast(replace_idx_node)).body[0].value

                            if if_vars is not None:
                                for var in z3_vars:
                                    if var in if_vars:
                                        has_if = True
                                        if not isinstance(replace_idx_node, ast.Name):
                                            raise TransformError('cannot fuse variable {} because index is too complicated'.format(namenode.id))
                                        if var not in if_lookup:
                                            if_lookup[var] = dims[j]
                            
                            is_iter = False
                            for var in z3_vars:
                                if var in iter_keys:
                                    is_iter = True
                                    
                                    if not isinstance(replace_idx_node, ast.Name):
                                        raise TransformError('cannot fuse variable {} because index is too complicated'.format(namenode.id))
                                        return s
                                current_z3_vars[var] = z3.Int(var)
                            if is_iter:
                                range_arg = replace_iter[replace_idx_node.id]
                                try:
                                    assert(range_arg.func.id == 'range')
                                    if len(range_arg.args) == 1:
                                        lo_bound = '0'
                                        hi_bound = py_ast.dump_ast(range_arg.args[0])
                                    elif len(range_arg.args) == 2:
                                        lo_bound = py_ast.dump_ast(range_arg.args[0])
                                        hi_bound = py_ast.dump_ast(range_arg.args[1])
                                    elif len(range_arg.args) == 3 and range_arg.args[2].n == -1:
                                        lo_bound = py_ast.dump_ast(range_arg.args[1]) + ' + 1'
                                        hi_bound = py_ast.dump_ast(range_arg.args[0]) + ' + 1'
                                    elif len(range_arg.args) == 3 and range_arg.args[2].n == 1:
                                        lo_bound = py_ast.dump_ast(range_arg.args[0])
                                        hi_bound = py_ast.dump_ast(range_arg.args[1])
                                    else:
                                        raise TransformError('cannot process variable {} becaus ewe only accept range() with 1, 2 arguments or 3 arguments with step -1'.format(namenode.id))
                                        return s                               
                                except:
                                    raise TransformError('cannot process variable {} because we only accept range() with 1, 2 arguments or 3 arguments with step -1'.format(namenode.id))
                                    return s
                                lo_str = dims[j] + ' >= ' + lo_bound
                                hi_str = dims[j] + ' < ' + hi_bound
                                (lo_expr, lo_vars) = rewrite_expr_z3_py_ast(lo_str, False)
                                (hi_expr, hi_vars) = rewrite_expr_z3_py_ast(hi_str, False)
                                for var in lo_vars | hi_vars:
                                    if var not in current_z3_vars:
                                        current_z3_vars[var] = z3.Int(var)
                                current_z3_exprs.append(lo_expr)
                                current_z3_exprs.append(hi_expr)
                                bounds.append((lo_bound, hi_bound))
                            else:
                                eq_str = dims[j] + ' == ' + py_ast.dump_ast(replace_idx_node)
                                (eq_expr, eq_vars) = rewrite_expr_z3_py_ast(eq_str, False)
                                for var in eq_vars:
                                    if var not in current_z3_vars:
                                        current_z3_vars[var] = z3.Int(var)
                                current_z3_exprs.append(eq_expr)
                                bounds.append(py_ast.dump_ast(replace_idx_node))

                        for tempList in bounds:
                            for tempElement in tempList:
                                temp_vars = py_ast.find_all(py_ast.get_ast(tempElement), ast.Name)
                                for i in temp_vars:
                                    if (i.id not in callee_check_vars):
                                        callee_check_vars.append(i.id)

                        """
                        replace namenodes in if condition with dims
                        """
                        if len(replace_if):
                            if_node = py_ast.get_ast(replace_if)
                            if_namenodes = py_ast.find_all(if_node, ast.Name)
                            for if_namenode in if_namenodes:
                                if if_namenode.id in if_lookup:
                                    py_ast.replace_node(if_node, if_namenode, py_ast.get_ast(if_lookup[if_namenode.id]).body[0].value)
                            (if_expr, if_vars) = rewrite_expr_z3_py_ast(if_node)
                            for var in if_vars:
                                if var not in current_z3_vars:
                                    current_z3_vars[var] = z3.Int(var)
                            current_z3_exprs.append(if_expr)
                            bounds.append(if_node)
                        callee_z3_expr.append((current_z3_exprs, current_z3_vars, bounds))
                        replaced_names[namenode.id] = callee_z3_expr
                    else:
                        callee_z3_expr = replaced_names[namenode.id]
                        
                    for i in range(len(node_to_replace_info)):
                        for j in range(i, len(node_to_replace_info)):
                            if i == j:
                                continue
                            solver = z3.Solver()
                            (expr1, vars1, bounds1) = callee_z3_expr[i]
                            (expr2, vars2, bounds2) = callee_z3_expr[j]
                            for expr in expr1:
                                if('int' in expr):
                                    expr = expr.replace("int", "")
                                
                                solver.add(eval(expr, globals(), vars1))
                            for expr in expr2:
                                if('int' in expr):
                                    expr = expr.replace("int", "")
                                
                                solver.add(eval(expr, globals(), vars2))
                            if solver.check() != z3.unsat:
                                raise TransformError('cannot process variable {} because assignnodes are not mutually exclusive'.format(namenode.id))
                                return s
                
                is_unique = False
                replace_idx = None
                step_out = False
                (namenode_tuple, namenode_idx) = parse_array_slice_py_ast(namenode.parent)
                dim_replace = {}
                for i in range(len(namenode_tuple)):
                    dim_replace[dims[i]] = namenode_idx[i]
                
                for k in range(len(node_to_replace_info)):
                    (array_idx2, array_tuple2, value2, iter_lookup2, if_condition2, op2) = node_to_replace_info[k]
                    """
                    first checks if there exist a single callee that will satisfy the range condition for caller
                    """
                
                    """
                    if the caller calls the array without indexing
                    """
                    if not isinstance(namenode.parent, ast.Subscript):
                        if array_idx2 == []:
                            py_ast.replace_node(rootnode, namenode, py_ast.get_ast(py_ast.dump_ast(value2)).body[0])
                            is_unique = True
                            replace_idx = k
                            break
                        continue
                
                    """
                    if the caller calls the array with a different index dimension
                    eg: in callee, it's assigned as f[x, y]
                        in caller, it's called as   f[x]
                    """
                    if len(namenode_idx) != len(array_idx2):
                        continue
                
                    """
                    if in the callee, indexing is more than a single namenode
                    eg: a[r + 1, c + 1]
                    ignore the fusing for this case
                    because we might need to handle arithmatics in theorem proving
                    if a[r + 1, c + 1] is under the condition f(r, c) == True
                    then for the caller a[r, c]
                    we need to prove f(r - 1, c - 1) == True
                    """
                
                    """
                    compare each index to see if it can be inlined by callee
                    """
                    ind_replace = {}
                    (expr, vars, bounds) = callee_z3_expr[k]
                    for i in range(len(array_tuple2)):
                        callee_tuple = py_ast.get_ast(py_ast.dump_ast(array_tuple2[i])).body[0].value
                        caller_tuple = py_ast.get_ast(py_ast.dump_ast(namenode_tuple[i])).body[0].value
                        if isinstance(bounds[i], tuple):
                            lo_callee = bounds[i][0]
                            hi_callee = bounds[i][1]
                            checks = ['<', '>']
                        elif isinstance(bounds[i], ast.AST):
                            checks = []
                        else:
                            lo_callee = bounds[i]
                            checks = ['=']
                        if isinstance(callee_tuple, ast.Name) or len(checks) == 1:
                            current_z3_vars = {}
                            if isinstance(callee_tuple, ast.Name):
                                ind_replace[callee_tuple.id] = namenode_idx[i]
                            """
                            build z3 solver on this certain for condition
                            """ 
                            for check in checks:
                                solver = z3.Solver()
                                for caller_key, caller_condition in iter_lookup.items():
                                    lo_caller = lo_callers[caller_key]
                                    hi_caller = hi_callers[caller_key]
                                    lo_caller = py_ast.str_for_loop_fusion(lo_caller)
                                    hi_caller = py_ast.str_for_loop_fusion(hi_caller)
                                          
                                    lo_bound_str = caller_key + ' >= ' + lo_caller
                                    hi_bound_str = caller_key + ' < ' + hi_caller
                                    (lo_str, lo_vars) = rewrite_expr_z3_py_ast(lo_bound_str, False)
                                    (hi_str, hi_vars) = rewrite_expr_z3_py_ast(hi_bound_str, False)
                                    for var in hi_vars:
                                        if var not in current_z3_vars:
                                            current_z3_vars[var] = z3.Int(var)
                                    solver.add(eval(lo_str, globals(), current_z3_vars))
                                    solver.add(eval(hi_str, globals(), current_z3_vars))
                                    
                                    if len(if_condition):
                                        (cond_str, cond_vars) = rewrite_expr_z3_py_ast(if_condition, False)
                                        for var in cond_vars:
                                            if var not in current_z3_vars:
                                                current_z3_vars[var] = z3.Int(var)
                                        solver.add(eval(cond_str, globals(), current_z3_vars))
                                        
                                if check == '<':
                                    caller_bound = py_ast.dump_ast(caller_tuple) + ' < ' + lo_callee
                                elif check == '>':
                                    caller_bound = py_ast.dump_ast(caller_tuple) + ' >= ' + hi_callee
                                else:
                                    caller_bound = py_ast.dump_ast(caller_tuple) + ' != ' + lo_callee
                                (caller_str, caller_vars) = rewrite_expr_z3_py_ast(caller_bound, False)
                                #this part is not right
                                for var in caller_vars:
                                    if var not in current_z3_vars:
                                        current_z3_vars[var] = z3.Int(var)
                                caller_str = py_ast.str_for_loop_fusion(caller_str)
                                
                                solver.add(eval(caller_str, globals(), current_z3_vars))
                                if solver.check() != z3.unsat:
                                    step_out = True
                                    is_unique = False
                                    replace_idx = None
                                    break
                                else:
                                    is_unique = True
                                    replace_idx = k
                            if step_out:
                                step_out = False
                                break   
                        else:
                            if namenode_idx[i] != array_idx2[i]:
                                raise TransformError('cannot process variable {} because index is complicated'.format(namenode.id))
                    
                    if is_unique:
                        solver = z3.Solver()
                        """
                        then also check if_condition
                        """
                        if len(bounds) > len(array_tuple):
                            if_node = bounds[-1]
                            if_namenodes = py_ast.find_all(if_node, ast.Name)
                            for if_namenode in if_namenodes:
                                if if_namenode in dim_replace:
                                    py_ast.replace_node(if_node, if_namenode, py_ast.get_ast(dim_replace[if_namenode.id]).body[0].value)
                            (if_expr, if_vars) = rewrite_expr_z3_py_ast('not( ' + py_ast.dump_ast(if_node) + ')', False)
                            for var in if_vars:
                                if var not in current_z3_vars:
                                    current_z3_vars[var] = z3.Int(var)
                            solver.add(eval(if_expr, globals(), current_z3_vars))
                        
                        for caller_key, caller_condition in iter_lookup.items():
                            lo_caller = lo_callers[caller_key]
                            hi_caller = hi_callers[caller_key]
                            lo_caller = py_ast.str_for_loop_fusion(lo_caller)
                            hi_caller = py_ast.str_for_loop_fusion(hi_caller)
                                          
                            lo_bound_str = caller_key + ' >= ' + lo_caller
                            hi_bound_str = caller_key + ' < ' + hi_caller
                            (lo_str, lo_vars) = rewrite_expr_z3_py_ast(lo_bound_str, False)
                            (hi_str, hi_vars) = rewrite_expr_z3_py_ast(hi_bound_str, False)
                            solver.add(eval(lo_str, globals(), current_z3_vars))
                            solver.add(eval(hi_str, globals(), current_z3_vars))
                        
                        if len(if_condition):
                            (cond_str, cond_vars) = rewrite_expr_z3_py_ast(if_condition, False)
                            solver.add(eval(cond_str, globals(), current_z3_vars))
                                    
                        if solver.check() == z3.unsat:
                            break

                
                """
                finally, if an unique callee can be used to replace this namenode
                """
                if is_unique and isinstance(namenode.parent, ast.Subscript):
                    (array_idx2, array_tuple2, value2, iter_lookup2, if_conditions2, op2) = node_to_replace_info[replace_idx]
                    original_value = py_ast.get_ast(py_ast.dump_ast(value2)).body[0].value
                    possible_replace_namenodes = py_ast.find_all(original_value, ast.Name)
                    for possible_replace_namenode in possible_replace_namenodes:
                        if possible_replace_namenode.id in ind_replace:
                            py_ast.replace_node(original_value, possible_replace_namenode, py_ast.get_ast(ind_replace[possible_replace_namenode.id]).body[0].value)
                    py_ast.replace_node(rootnode, namenode.parent, original_value)
                
                """
                otherwise, try to check if it the caller range condition can be satisfied by the union of all the callees
                """
                if not is_unique:
                    (namenode_tuple, namenode_idx) = parse_array_slice_py_ast(namenode.parent)
                    solver = z3.Solver()
                    """
                    first create a giant z3 expression as the union of the range of all callees
                    """
                    all_callee_condition = []
                    for k in range(len(node_to_replace_info)):
                        (expr, vars, bounds) = callee_z3_expr[k]
                        condition_strs = []
                        for i in range(len(namenode_idx)):
                            caller_tuple = namenode_tuple[i]
                            if(isinstance(caller_tuple, ast.Slice)):
                                continue
                            if isinstance(bounds[i], tuple):
                                condition_strs.append(py_ast.str_for_loop_fusion(py_ast.dump_ast(caller_tuple)) + ' >= ' + py_ast.str_for_loop_fusion(bounds[i][0]))
                                condition_strs.append(py_ast.str_for_loop_fusion(py_ast.dump_ast(caller_tuple)) + ' < ' + py_ast.str_for_loop_fusion(bounds[i][1]))
                            else:
                                condition_strs.append(py_ast.str_for_loop_fusion(py_ast.dump_ast(caller_tuple)) + ' == ' + py_ast.str_for_loop_fusion(bounds[i]))
                        if len(bounds) > len(namenode_tuple):
                            if_node = bounds[-1]
                            if_namenodes = py_ast.find_all(if_node, ast.Name)
                            for if_namenode in if_namenodes:
                                if if_namenode in dim_replace:
                                    py_ast.replace_node(if_node, if_namenode, py_ast.get_ast(dim_replace[if_namenode.id]).body[0].value)
                            if_str = py_ast.str_for_loop_fusion(py_ast.dump_ast(if_node))
                            condition_strs.append(if_str)
                        for i in range(len(condition_strs)):
                            condition_strs[i] = py_ast.z3_div_solver(condition_strs[i]);

                        
                        single_callee_condition = ' and '.join(condition_strs)
                        all_callee_condition.append(single_callee_condition)
                    final_condition = 'not(' + ' or ' .join(all_callee_condition) + ')'
                    (final_str, final_vars) = rewrite_expr_z3_py_ast(final_condition, False)
                    current_z3_vars = {}
                    for var in final_vars:
                        if var not in current_z3_vars:
                            current_z3_vars[var] = z3.Int(var)
                    
                    """
                    then create the z3 expression for caller condition on all dimensions
                    """
                    all_caller_condition = []
                    for caller_key, caller_condition in iter_lookup.items():
                        lo_caller = lo_callers[caller_key]
                        hi_caller = hi_callers[caller_key]
                        lo_caller = py_ast.str_for_loop_fusion(lo_caller)
                        hi_caller = py_ast.str_for_loop_fusion(hi_caller)
                                                
                        all_caller_condition.append(caller_key + ' >= ' + lo_caller)
                        all_caller_condition.append(caller_key + ' < ' + hi_caller)
                    if len(if_condition):
                        all_caller_condition.append(if_condition)
                    
                    caller_condition = ' and '.join(all_caller_condition)
                    (caller_str, caller_vars) = rewrite_expr_z3_py_ast(caller_condition, False)
                    #z3.And((y >= 0), (y < h_j))
                    #z3.Not(z3.Or(z3.And((0 >= 0), (0 < w_j), (y >= 1), (y < h_j)), z3.And((0 >= 0), (0 < w_j), (y == 0))))

                    for var in caller_vars:
                        if var not in current_z3_vars:
                            current_z3_vars[var] = z3.Int(var)
                    # cheat_caller_str = 'z3.And((x >= 1), (x < w_j), (y >= 0), (y < h_j))'
                    # cheat_final_str = 'z3.Not(z3.Or(z3.And(((((x + 1) / 2) - 1) >= 0), ((((x + 1) / 2) - 1) < (w_j / 2)), (y >= 1), (y < h_j)), z3.And(((((x + 1) / 2) - 1) >= 0), ((((x + 1) / 2) - 1) < (w_j / 2)), (y == 0))))'
                    # foo = 1
                    # z3_index_str = ''
                    var_key_list = list(current_z3_vars.keys())
                    # for i in var_key_list:
                    #     z3_index_str = z3_index_str + '(' + i + '>0),'
                    # z3_index_str = z3_index_str[:-1]
                    # z3_index_str = 'z3.And(' + z3_index_str + ')'
                    new_callee_check_vars = []
                    for i in callee_check_vars:
                        if(i in var_key_list):
                            new_callee_check_vars.append(i)


                    solver.add(eval(caller_str, globals(), current_z3_vars))
                    solver.add(eval(final_str, globals(), current_z3_vars))
                    # solver.add(eval(z3_index_str, globals(), current_z3_vars))
                    add_if_flag = -1
                    if solver.check() != z3.unsat:
                        for i in range(5):
                            z3_index_str = ''
                            for j in (new_callee_check_vars):
                                z3_index_str = z3_index_str + '(' + j + '>' + str(i) + '),'
                            z3_index_str = z3_index_str[:-1]
                            z3_index_str = 'z3.And(' + z3_index_str + ')'
                            new_solver = z3.Solver()
                            new_solver.add(eval(caller_str, globals(), current_z3_vars))
                            new_solver.add(eval(final_str, globals(), current_z3_vars))
                            new_solver.add(eval(z3_index_str, globals(), current_z3_vars))
                            if new_solver.check() == z3.unsat:
                                add_if_flag = i
                                max_add_if_flag = max(max_add_if_flag, add_if_flag)
                                for j in (new_callee_check_vars):
                                    caller_if_conditions.append('(' + j + '>' + str(i) + ')')
                                break
                        if add_if_flag == -1:
                            raise TransformError('cannot process variable {} because caller range is not fully covered by callees'.format(namenode.id))
                            return s
                    i_to_replace_node.append(namenode.parent)
                    i_to_replace_info.append(callee_z3_expr)
                    i_to_namenode.append(namenode)
            
            def incr(id_list, current_id = None):
                """
                small counting method to help recording different if condition
                """
                if current_id is None:
                    return [0 for i in id_list]
                current_dim = 0
                while current_dim < len(id_list):
                    if current_id[current_dim] + 1 < len(id_list[current_dim]):
                        current_id[current_dim] += 1
                        return current_id
                    current_id[current_dim] = 0
                    current_dim += 1
                return None
                        
            if len(i_to_namenode):
                """
                if there exist multiple callees that haven't been processed, process them now
                """
                current_id = incr(i_to_replace_info)
                """
                while we haven't enumerated every combination
                """
                caller_expr = []
                current_z3_vars = {}
                for caller_key, caller_condition in iter_lookup.items():
                    lo_caller = lo_callers[caller_key]
                    hi_caller = hi_callers[caller_key]
                    lo_bound_str = caller_key + ' >= ' + lo_caller
                    hi_bound_str = caller_key + ' < ' + hi_caller
                    (lo_str, lo_vars) = rewrite_expr_z3_py_ast(lo_bound_str, False)
                    (hi_str, hi_vars) = rewrite_expr_z3_py_ast(hi_bound_str, False)
                    for var in set(lo_vars) | set(hi_vars):
                        if var not in current_z3_vars:
                            current_z3_vars[var] = z3.Int(var)
                    caller_expr.append(lo_str)
                    caller_expr.append(hi_str)
                if len(if_condition):
                    (cond_str, cond_vars) = rewrite_expr_z3_py_ast(if_condition, False)
                    for var in cond_vars:
                        if var not in current_z3_vars:
                            current_z3_vars[var] = z3.Inf(var)
                    caller_expr.append(cond_str)
                    
                new_str = ''
                    
                while current_id is not None:
                    """
                    first use a z3 solver to see if this combination always ends in False
                    if so, directly ignore this combination
                    """
                    solver = z3.Solver()
                    for expr in caller_expr:
                        solver.add(eval(expr, globals(), current_z3_vars))
                    
                    condition_strs = []
                    for i, ind in enumerate(current_id):
                        namenode = i_to_namenode[i]
                        callee_z3_expr = i_to_replace_info[i]
                        replaced_node = i_to_replace_node[i]
                        (expr, vars, bounds) = callee_z3_expr[ind]
                        (namenode_tuple, namenode_idx) = parse_array_slice_py_ast(namenode.parent)
                        dim_replace = {}
                        for i in range(len(namenode_tuple)):
                            dim_replace[dims[i]] = namenode_idx[i]
                        for k in range(len(namenode_idx)):
                            caller_tuple = namenode_tuple[k]
                            if(isinstance(caller_tuple, ast.Slice)):
                                continue
                            if isinstance(bounds[k], tuple):
                                lo_bound_str = py_ast.dump_ast(caller_tuple) + ' >= ' + bounds[k][0]
                                hi_bound_str = py_ast.dump_ast(caller_tuple) + ' < ' + bounds[k][1]
                                (lo_str, lo_vars) = rewrite_expr_z3_py_ast(lo_bound_str, False)
                                (hi_str, hi_vars) = rewrite_expr_z3_py_ast(hi_bound_str, False)
                                for var in set(lo_vars) | set(hi_vars):
                                    if var not in current_z3_vars:
                                        current_z3_vars[var] = z3.Int(var)
                                lo_str = py_ast.str_for_loop_fusion(lo_str)
                                hi_str = py_ast.str_for_loop_fusion(hi_str)
                                solver.add(eval(lo_str, globals(), current_z3_vars))
                                solver.add(eval(hi_str, globals(), current_z3_vars))
                                condition_strs.append(lo_bound_str)
                                condition_strs.append(hi_bound_str)
                            else:
                                eq_bound_str = py_ast.dump_ast(caller_tuple) + ' == ' + bounds[k]
                                (eq_str, eq_vars) = rewrite_expr_z3_py_ast(eq_bound_str, False)
                                for var in eq_vars:
                                    if var not in current_z3_vars:
                                        current_z3_vars[var] = z3.Int(var)
                                eq_str = py_ast.str_for_loop_fusion(eq_str)
                                solver.add(eval(eq_str, globals(), current_z3_vars))
                                condition_strs.append(eq_bound_str)
                        if len(bounds) > len(namenode_tuple):
                            if_node = bounds[-1]
                            if_namenodes = py_ast.find_all(if_node, ast.Name)
                            for if_namenode in if_namenodes:
                                if if_namenode.id in dim_replace:
                                    py_ast.replace_node(if_node, if_namenode, py_ast.get_ast(dim_replace[if_namenode.id]).body[0].value)
                            (if_expr, if_vars) = rewrite_expr_z3_py_ast(py_ast.dump_ast(if_node), False)
                            for var in if_vars:
                                if var not in current_z3_vars:
                                    current_z3_vars[var] = z3.Int(var)
                            if_expr = py_ast.str_for_loop_fusion(if_expr)
                            solver.add(eval(if_expr, globals(), current_z3_vars))
                            condition_strs.append(py_ast.dump_ast(if_node))
                    if solver.check() == z3.unsat:
                        #ignore this case
                        current_id = incr(i_to_replace_info, current_id)
                        continue
                    """
                    otherwise, create an elif clause
                    """
                    new_condition = ' and '.join(condition_strs)
                    for i,ind in enumerate(current_id):
                        namenode = i_to_namenode[i]
                        callee_z3_expr = i_to_replace_info[i]
                        replaced_node = i_to_replace_node[i]
                        (expr, vars, bounds) = callee_z3_expr[ind]
                        (namenode_tuple, namenode_idx) = parse_array_slice_py_ast(namenode.parent)
                        node_to_replace_info = loop_fusion_info[self.function_name][namenode.id]['idx']
                        (array_idx2, array_tuple2, value2, iter_lookup2, if_conditions2, op2) = node_to_replace_info[ind]
                        original_value = py_ast.get_ast(py_ast.dump_ast(value2)).body[0].value
                        for k in range(len(namenode_idx)):
                            callee_tuple = array_tuple2[k]
                            possible_replace_namenodes = py_ast.find_all(original_value, ast.Name)
                            for possible_replace_namenode in possible_replace_namenodes:
                                if isinstance(callee_tuple, ast.Name) and possible_replace_namenode.id == callee_tuple.id:
                                    py_ast.replace_node(original_value, possible_replace_namenode, py_ast.get_ast(namenode_idx[k]).body[0].value)
                        py_ast.replace_node(assignnode, replaced_node, original_value)
                        i_to_replace_node[i] = original_value
                    new_body = py_ast.dump_ast(assignnode)
                    new_str += """
elif {new_condition}:
    {new_body}
""".format(**locals())
                    current_id = incr(i_to_replace_info, current_id)
                new_str = new_str[3:]
                py_ast.replace_node(rootnode, assignnode, py_ast.get_ast(new_str).body[0])

            if max_add_if_flag >= 0:
                condition_for_str = ""
                fused_loop = py_ast.dump_ast(caller_fornode)
                orig_loop = caller_fornode_str
                for con in caller_if_conditions:
                    condition_for_str += (con + "and")
                condition_for_str += "(0==0)"
                new_for_str = """
if {condition_for_str}:
    {fused_loop}
else:
    {orig_loop}
""".format(**locals())
                py_ast.replace_node(rootnode, fornode, py_ast.get_ast(new_for_str).body[0])
                y_ast.add_parent_info(rootnode)
                            
            """
            check if the callee is not used elsewhere, if so, delete it
            """
            py_ast.add_parent_info(rootnode)
            callee_nodes = py_ast.find_all(defnode, ast.Name)
            callee_namenodes = [node for node in callee_nodes if node.id in replaced_names]
                
            not_used = True
            for callee_namenode in callee_namenodes:
                if isinstance(callee_namenode.parent, ast.Assign):
                    if callee_namenode in callee_namenode.parent.targets:
                        continue
                if isinstance(callee_namenode.parent, ast.Subscript):
                    if isinstance(callee_namenode.parent.parent, ast.Assign):
                        if callee_namenode.parent in callee_namenode.parent.parent.targets:
                            continue
                not_used = False
                break
                
            if not_used:
                for callee_namenode in callee_namenodes:
                    parent = callee_namenode.parent
                    while not isinstance(parent, ast.Assign):
                        parent = parent.parent
                    parent_of_assign = parent.parent
                    py_ast.replace_node(defnode, parent, py_ast.get_ast('pass').body[0])
                    parent = parent_of_assign
                    while isinstance(parent, ast.For) or isinstance(parent, ast.If):
                        if isinstance(parent, ast.For):
                            if all([isinstance(node, ast.Pass) for node in parent.body]):
                                parent_of_for = parent.parent
                                py_ast.replace_node(defnode, parent, py_ast.get_ast('pass').body[0])
                                parent = parent_of_for
                            else:
                                break
                        else:
                            if all([isinstance(node, ast.Pass) for node in parent.body]) and all([isinstance(node, ast.Pass) for node in parent.orelse]):
                                parent_of_for = parent.parent
                                py_ast.replace_node(defnode, parent, py_ast.get_ast('pass').body[0])
                                parent = parent_of_for
                            else:
                                break
                
        return py_ast.dump_ast(rootnode)
    
    
    def mutate(self):
        """
        to be implemented
        """
        cache_key = self.program_info.s_orig
        if cache_key in fusion_info_cache:
            (rootnode, loop_fusion_info) = fusion_info_cache[cache_key]
        else:
            rootnode = py_ast.get_ast(self.program_info.s_orig)
            py_ast.add_parent_info(rootnode)
            loop_fusion_info = loop_fusion_program_analysis(rootnode)
            fusion_info_cache[cache_key] = (rootnode, loop_fusion_info)
        
        """
        first randomsly choose function_name
        """
        while len(loop_fusion_info):
            func_name = random.choice(list(loop_fusion_info.keys()))
            while not len(loop_fusion_info[func_name]):
                del loop_fusion_info[func_name]
                func_name = random.choice(list(loop_fusion_info.keys()))
        
            """
            then randomly choose var_name
            """
            var_name = random.choice(list(loop_fusion_info[func_name].keys()))
            """
            checkes whether var_name is fusable
            """
            while not len(loop_fusion_info[func_name]):
                idxes = loop_fusion_info[func_name][var_name]['idx']
                seen_names = set()
                for item in idxes:
                    namenodes = py_ast.find_all(item[2], ast.Name)
                    names = [node.id for node in namenodes]
                    seen_names |= set(names)
                if not len(seen_names & set(loop_fusion_info[func_name].keys())):
                    del loop_fusion_info[func_name][var_name]
                    var_name = random.choice(list(loop_fusion_info[func_name].keys()))
                else:
                    break
            if len(loop_fusion_info[func_name]):
                break
            
        if len(loop_fusion_info):
            self.funcion_name = func_name
            self.var_name = var_name
            self.line = py_ast.get_line(rootnode, loop_fusion_info[self.funcion_name][self.var_name]['idx'][0][2])
        else:
            raise TransformError('no available variable to fuse')