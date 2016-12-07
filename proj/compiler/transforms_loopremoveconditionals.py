from transforms_util import *
from transforms_base import BaseTransform

class LoopRemoveConditionals(BaseTransform):
    min_radius = 0
    max_radius = 5
    
    def apply(self, s):
        verbose = get_verbose()
        T0 = time.time()
        
        if verbose:
            util.print_header('Begin LoopRemoveConditionals, input:', s)

        rootnode = py_ast.get_ast(s)
        
        defnode = self.get_previous_node_py_ast(rootnode, ast.FunctionDef)
        local_types = chosen_typespec_loads_py_ast(self.program_info, defnode)
        T1 = time.time()
        
        if local_types is None:         # In the non-type-specialized function, do nothing
            return s
        
        base_fornode = self.get_next_node_py_ast(rootnode, ast.For)
        T2 = time.time()
        
        # Find previous comments that are openmp_add_private_str
        comments_L = []
        all_L = py_ast.get_all_nodes_in_lineno_order(rootnode)
        base_i = all_L.index(base_fornode)
        delete_nodes = []
        for i in range(base_i - 1, -1, -1):
            if isinstance(all_L[i], ast.Str) and py_ast.dump_ast(all_L[i]).startswith(openmp_add_private_str):
                comments_L.append(py_ast.dump_ast(all_L[i]))
                delete_nodes.append(all_L[i])
            else:
                break
        comments_L = comments_L[::-1]
        for delete_node in delete_nodes:
            py_ast.replace_node(rootnode, delete_node, py_ast.get_ast('#deleted_comment').body[0])
        
        if verbose:
            util.print_header('LoopRemoveConditionals comments_L:', '\n'.join(comments_L))
        T3 = time.time()
        
        # Find one or more continuous for loops inside which all other code blocks are nested
        all_fornode_L = py_ast.find_all(base_fornode, ast.For)
        fornode_L = []
        for (i, fornode) in enumerate(all_fornode_L):
            fornode_L.append(fornode)
            subnodes = [node for node in fornode.body if not isinstance(node, ast.Expr)]
            if len(subnodes) == 1 and i + 1 < len(all_fornode_L) and subnodes[0] == all_fornode_L[i + 1]:
                pass
            else:
                break
        T4 = time.time()
        
        # For each such for loop, identify variable name, lower, upper bounds
        assert len(fornode_L)
        if not all(isinstance(x.target, ast.Name) for x in fornode_L):
            raise TransformError('each for loop must have a single var for LoopRemoveConditionals')
        var_L = [x.target.id for x in fornode_L]
        lo_L = []
        hi_L = []
        zero = py_ast.find_all(py_ast.get_ast('0'), ast.Num)[0]
        
        is_parallel_L = []
        T5 = time.time()

        for fornode in fornode_L:
            try:
                fornode_value_str = py_ast.dump_ast(fornode.iter.func)
                try:
                    if fornode.iter.func.value.value.id == 'cython' and fornode.iter.func.value.attr == 'parallel' and fornode.iter.func.attr == 'prange':
                        is_parallel = True
                except:
                    if fornode_value_str in ['range', 'xrange']:
                        is_parallel = False
                    else:
                        raise TransformError('each for loop must have range or xrange for LoopRemoveConditionals ({})', fornode_value_str)
            except:
                raise TransformError('each for loop must have range or xrange for LoopRemoveConditionals ({})', py_ast.dump_ast(fornode.iter))
            is_parallel_L.append(is_parallel)
            
            # Get positional arguments
            call_L = fornode.iter.args
            if len(call_L) == 1:
                lo_L.append(zero)
                hi_L.append(call_L[0])
            elif len(call_L) == 2:
                lo_L.append(call_L[0])
                hi_L.append(call_L[1])
            elif len(call_L) == 3:
                success = False
                if py_ast.is_int_constant_py_ast(py_ast.dump_ast(call_L[2])):
                    val = eval(py_ast.dump_ast(call_L[2]))
                    if val == 1:
                        lo_L.append(call_L[0])
                        hi_L.append(call_L[1])
                        success = True
                if not success:
                    raise TransformError('for loop range currently must have constant step of exactly 1 to apply LoopRemoveConditionals')
            else:
                raise TransformError('for loop range must have 1 to 3 arguments to apply LoopRemoveConditionals')
            
        T6 = time.time()
        
        # Get z3 expression strings
        lo_z3_L = []
        hi_z3_L = []
        
        all_z3_vars = set(var_L)
        for i in range(len(lo_L)):
            (lo_z3, z3_vars) = rewrite_expr_z3_py_ast(lo_L[i])
            all_z3_vars |= z3_vars
            (hi_z3, z3_vars) = rewrite_expr_z3_py_ast(hi_L[i])
            all_z3_vars |= z3_vars
            lo_z3_L.append(lo_z3)
            hi_z3_L.append(hi_z3)
        T7 = time.time()

        if verbose:
            print('LoopRemoveConditionals variable list:', var_L)
            print('LoopRemoveConditionals lower bounds:', lo_L)
            print('LoopRemoveConditionals upper bounds:', hi_L)
            print('LoopRemoveConditionals lower bounds (z3):', lo_z3_L)
            print('LoopRemoveConditionals upper bounds (z3):', hi_z3_L)
            print('LoopRemoveConditionals z3 vars:', all_z3_vars)
        
        # Find all array getitem/setitem expressions (e.g. a[y,x])
        getitem_L = []
        
        if verbose:
            util.print_header('LoopRemoveConditionals, base_fornode:', py_ast.dump_ast(base_fornode))
            
        for node in py_ast.find_all(base_fornode, ast.Subscript):
            if isinstance(node.value, ast.Name):
                getitem_L.append(node)
        if verbose:
            util.print_header('LoopRemoveConditionals, getitem_L:', repr(getitem_L))
            
        T8 = time.time()
        
        # For each dimension j of each expression of an array A, find radius r_j such that we can prove that the expression is always in bounds if we are more than r_j away from edge of array. Associate r_j with corresponding for loop dimension i (i.e. var_L[i]).

        # List of all candidate radii
        radius_L = [str(rval) for rval in range(self.min_radius, self.max_radius+1)]
        radius_set = set(radius_L)
        
        getitem_arrayname_args_L = []
        for getitem_node in getitem_L:
            arrayname = getitem_node.value.id
            if hasattr(getitem_node.slice, 'value'):
                if isinstance(getitem_node.slice.value, ast.Tuple):
                    args = getitem_node.slice.value.elts
                else:
                    args = [getitem_node.slice.value]
            elif hasattr(getitem_node.slice, 'dims'):
                args = getitem_node.slice.dims
            else:
                args = [getitem_node.slice]
            getitem_arrayname_args_L.append((getitem_node, arrayname, args))
            
            for arg in args:
                for current_node in py_ast.get_all_nodes_in_lineno_order(arg):
                    try:
                        current_z3 = rewrite_expr_z3_py_ast(current_node)[0].strip()
                    except:
                        continue
                    if len(current_z3):
                        if current_z3 not in radius_set:
                            radius_set.add(current_z3)
                            radius_L.append(current_z3)
        T9 = time.time()

        if verbose:
            print('LoopRemoveConditionals, radius_L=', radius_L)
            print('LoopRemoveConditionals, getitem_arrayname_args_L=', getitem_arrayname_args_L)
        
        # Intersection of success radii from each dimension
        success_radius_intersect_L = [set(radius_L) for j in range(len(var_L))]
        
        def get_bound(i, is_upper, radius='0'):
            bound = lo_z3_L[i] if not is_upper else hi_z3_L[i]
            ineq = '>=' if not is_upper else '<'
            pad = ('+' if not is_upper else '-') + '(' + radius + ')'
            z3_expr = '(' + var_L[i] + ')' + ineq + '((' + bound + ')' + pad + ')'
            return z3_expr
        
        for (getitem_node, arrayname, args) in getitem_arrayname_args_L:
            try:
                if verbose:
                    print()
                    print('LoopRemoveConditionals, getitem node:', getitem_node)
                
                # Add assumptions that current argument to the array 'arrayname' is out of bounds along dimension j,
                # on either the lower bound side (if is_lo_bound) or else the upper bound side.
                for (j, arg) in enumerate(args):
                    # Skip indices that are greater than the number of loops
                    if j >= len(var_L):
                        continue
                    
                    # Skip getitem index expressions that involve only a single variable, e.g. 'r' and 'c' in A[r,c]
                    if isinstance(arg, ast.Name):
                        continue
                    
                    success_radius_L = []
                    
                    # Find radii that work out of the list of candidate radii, where we can prove in-bounds property on
                    # both sides.
                    for radius in radius_L:
                        radius_success = True
                        radius_is_constant = py_ast.is_int_constant_py_ast(radius)
                        
                        if verbose:
                            print()
                            print('LoopRemoveConditionals, getitem node:', getitem_node, ' radius=', radius)
                            
                        arg_s = py_ast.dump_ast(arg)
                        for is_lo_bound in [False, True]:
                            if ':' in arg_s:
                                continue
                            
                            solver = z3.Solver()
                            
                            current_z3_vars = {}
                            for z3_var in all_z3_vars:
                                current_z3_vars[z3_var] = z3.Int(z3_var)
                                
                            if verbose:
                                print()
                                print('  LoopRemoveConditionals, created z3 solver')
                            
                            # Add assumptions that each for loop variable is in bounds, with a distance 'radius' to the edge.
                            i = j
                            for k in range(2):
                                z3_expr = get_bound(i, k, radius)
                                if verbose:
                                    print('  LoopRemoveConditionals, adding assumption:', z3_expr)
                                try:
                                    z3_expr_eval = eval(z3_expr, globals(), current_z3_vars)
                                except:
                                    radius_success = False
                                    break
                                    #raise TransformError('LoopRemoveConditionals: could not eval expression {}'.format(z3_expr))
                                solver.add(z3_expr_eval)
                            
                            if not radius_success:
                                break
                            # Add dubious assumption that all arrays with the same shapes during unit testing are always equal in shape.
                            # Really we should probably prove this is true by inspecting the numpy code used and its dependencies
                            # instead of just assuming it.
                            if verbose and False:
                                print('Adding assumptions:')
                            arraytype = local_types[arrayname]
                            for (arrayname_p, arraytype_p) in local_types.items():
                                if (arraytype_p.cython_type == arraytype.cython_type and
                                    arraytype_p.shape_list == arraytype.shape_list):
                                    # It is an array of the same size, so assume equal shape
                                    for k in range(len(arraytype_p.shape)):
                                        z3_var_p = rewrite_expr_z3_py_ast(arrayname_p + '.shape[{}]'.format(k), False)[0]
                                        z3_var_current = rewrite_expr_z3_py_ast(arrayname + '.shape[{}]'.format(k), False)[0]
                                        for z3_v in [z3_var_p, z3_var_current]:
                                            if z3_v not in current_z3_vars:
                                                current_z3_vars[z3_v] = z3.Int(z3_v)
                                        if verbose and False:
                                            print('adding assumption {} == {}'.format(z3_var_p, z3_var_current))
                                        solver.add(eval(z3_var_p, globals(), current_z3_vars) ==
                                                   eval(z3_var_current, globals(), current_z3_vars))
                            
                            # Assume that current variable used in getitem is out of bounds (either too low or too high),
                            # and attempt to derive a contradiction.
                            if is_lo_bound:
                                out_of_bounds = arg_s + ' < 0'
                            else:
                                getitem_hi = rewrite_expr_z3_py_ast(arrayname + '.shape[{}]'.format(j), False)[0]
                                out_of_bounds = arg_s + ' >= (' + getitem_hi + ')'
                            if verbose:
                                print('  LoopRemoveConditionals, out of bounds assumption:', out_of_bounds)
                            try:
                                out_of_bounds_eval = eval(out_of_bounds, globals(), current_z3_vars)
                            except:
                                raise LoopRemoveConditionalsProofFailed('LoopRemoveConditionals: could not eval out of bounds expression {}'.format(out_of_bounds))
                            solver.add(out_of_bounds_eval)
                            
                            check = solver.check()
                            if check != z3.unsat:
                                # We did not derive a contraction. This radius is invalid.
                                if verbose:
                                    print('  * No contradiction, radius {} is invalid'.format(radius))
                                radius_success = False
                                break
                        
                        if radius_success:
                            success_radius_L.append(radius)
                            
                    if verbose:
                        print(' ===> LoopRemoveConditionals, getitem_node={}, j={}, arg={}, success_radius_L={}'.format(getitem_node, j, arg, success_radius_L))

                    success_radius_intersect_L[j] &= set(success_radius_L)
                    if verbose:
                        print(' ===> LoopRemoveConditionals, getitem_node={}, j={}, arg={}, success_radius_intersect_L[{}]={}'.format(getitem_node, j, arg, j, success_radius_intersect_L[j]))
            except LoopRemoveConditionalsProofFailed:
                if verbose:
                    print(' ===> LoopRemoveConditionals, proof failed for getitem_node={}'.format(getitem_node))
                pass
        T10 = time.time()

        if verbose:
            for j in range(len(success_radius_intersect_L)):
                print(' ===> LoopRemoveConditionals, success_radius_intersect_L[{}]={}'.format(j, sorted(success_radius_intersect_L[j])))
        
        # Place all constraints in a list
        constraints = []
        for j in range(len(var_L)):
            for k in range(2):
                constraints.append(get_bound(j, k))
        if verbose:
            print('LoopRemoveConditionals, constraints={}'.format(constraints))
        T11 = time.time()

        pre_comments_id = get_unique_id('loop_remove_conditionals')
        pre_comments = ('\n'.join(comments_L) + '\n')
        
        # Reduce success_radius_intersect_L to a single radius along each dimension (chosen_radius_L)
        chosen_radius_L = []
        for j in range(len(var_L)):
            success_L = list(success_radius_intersect_L[j])
            is_constant_L = [py_ast.is_int_constant_py_ast(x) for x in success_L]
            constant_val_L = [(int(success_L[i].strip('()')) if is_constant_L[i] else numpy.nan) for i in range(len(success_L))]
            try:
                min_constant_index = numpy.nanargmin(constant_val_L)
            except ValueError:
                raise TransformError('LoopRemoveConditionals not valid because no radius could be proved')
            success_L = [success_L[min_constant_index]] + [success_L[i] for i in range(len(success_L)) if not is_constant_L[i]]
            if verbose:
                print(' ====> LoopRemoveConditionals, success_L[{}] after filtering: {}'.format(j, success_L))

            try:
                chosen_radius_L.append(z3_util.prove_smallest(success_L, all_z3_vars, constraints))
            except z3_util.ProveSmallestError:
                raise TransformError('LoopRemoveConditionals could not prove smallest element')
            if verbose:
                print(' ====> LoopRemoveConditionals, chosen_radius_L[{}] after prove_smallest: {}'.format(j, chosen_radius_L[j]))
        T12 = time.time()

        T_subpart0 = 0.0
        T_subpart1 = 0.0
        T_subpart2 = 0.0
        T_subpart3 = 0.0
        T_subpart4 = 0.0
        T_subpart5 = 0.0
        T_subpart6 = 0.0
        
        # Split each loop into three sections: [lo, lo+r), [lo+r, hi-r), [hi-r, hi).
        for j in range(len(var_L) - 1, -1, -1):
            radius = chosen_radius_L[j]
            if radius.strip() == '0':
                continue
            
            T_sub0 = time.time()
            fornode = fornode_L[j]
            body = '\n'.join([py_ast.dump_ast(tnode) for tnode in fornode.body])
            bodys = body.split('\n')
            body = '\n    '.join(bodys)
            #body = py_ast.dump_ast(fornode.body)
            T_sub1 = time.time()
            
            lo = lo_L[j]
            hi = hi_L[j]

            lo_star = '(' + '(' + py_ast.dump_ast(lo) + ') + ' + radius + ')'
            hi_star = '(' + '(' + py_ast.dump_ast(hi) + ') - ' + radius + ')'
            lo_lo = py_ast.dump_ast(lo)
            hi_hi = py_ast.dump_ast(hi)
            T_sub2 = time.time()

            var = var_L[j]

            loop_range = 'cython.parallel.prange' if is_parallel_L[j] else 'range'
            loop_options = ', nogil=True' if is_parallel_L[j] else ''
            
            loop1 = """
for {var} in range({lo_lo}, {lo_star}):
    {body}
""".format(**locals())

            loop2 = """
for {var} in {loop_range}({lo_star}, {hi_star}{loop_options}):
    {body}
""".format(**locals())

            loop3 = """
for {var} in range({hi_star}, {hi_hi}):
    {body}
""".format(**locals())

            # In loop2, simplify any conditions used in if (IfNode), elif (ElifNode), '1 if a else 0' (TernaryOperatorNode)
            r_loop2 = py_ast.get_ast(loop2)
            T_sub3 = time.time()
            
            current_z3_vars = {}
            for z3_var in all_z3_vars:
                current_z3_vars[z3_var] = z3.Int(z3_var)
            
            lo_constraint = rewrite_expr_z3_py_ast('{var} >= {lo_star}'.format(**locals()), False)[0]
            hi_constraint = rewrite_expr_z3_py_ast('{var} < {hi_star}'.format(**locals()), False)[0]
            
            T_sub4 = time.time()
            for nodetype in [ast.If, ast.IfExp]:
                for node in py_ast.find_all(r_loop2, nodetype):
                    conditional = node.test
                    
                    for subnode in py_ast.get_all_nodes_in_lineno_order(conditional)[::-1]:
                        for prove_true in [False, True]:
                            solver = z3.Solver()

                            solver.add(eval(lo_constraint, globals(), current_z3_vars))
                            solver.add(eval(hi_constraint, globals(), current_z3_vars))
                            
                            # Attempt to prove a contradiction
                            subnode_s = py_ast.dump_ast(subnode)
                            if prove_true:
                                subnode_s = 'not (' + subnode_s + ')'
                            subnode_s = rewrite_expr_z3_py_ast(subnode_s, False)[0]
                            
                            if verbose:
                                print('LoopRemoveConditionals, prove_true={prove_true}, subnode_s={subnode_s}'.format(**locals()))
                            try:
                                subnode_eval = eval(subnode_s, globals(), current_z3_vars)
                            except:
#                                if verbose:
#                                    traceback.print_exc()
                                continue
                            
                            try:
                                solver.add(subnode_eval)
                            except z3.Z3Exception:
                                continue
                            
                            if solver.check() == z3.unsat:
                                # Always has value of prove_true, so replace with prove_true
                                if verbose:
                                    conditional_s = py_ast.dump_ast(conditional)
                                    print()
                                    print('LoopRemoveConditionals, lo_constraint={lo_constraint}'.format(**locals()))
                                    print('LoopRemoveConditionals, hi_constraint={hi_constraint}'.format(**locals()))
                                    print('LoopRemoveConditionals, subnode_s={subnode_s}'.format(**locals()))
                                    print('LoopRemoveConditionals, replacing {conditional_s} with {prove_true}'.format(**locals()))
                                py_ast.replace_node(r_loop2, subnode, py_ast.get_ast(str(prove_true)).body[0].value)
                                break
            T_sub5 = time.time()

            loop2 = py_ast.dump_ast(r_loop2)
            
            if verbose:
                util.print_header('LoopRemoveConditionals, loop1 (j={}):'.format(j), loop1)
                util.print_header('LoopRemoveConditionals, pre_comments (j={}):'.format(j), pre_comments)
                util.print_header('LoopRemoveConditionals, loop2 (j={}):'.format(j), loop2)
                util.print_header('LoopRemoveConditionals, loop3 (j={}):'.format(j), loop3)
                
            node_loop3 = py_ast.get_ast(loop3).body[0]
            T_sub6 = time.time()
            py_ast.replace_node(rootnode, fornode, node_loop3)
            py_ast.add_before_node(rootnode, node_loop3, py_ast.get_ast(loop1).body[0])
            if j == 0:
                add_comment_node = py_ast.get_ast(pre_comments).body
                if len(add_comment_node):
                    py_ast.add_before_node(rootnode, node_loop3, add_comment_node[0])
            py_ast.add_before_node(rootnode, node_loop3, py_ast.get_ast(loop2).body[0])
            T_sub7 = time.time()
            
            T_subpart0 += T_sub1 - T_sub0
            T_subpart1 += T_sub2 - T_sub1
            T_subpart2 += T_sub3 - T_sub2
            T_subpart3 += T_sub4 - T_sub3
            T_subpart4 += T_sub5 - T_sub4
            T_subpart5 += T_sub6 - T_sub5
            T_subpart6 += T_sub7 - T_sub6
            
        result_s = py_ast.dump_ast(rootnode)
        if verbose:
            util.print_header('LoopRemoveConditionals result:', result_s)
        T13 = time.time()

        if do_profile:
            profile['LoopRemoveConditionals_part0'] += T1-T0
            profile['LoopRemoveConditionals_part1'] += T2-T1
            profile['LoopRemoveConditionals_part2'] += T3-T2
            profile['LoopRemoveConditionals_part3'] += T4-T3
            profile['LoopRemoveConditionals_part4'] += T5-T4
            profile['LoopRemoveConditionals_part5'] += T6-T5
            profile['LoopRemoveConditionals_part6'] += T7-T6
            profile['LoopRemoveConditionals_part7'] += T8-T7
            profile['LoopRemoveConditionals_part8'] += T9-T8
            profile['LoopRemoveConditionals_part9'] += T10-T9
            profile['LoopRemoveConditionals_part10'] += T11-T10
            profile['LoopRemoveConditionals_part11'] += T12-T11
            profile['LoopRemoveConditionals_part12'] += T13-T12

            profile['LoopRemoveConditionals_subpart0'] += T_subpart0
            profile['LoopRemoveConditionals_subpart1'] += T_subpart1
            profile['LoopRemoveConditionals_subpart2'] += T_subpart2
            profile['LoopRemoveConditionals_subpart3'] += T_subpart3
            profile['LoopRemoveConditionals_subpart4'] += T_subpart4
            profile['LoopRemoveConditionals_subpart5'] += T_subpart5
            profile['LoopRemoveConditionals_subpart6'] += T_subpart6

        return result_s
                        
                

    def mutate(self):
        (line, node) = self.get_line_for_mutation(ast.For, outermost=True)
        self.line = line
        self.orig_num = node.orig_lineno