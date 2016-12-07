from transforms_util import *
from transforms_base import BaseTransform
from transforms_loopimplicit import LoopImplicit
from transforms_loopimplicit import VectorizeInnermost
from transforms_typespecialize import TypeSpecialize

class Parallel(BaseTransform):
    """
    Loop parallelism transform.
    """
    def __init__(self, program_info, line=None, randomize=False):
        """
        Initialize with line-number 'line' which should precede a for statement.
        
        If randomize is True then instead of parallelizing, runs over the loop in random order.
        """
        BaseTransform.__init__(self, program_info, line)

        self.randomize = randomize
    
    def is_consistent(self):
        """
        Verify that parallel for loops are not nested.
        """
        parallelL = [transform for transform in self.program_info.transformL if isinstance(transform, Parallel)]
        parallel_lines = set([transform.annotated_line + 1 if hasattr(transform, 'annotated_line') else transform.line for transform in parallelL])
        
        rootnode = py_ast.get_ast(self.program_info.s_orig)
        py_ast.add_line_info(rootnode)
        py_ast.add_parent_info(rootnode)
        
        for fornode in py_ast.find_all(rootnode, ast.For):
            line = py_ast.get_line(rootnode, fornode)
            if line in parallel_lines:
                parent = fornode.parent
                while parent is not None:
                    if py_ast.get_line(rootnode, parent) in parallel_lines:
                        return False
                    parent = parent.parent
        
        return True

    def apply(self, s):
        verbose = get_verbose()
        if verbose:
            util.print_header('Parallel input:', s)

        a = py_ast.get_ast(s)
        py_ast.add_parent_info(a)
        fornode = self.get_next_node_py_ast(a, ast.For)
        defnode = self.get_previous_node_py_ast(a, ast.FunctionDef)

        # Only apply if function has been type-specialized
        if defnode.name.endswith(typespecialize_trailer):
            if verbose:
                print('Parallel: defnode ends with typespecialize trailer, proceeding')

            # NOTE: fornode.target (in original RedBaron code) refers to the 
            # thing after the "in" in a for-loop statement. The "target" 
            # attribute also exists in py_ast's For node, but in that case it 
            # refers to the looping variable (e.g. "x" in "for x in range(10)").
            if isinstance(fornode.iter, ast.Call) and \
               fornode.iter.func.id in ['range', 'xrange']:
                if verbose:
                    print('Parallel: fornode target is range, proceeding')

                # collect args to range/xrange call:
                iter_name = fornode.iter.func.id
                iter_args = [py_ast.dump_ast(arg) for arg in fornode.iter.args]

                if self.randomize:
                    iter_name = 'range_shuffled'
                else:
                    iter_name = 'cython.parallel.prange'
                    iter_args.append('nogil=True')

                # now build and replace the function call for the iterator:
                iter_func = "%s(%s)" % (iter_name, ",".join(iter_args))
                iter_ast_node = py_ast.get_ast(iter_func) # create new node
                fornode.iter = iter_ast_node.body[0].value # replace node in ast
        else:
            return s

        if verbose:
            util.print_header('Parallel defnode:', py_ast.dump_ast(defnode))
            
        """
        find preallocate variables inside for loop, if found, turn them back into normal array assignment
        """
        
        if use_py_ast_parallel_preallocate:
            global_nodes = py_ast.find_all(fornode, ast.Global)
            comment_nodes = py_ast.find_all(fornode, ast.Str)
            
            preallocated_vars = []
            preallocated_dimensions = {}
            initialized_vars = []
            
            for global_node in global_nodes:
                if global_node.names[0].startswith(preallocate_prefix):
                    array_var = global_node.names[0].rstrip('_global').lstrip(preallocate_prefix)
                    ind = global_node.parent.body.index(global_node)
                    if_node = global_node.parent.body[ind + 1]
                    assign_node = if_node.body[1]
                    assign_dump = py_Ast.dump_ast(assign_node.value)
                    if assign_dump.endswith("'float32')"):
                        assign_value_new = assign_dump.rstrip("'float32')").rstrip().rstrip(',') + ')'
                    elif assign_dump.endswith('"float32")'):
                        assign_value_new = assign_dump.rstrip('"float32")').rstrip().rstrip(',') + ')'
                    else:
                        assing_value_new = assign_dump
                        
                    assign_value_node = py_ast.get_ast(assign_value_new).body[0].value
                    assert(isinstance(assign_value_node, ast.Call))
                    call_args = py_ast.dump_ast(assign_value_node.args[0])
                    if call_args.startswith('(') or call_args.startswith('['):
                        call_args = call_args[1:-1]
                    dimensions = call_args.split(',')
                    dimensions = [dimension.strip() for dimension in dimensions]
                    if not all([py_ast.is_int_constant_py_ast(dimension) for dimension in dimensions]):
                        continue
                    preallocated_dimensions[array_var] = dimensions
                    #preallocated_dimensions.append(dimensions)
                        
                    if len(global_node.parent.body) > ind + 3:
                        possible_init_node = global_node.parent.body[ind + 3]
                        if py_ast.dump_ast(possible_init_node) == array_var + '[:] = 0':
                            #py_ast.replace_node(fornode, possible_init_node, py_ast.get_ast('pass').body[0])
                            initialized_vars.append(array_var)
                    
                    py_ast.replace_node(fornode, global_node.parent.body[ind + 2], py_ast.get_ast('pass').body[0])
                    py_ast.replace_node(fornode, if_node, py_ast.get_ast('pass').body[0])
                    assign_new = array_var + ' = ' + assign_value_new

                    #py_ast.replace_node(fornode, global_node, py_ast.get_ast('pass').body[0])
                    py_ast.replace_node(fornode, global_node, py_ast.get_ast(assign_new).body[0])
                    preallocated_vars.append(array_var)
                    
            for comment_node in comment_nodes:
                comment = py_ast.dump_ast(comment_node)
                if comment.startswith(cython_preallocate_intermediate):
                    (prefix, varname, hexcode_ordinary, hexcode_float32) = comment.split()[:4]
                    code_block = binascii.unhexlify(hexcode_ordinary.encode('ascii')).decode('ascii')
                    code_blockL = code_block.split('\n')
                    array_var_global = preallocate_prefix + varname + '_global'
                    assign_value_new = None
                    initialized_line = None
                    for line in code_blockL:
                        if line.strip().startswith(array_var_global + ' ='):
                            #print(line.strip())
                            #print(array_var_global)
                            #print(line.strip().lstrip(array_var_global + ' ='))
                            assign_value_new = line.strip().lstrip(array_var_global).strip().lstrip(' = ')
                            #print('--------------------------------------------------------')
                            assign_new = varname + ' = ' + assign_value_new
                            
                            assign_value_node = py_ast.get_ast(assign_value_new).body[0].value
                            assert(isinstance(assign_value_node, ast.Call))
                            call_args = py_ast.dump_ast(assign_value_node.args[0])
                            if call_args.startswith('(') or call_args.startswith('['):
                                call_args = call_args[1:-1]
                            dimensions = call_args.split(',')
                            dimensions = [dimension.strip() for dimension in dimensions]
                            
                            is_constant = True
                            for i in range(len(dimensions)):
                                dimension = dimensions[i]
                                try:
                                    value = str(find_const_value(defnode, dimension, []))
                                    if py_ast.is_int_constant_py_ast(value):
                                        dimensions[i] = value
                                    else:
                                        is_constant = False
                                        break
                                except TransformError:
                                    is_constant = False
                                    break
                            
                            if not is_constant:
                                assign_value_new = None
                                continue
                            #preallocated_dimensions.append(dimensions)
                            preallocated_dimensions[varname] = dimensions
                            
                        if line.strip() == varname + '[:] = 0':
                            if assign_value_new is not None:
                                initialized_line = line.strip()
                                initialized_vars.append(varname)
                                break
                            
                    if assign_value_new is not None:
                        if initialized_line is not None:
                            py_ast.replace_node(fornode, comment_node, py_ast.get_ast(assign_new).body[0])
                            #py_ast.replace_node(fornode, comment_node, py_ast.get_ast(initialized_line).body[0])
                        else:
                            py_ast.replace_node(fornode, comment_node, py_ast.get_ast(assign_new).body[0])
                            #py_ast.replace_node(fornode, comment_node, py_ast.get_ast('pass').body[0])
                        preallocated_vars.append(varname)
            
            """
            for i in range(len(defnode.body)):
                node = defnode.body[i]
                if not isinstance(node, ast.Expr):
                    break
                comment = py_ast.dump_ast(node)
                if comment.startswith(cython_str):
                    varname = comment.split()[-1][:-1]
                    if varname in preallocated_vars:
                        new_cdef_str = cython_str + '(cdef float[' + ']['.join(preallocated_dimensions[varname]) + '] ' + varname + ')'
                        py_ast.replace_node(defnode, node, py_ast.get_ast(new_cdef_str).body[0])"""

        def referred_to_after_loop(assign_node):
            """Returns true if the target (i.e. the "x" in "x = 2") of the
            assignment node is ever referred to after the for-loop. Otherwise, 
            returns false.
            """
            nodes_in_defnode = py_ast.get_all_nodes_in_lineno_order(defnode)

            # first, skip the nodes before the for-loop:
            skip_start_index = 0

            for node in nodes_in_defnode:
                if node.lineno <= fornode.lineno or \
                   node.col_offset > fornode.col_offset:
                    skip_start_index += 1
                else:
                    break

            # for node in nodes_in_defnode:
            #     if node.lineno <= fornode.lineno:
            #         skip_start_index += 1
            #     else:
            #         break

            # # chop off the part of the list we need to skip:
            # nodes_in_defnode = nodes_in_defnode[skip_start_index:]

            # # next, skip the nodes within the for-loop:
            # skip_start_index = 0
            # fornode_col_offset = fornode.col_offset

            # for node in nodes_in_defnode:
            #     if node.col_offset > fornode.col_offset:
            #         skip_start_index += 1
            #     else:
            #         break

            # # chop off the part of the list we need to skip:
            # nodes_in_defnode = nodes_in_defnode[skip_start_index:]

            # NOTE: the following code for getting target_str may need to be 
            # changed if assign_node has multiple targets. e.g.:
            # a = b = c = d
            #target_str = ""

            if isinstance(assign_node, ast.Assign):
                target = assign_node.targets[0]
            elif isinstance(assign_node, ast.AugAssign):
                target = assign_node.target
            else:
                raise ValueError("assign_node is not of type ast.Assign or ast.AugAssign")
            
            if isinstance(target, ast.Subscript):
                try:
                    target_str = target.value.id
                except:
                    target_str = py_ast.dump_ast(target.value)
            else:
                target_str = py_ast.dump_ast(target)

            # now, in the remaining nodes in the function, check if the target
            # of the assignment node ever shows up again:
            for node in nodes_in_defnode[skip_start_index:]:
                node_str = py_ast.dump_ast(node).strip()

                if node_str == target_str:
                    return True
                # if isinstance(node, ast.Assign):
                #     for target in node.targets:
                #         if py_ast.dump_ast(target).strip() == target_str:
                #             return True
                # elif isinstance(node, ast.AugAssign):
                #     if py_ast.dump_ast(target).strip() == target_str:
                #         return True

            return False

        # Rewrite x += y, x *= y, etc within for loop to use instead x = x + y, 
        # x = x * y, etc, so as to avoid triggering Cython's auto-reduction var 
        # inference. As a heuristic, only do this if the variable x is not 
        # referred to after the loop.

        # Likewise, rewrite x = x + y, etc to x += y if the variable x is 
        # referred to after the loop, so Cython applies auto-reduction var
        # inference.

        # collect all of the assignment nodes within the for loop:
        assign_nodes = py_ast.get_all_nodes_in_lineno_order(fornode, 
            (ast.Assign, ast.AugAssign))

        for assign_node in assign_nodes:
            # Rewrite x += y to x = x + y. (ast.AugAssign nodes are specifically
            # nodes that use +=, -=, *=, etc.)
            node_is_referred_to_after_loop = \
                    referred_to_after_loop(assign_node)

            # print("node:", py_ast.dump_ast(assign_node))
            # print("referred to after loop?", node_is_referred_to_after_loop)

            if isinstance(assign_node, ast.AugAssign): 
                node_is_referred_to_after_loop = \
                    referred_to_after_loop(assign_node)

                if verbose:
                    print(
                        'Operator is non-empty, considering rewrite of', 
                        py_ast.dump_ast(assign_node), 
                        node_is_referred_to_after_loop)

                if not node_is_referred_to_after_loop:
                    if verbose:
                        print('Rewriting', py_ast.dump_ast(assign_node))

                    new_assign_node_str = "%s = %s %s %s" % (
                        py_ast.dump_ast(assign_node.target).strip(),
                        py_ast.dump_ast(assign_node.target).strip(),
                        py_ast.get_binary_op_str(assign_node.op),
                        py_ast.dump_ast(assign_node.value).strip())

                    # this should hopefully work, unless you're doing something
                    # weird like nesting assignment statements in other 
                    # assignment statements
                    py_ast.replace_node(
                        a, 
                        assign_node, 
                        py_ast.get_ast(new_assign_node_str).body[0])

            # Similar idea to the previous if: rewrite x = x + y to x += y.
            elif isinstance(assign_node, ast.Assign):
                node_is_referred_to_after_loop = \
                    referred_to_after_loop(assign_node)

                if node_is_referred_to_after_loop and \
                   isinstance(assign_node.value, ast.BinOp):
                    lhs_of_assignment = \
                        py_ast.dump_ast(assign_node.targets[0]).strip()
                    lhs_of_binop = \
                        py_ast.dump_ast(assign_node.value.left).strip()
                    rhs_of_binop = \
                        py_ast.dump_ast(assign_node.value.right).strip()

                    if verbose:
                        print("Found assignment node with binop that is also referred to after the for-loop")
                        print("Node:", py_ast.dump_ast(assign_node).strip())
                        print("Parsed assignment var:", lhs_of_assignment)
                        print("Parsed LHS of binop:", lhs_of_binop)
                        print("Parsed RHS of binop:", rhs_of_binop)

                    # e.g.: x = x + y --> x += y
                    if lhs_of_assignment == lhs_of_binop:
                        new_assign_node_str = "%s %s= %s" % (
                            lhs_of_assignment,
                            py_ast.get_binary_op_str(assign_node.value.op),
                            rhs_of_binop)
                        py_ast.replace_node(
                            a,
                            assign_node,
                            py_ast.get_ast(new_assign_node_str).body[0])

                    # e.g.: x = y + x --> x += y
                    elif lhs_of_assignment == rhs_of_binop:
                        new_assign_node_str = "%s %s= %s" % (
                            lhs_of_assignment,
                            py_ast.get_binary_op_str(assign_node.value.op),
                            lhs_of_binop)
                        py_ast.replace_node(
                            a,
                            assign_node,
                            py_ast.get_ast(new_assign_node_str).body[0])

        # If a variable is an array, has constant shape, and is declared in a 
        # parallel loop, then attempt to rewrite it to be a private variable 
        # with C array type so it can be multi-threaded without using "Python 
        # objects" (GIL limitations). This assumes we have applied Parallel() 
        # and TypeImplicit() previously 
        # (TODO: Apply transforms in a certain order?).

        local_types = chosen_typespec_loads_py_ast(self.program_info, defnode)

        c_rewrite_vars = []
        unique_id_to_str = {}
        assign_nodes = py_ast.get_all_nodes_in_lineno_order(fornode, ast.Assign)

        # for assign_node in assign_nodes:
        #     print("==>", py_ast.dump_ast(assign_node).strip(), assign_node)
        #     for target in assign_node.targets:
        #         print("\t==>", py_ast.dump_ast(target).strip(), target, hasattr(target, "value"))

        # exit()

        # loop over all AST nodes in assignment statements within the for-loop:
        for assign_node in assign_nodes:
            for target in assign_node.targets:
                # print(target, py_ast.dump_ast(target))

                # if the assignment node is something like "array[0] = ...", then 
                # the lhs_name should be "array" (without the brackets)
                try:
                    lhs_array = target.value
                    lhs_name = py_ast.dump_ast(lhs_array).strip()
                    # print("-->", lhs_name)

                # otherwise, the target node is something like "array = ..."
                except:
                    lhs_name = py_ast.dump_ast(target).strip()

                # if we don't know the type, skip that node:
                if not lhs_name in local_types:
                    continue

                # but if we do know the type...
                lhs_type = local_types[lhs_name]
                
                is_small_constant = lhs_type.small_constant_shape()
                if not is_small_constant:
                    try:
                        dimension = preallocated_dimensions[lhs_name]
                    except:
                        continue
                            

                if lhs_type.small_constant_shape():# and is_parallel:
                    if verbose:
                        print('modifying private constant shape array {} in parallel block to C array type'.format(lhs_name))
                        util.print_header('DefNode contents before modifying array type:', py_ast.dump_ast(defnode))

                    initialized_var_strL = []
                    comment_nodes = py_ast.get_comment_strings(defnode)
                    
                    for comment_node in comment_nodes:
                        value_str = py_ast.dump_ast(comment_node).strip()

                        if verbose:
                            print('  found %s, looking for %s' % (value_str, lhs_name))

                        if value_str.startswith('%s(cdef ' % cython_str) and \
                           value_str.endswith(' %s)' % lhs_name):
                            if verbose:
                                print('    doing replacement')

                            replacement_str = '%s(cdef %s %s%s)' % (
                                cython_str,
                                lhs_type.primitive_type(),
                                lhs_name,
                                lhs_type.c_array_type_suffix())

                            py_ast.replace_node(
                                a,
                                comment_node,
                                py_ast.get_ast(replacement_str).body[0].value)

                            c_rewrite_vars.append(lhs_name)

                            openmp_private = '%s %s' % (
                                openmp_add_private_str, 
                                lhs_name)
                            
                            if verbose:
                                util.print_header(
                                    'DefNode contents before fornode change:', 
                                    py_ast.dump_ast(defnode))

                            py_ast.add_before_node(
                                a, 
                                fornode, 
                                py_ast.get_ast(openmp_private).body[0])

                            # print("--> added node", openmp_private)

                            break

                    if verbose:
                        util.print_header(
                            'DefNode contents after modifying array type:', 
                            py_ast.dump_ast(defnode))
                #if can be found as previously preallocated array
                elif isinstance(target, ast.Name):# and is_parallel:
                    if verbose:
                        print('modifying private constant shape array {} in parallel block to C array type'.format(lhs_name))
                        util.print_header('DefNode contents before modifying array type:', py_ast.dump_ast(defnode))

                    if lhs_name in initialized_vars:
                        assign_new = lhs_name + '[:] = 0'
                        py_ast.replace_node(fornode, assign_node, py_ast.get_ast(assign_new).body[0])
                    else:
                        py_ast.replace_node(fornode, assign_node, py_ast.get_ast('pass').body[0])
                    
                    
                    initialized_var_strL = []
                    comment_nodes = py_ast.get_comment_strings(defnode)
                    
                    for comment_node in comment_nodes:
                        value_str = py_ast.dump_ast(comment_node).strip()

                        if verbose:
                            print('  found %s, looking for %s' % (value_str, lhs_name))

                        if value_str.startswith('%s(cdef ' % cython_str) and \
                           value_str.endswith(' %s)' % lhs_name):
                            if verbose:
                                print('    doing replacement')
                                
                            array_size = '[' + ']['.join(dimension) + ']'

                            replacement_str = '%s(cdef %s %s%s)' % (
                                cython_str,
                                lhs_type.primitive_type(),
                                lhs_name,
                                array_size)

                            py_ast.replace_node(
                                a,
                                comment_node,
                                py_ast.get_ast(replacement_str).body[0].value)

                            c_rewrite_vars.append(lhs_name)

                            openmp_private = '%s %s' % (
                                openmp_add_private_str, 
                                lhs_name)
                            
                            if verbose:
                                util.print_header(
                                    'DefNode contents before fornode change:', 
                                    py_ast.dump_ast(defnode))

                            py_ast.add_before_node(
                                a, 
                                fornode, 
                                py_ast.get_ast(openmp_private).body[0])

                            # print("--> added node", openmp_private)

                            break

                    if verbose:
                        util.print_header(
                            'DefNode contents after modifying array type:', 
                            py_ast.dump_ast(defnode))

        #top_of_func_str = '\n'.join(
         #   "%s(%s)" % (cython_c_rewrite, v) for v in c_rewrite_vars)
         
        top_of_func_strs = ["%s(%s)" % (cython_c_rewrite, v) for v in c_rewrite_vars]
        
        for i in range(len(top_of_func_strs)):
            py_ast.add_before_node(a, defnode.body[0], py_ast.get_ast(top_of_func_strs[i]).body[0])

        #if len(top_of_func_str) > 0:
         #   py_ast.add_before_node(
          #      a,
           #     defnode.body[0],
            #    py_ast.get_ast(top_of_func_str).body[0])

        ans = py_ast.dump_ast(a)

        if verbose:
            util.print_header('Parallel output:', ans)

        return ans
        
    def mutate(self):
        rootnode = py_ast.get_ast(self.program_info.s_orig)
        py_ast.add_line_info(rootnode)
        py_ast.add_parent_info(rootnode)
        parentL = []
        if hasattr(self, 'annotated_line'):
            current = self.get_next_node_py_ast(rootnode, ast.For, self.line)
            while current is not None:
                current = current.parent
                if isinstance(current, ast.For):
                    parentL.append(current)
        
        if len(parentL) and random.random() < 0.5:
            node = random.choice(parentL)
            line = py_ast.get_line(rootnode, node)
            self.line = line
            self.orig_num = node.orig_lineno
            success = self.program_info.parallel_cache(self, node)
            if success:
                return

        for i in range(parallel_mutate_tries):
            (line, node) = self.get_line_for_mutation(ast.For)
            self.line = line
            self.orig_num = node.orig_lineno
            success = self.program_info.parallel_cache(self, node)
            if success:
                break
        if not success:
            raise MutateError

    def dependencies(self):
        r = py_ast.get_ast(self.program_info.s_orig)
        py_ast.add_line_info(r)

        fornode = self.get_next_node_py_ast(r, ast.For, self.line)
        defnode = self.get_previous_node_py_ast(r, ast.FunctionDef, self.line)

        loop_implicitL = []
        try:
            loop_implicit = LoopImplicit(self.program_info)
            can_add = True
        except MutateError:
            can_add = False
        if can_add:
            for node in py_ast.find_all(fornode, (ast.Assign, ast.AugAssign)):
                if loop_implicit.can_apply(r, py_ast.get_line(r, node)):
                    loop_implicitL.append(LoopImplicit(self.program_info, py_ast.get_line(r, node)))
    
        try:
            vectorize = VectorizeInnermost(self.program_info)
            can_add = True
        except MutateError:
            can_add = False
        if can_add:
            for node in py_ast.find_all(fornode, (ast.Assign, ast.AugAssign)):
                if vectorize.can_apply(r, py_ast.get_line(r, node)):
                    loop_implicitL.append(VectorizeInnermost(self.program_info, py_ast.get_line(r, node)))
        
        return [TypeSpecialize(self.program_info, py_ast.get_line(r, defnode), None, False),
                TypeSpecialize(self.program_info, py_ast.get_line(r, defnode), None, True)] + loop_implicitL

