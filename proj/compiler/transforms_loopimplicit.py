from transforms_util import *
from transforms_base import BaseTransform

class LoopImplicit(BaseTransform):
    """
    Loop over implicit vars during Numpy assignment operations.
    """
    
    def get_lhs_name_py_ast(self, assignnode):
        if isinstance(assignnode, ast.Assign):
            no_slice = isinstance(assignnode.targets[0], ast.Name)
        
            if no_slice:
                lhs_name = assignnode.targets[0].id
            else:
                try:
                    lhs_name = assignnode.targets[0].value.id
                except:
                    pass
            
            return lhs_name
        else:
            no_slice = isinstance(assignnode.target, ast.Name)
            if no_slice:
                lhs_name = assignnode.target.id
            else:
                lhs_name = assignnode.target.value.id
            return lhs_name
    
    def __eq__(self, other):
        if isinstance(self, (LoopImplicit, VectorizeInnermost)) and isinstance(other, (LoopImplicit, VectorizeInnermost)):
            return self.line == other.line
        return BaseTransform.__eq__(self, other)
    
    def is_slice_py_ast(self, getitem_tuple, dim):
        if hasattr(getitem_tuple[dim], 'step'):
            return (isinstance(getitem_tuple[dim], ast.Slice) and
                    getitem_tuple[dim].step is None)
        else:
            return isinstance(getitem_tuple[dim], ast.Slice)
    
    def slice_lo_bound_py_ast(self, getitem_tuple, dim):
        if not hasattr(getitem_tuple[dim], 'lower'):
            return '(0)'
        elif getitem_tuple[dim].lower is None:
            return '(0)'
        return '(' + py_ast.dump_ast(getitem_tuple[dim].lower) + ')'
       
    def get_lhs_info_py_ast(self, assignnode, local_types, no_slice):
        full_slice_dims = []
        full_slice_dim_to_index = {}
        full_slice_shapes = []
        
        lhs_name = self.get_lhs_name_py_ast(assignnode)
        try:
            lhs_type = local_types[lhs_name]
        except KeyError:
            raise ArrayParseFailed
        lhs_shape = lhs_type.shape
        lhs_dims = len(lhs_type.shape)
        targets = assignnode.targets if isinstance(assignnode, ast.Assign) else [assignnode.target]
        if no_slice:
            getitem_tuple = []
        else:
            try:
                getitem_tuple = targets[0].slice.value
            except:
                try:
                    getitem_tuple = targets[0].slice.dims
                except:
                    getitem_tuple = targets[0].slice
            if hasattr(getitem_tuple, 'elts'):
                getitem_tuple = getitem_tuple.elts
            if not isinstance(getitem_tuple, list):
                getitem_tuple = [getitem_tuple]
            for dim in range(len(getitem_tuple)):
                if self.is_slice_py_ast(getitem_tuple, dim):
                    full_slice_dim_to_index[dim] = len(full_slice_dims)
                    full_slice_dims.append(dim)
                    if getitem_tuple[dim].lower is not None and getitem_tuple[dim].upper is not None:
                        full_slice_shapes.append('(' + py_ast.dump_ast(getitem_tuple[dim].upper) + ') - (' + py_ast.dump_ast(getitem_tuple[dim].lower) + ')')
                    else:
                        full_slice_shapes.append(lhs_type.shape[dim])
        for dim in range(len(getitem_tuple), lhs_dims):
            full_slice_dim_to_index[dim] = len(full_slice_dims)
            full_slice_dims.append(dim)
            full_slice_shapes.append(lhs_type.shape[dim])
        return (full_slice_dims, full_slice_dim_to_index, lhs_type, lhs_name, lhs_dims, lhs_shape, full_slice_shapes)
    
    def parse_array_slice_py_ast(self, array_node, local_types):
        """
        Parse AST node representing array slice (getitem/setitem) operation.
        
        Return (getitem_tuple, getitem_strs, array_dims) or raise ArrayParseFailed if failed. The array_dims value
        is a constant if RHS is a simple array name, or None if the RHS is a complex expression that is not currently
        analyzed in the type analysis.
        """
        current = array_node
        ok = True
        current.parsed_loopimplicit = True
        
        while current is not None:
            current = current.parent
            if isinstance(current, (ast.Subscript, ast.Name)):
                if hasattr(current, 'parsed_loopimplicit') and current.parsed_loopimplicit:
                    ok = False
                    break
        
        if isinstance(array_node, ast.Name):
            if not array_node.id in local_types or local_types[array_node.id].shape == ():
                ok = False
        
        if not ok:
            raise ArrayParseFailed
        
        array_dims = None
        array_name = None
        if isinstance(array_node, ast.Name):
            array_name = array_node.id
            try:
                array_dims = len(local_types[array_node.id].shape)
            except KeyError:
                pass
        elif isinstance(array_node, ast.Subscript):
            if hasattr(array_node, 'value') and isinstance(array_node.value, ast.Name):
                array_name = array_node.value.id
                try:
                    array_dims = len(local_types[array_name].shape)
                except KeyError:
                    pass
        
        if isinstance(array_node, ast.Subscript):
            if not hasattr(array_node.slice, 'value') and not hasattr(array_node.slice, 'dims') and not isinstance(array_node.slice, ast.Slice):
                raise ArrayParseFailed
            if hasattr(array_node.slice, 'value'):
                getitem_tuple = array_node.slice.value
            elif hasattr(array_node.slice, 'dims'):
                getitem_tuple = array_node.slice.dims
            elif isinstance(array_node.slice, ast.Slice):
                getitem_tuple = array_node.slice
            if hasattr(getitem_tuple, 'elts'):
                getitem_tuple = getitem_tuple.elts
            if isinstance(getitem_tuple, ast.Name) or isinstance(getitem_tuple, ast.Slice):
                getitem_tuple = [getitem_tuple]
            try:
                getitem_strs = [py_ast.dump_ast(x) for x in getitem_tuple]
            except:
                try:
                    getitem_strs = [py_ast.dump_ast(x) for x in getitem_tuple.elts]
                except:
                    raise TransformError('LoopImplicit failed on line {} (annotated_line={})'.format(self.line, getattr(self, 'annotated_line', -1)))
            
        else:
            getitem_tuple = []
            getitem_strs = []
            
        return (getitem_tuple, getitem_strs, array_dims, array_name)
    
    def apply(self, s):
        
        verbose = get_verbose()
        
        if verbose:
            util.print_header('LoopImplicit input:', s)
            
        rootnode = py_ast.get_ast(s)
        py_ast.add_parent_info(rootnode)
        
        try:
            assignnode1 = self.get_next_node_py_ast(rootnode, ast.Assign)
        except:
            assignnode1 = None
        try:
            assignnode2 = self.get_next_node_py_ast(rootnode, ast.AugAssign)
        except:
            assignnode2 = None
        
        if assignnode1 is not None:
            if assignnode2 is not None:
                if assignnode1.lineno > assignnode2.lineno:
                    assignnode = assignnode2
                else:
                    assignnode = assignnode1
            else:
                assignnode = assignnode1
        else:
            if assignnode2 is not None:
                assignnode = assignnode2
            else:
                raise TransformError("Could not get next node")
        
        defnode = self.get_previous_node_py_ast(rootnode, ast.FunctionDef)
        
        if verbose:
            print('defnode:', defnode.name, 'annotated_line:', self.annotated_line)
            if verbose:
                util.print_header('DefNode contents at beginning of LoopImplicit:', defnode.dumps())
                
        local_types = chosen_typespec_loads_py_ast(self.program_info, defnode)
        
        if local_types is None:         # In the non-type-specialized function, do nothing
            return s
        
        initialized_vars = set()        # We promote assignments e.g. 'a = b' so 'a' is initialized at top of function and then we unroll: 'a[i] = b[i]'
        
        initialized_var_strL = []
        
        if isinstance(assignnode, ast.Assign):
            no_slice = len(assignnode.targets) == 1 and isinstance(assignnode.targets[0], ast.Name)
            is_index = len(assignnode.targets) == 1 and isinstance(assignnode.targets[0], ast.Subscript)
        else:
            no_slice = isinstance(assignnode.target, ast.Name)
            is_index = isinstance(assignnode.target, ast.Subscript)
            
        loop_vars = []
        
        if is_index or no_slice:
            
            # Find LHS with array getitem or slice syntax, and determine which dimensions to loop over as well as their shape (or None to use array shape)
            try:
                (full_slice_dims, full_slice_dim_to_index, lhs_type, lhs_name, lhs_dims, lhs_shape, full_slice_shapes) = self.get_lhs_info_py_ast(assignnode, local_types, no_slice)
            except ArrayParseFailed:
                raise TransformError('could not parse LHS')
            
            # Add any needed implicit loop variables
            if len(full_slice_dims):
                # Get a variable on RHS to extract shape from if needed
                rhs_shape_var = None
                for array_node in (list(py_ast.find_all(assignnode, ast.Subscript))[::-1] + 
                                   list(py_ast.find_all(assignnode, ast.Call))[::-1] +
                                   list(py_ast.find_all(assignnode, ast.Name))):
                    if not isinstance(array_node, (ast.Subscript, ast.Name)):
                        continue
                    try:
                        (getitem_tuple, getitem_strs, array_dims, ignore_name) = self.parse_array_slice_py_ast(array_node, local_types)
                    except ArrayParseFailed:
                        continue
                    if isinstance(array_node, ast.Name):
                        rhs_shape_var = array_node.id
                        break
                    elif isinstance(array_node, ast.Subscript) and isinstance(array_node.value, ast.Name):
                        rhs_shape_var = array_node.value.id
                        break
                    elif isinstance(array_node, ast.Call) and isinstance(array_node.func, ast.Name):
                        rhs_shape_var = array_node.func.id
                        break
                    elif isinstance(array_node, ast.Call) and isinstance(array_node.func, ast.Attribute):
                        if isinstance(array_node.func.value, ast.Name):
                            rhs_shape_var = array_node.func.value.id
                            break
                    
                is_rewrite = is_array_rewrite_var_py_ast(defnode, lhs_name)
                
                # Add initialize statement if needed
                if no_slice and isinstance(assignnode, ast.Assign):
                    if lhs_name not in initialized_vars:
                        defnode_dumps = py_ast.dump_ast(defnode)
                        if after_cdef_trailer in defnode_dumps and not is_rewrite:
                            lhs_type_primitive = lhs_type.primitive_type(True, False)
                            if lhs_type.small_constant_shape():
                                lhs_declare = '{lhs_name} = numpy.empty({lhs_type.shape}, numpy.{lhs_type_primitive})'.format(**locals())
                            elif rhs_shape_var is not None:
                                rhs_shape_var_shape = '(' + ','.join(rhs_shape_var + '.shape[{}]'.format(_rhs_j) for _rhs_j in range(lhs_dims)) + ',)'
                                lhs_declare = '{lhs_name} = numpy.empty({rhs_shape_var_shape}, numpy.{lhs_type_primitive})'.format(**locals())
                            else:
                                raise TransformError('no RHS var to extract shape from and shape is not small and constant')
                            
                            declarations = defnode_dumps[:defnode_dumps.index(after_cdef_trailer)]
                            
                            if lhs_declare not in declarations:
                                if verbose:
                                    print('initializing array variable {} (initialized_vars={}, defnode={}, assignnode={})'.format(lhs_name, initialized_vars, defnode.name, py_ast.dump_ast(assignnode)))
                                initialized_vars.add(lhs_name)
                                initialized_var_strL.append(lhs_declare)
                
                # Introduce new loop variables
                new_code = []
                loop_vars = ['_i{}{}'.format(dim, loopimplicit_trailer) for dim in full_slice_dims]
                if verbose:
                    print()
                    print('assignnode:', py_ast.dump_ast(assignnode))
                    print('loop_vars:', loop_vars, 'lhs_dims', lhs_dims, 'full_slice_dims:', full_slice_dims, 'full_slice_shapes:', full_slice_shapes)
                indent = 0
                for (idx, dim) in enumerate(full_slice_dims):
                    varname = loop_vars[idx]
                    if full_slice_shapes[idx] is None or isinstance(full_slice_shapes[idx], str):
                        unknown_shape = True
                    elif isinstance(full_slice_shapes[idx], int) and util.CythonType.dim_has_small_constant_shape(full_slice_shapes[idx]):
                        unknown_shape = False
                    elif isinstance(full_slice_shapes[idx], int) and is_rewrite:
                        unknown_shape = False
                    else:
                        unknown_shape = True
                    if unknown_shape:
                        new_code.append((indent) * ' ' + 'for {varname} in range({lhs_name}.shape[{dim}]):\n'.format(**locals()))
                    else:
                        dim_shape = full_slice_shapes[idx]
                        new_code.append((indent) * ' ' + 'for {varname} in range({dim_shape}):\n'.format(**locals()))
                    indent += 4
                    
                assignnode_new = py_ast.get_ast(py_ast.dump_ast(assignnode)).body[0]
                py_ast.add_parent_info(assignnode_new)
                
                # Remap all array getitem/slice expressions on the assignment line to the same format as the LHS after adding new loop variables.
                # TODO: This is a hack that will fail if we have different kinds of arrays on the RHS: we should really do some kind of data-flow
                # analysis and inspect the types of temporary expressions.
                for array_node in (list(py_ast.find_all(assignnode_new, ast.Subscript))[::-1] +
                                   list(py_ast.find_all(assignnode_new, ast.Call))[::-1] +
                                   list(py_ast.find_all(assignnode_new, ast.Name))):
                    if not isinstance(array_node, (ast.Subscript, ast.Name)):
                        continue
                    
                    if isinstance(array_node, ast.Call):
                        if isinstance(array_node.func, ast.Attribute):
                            if isinstance(array_node.func.value, ast.Name):
                                if array_node.func.value.id in macros.numpy_modules and array_node.func.attr in ['zeros', 'zeros_like', 'empty', 'empty_like']:
                                    py_ast.replace_node(assignnode_new, array_node, py_ast.get_ast('0').body[0].value)
                                    continue
                    
                    if verbose:
                        print('LoopImplicit array_node: {}, {}'.format(py_ast.dump_ast(array_node), type(array_node)))
                    try:
                        (getitem_tuple, getitem_strs, array_dims, array_name) = self.parse_array_slice_py_ast(array_node, local_types)
                    except ArrayParseFailed:
                        continue
                    
                    if verbose:
                        print('=> LoopImplicit array_node: {}, getitem_tuple: {}, getitem_strs: {}, array_dims: {}'.format(py_ast.dump_ast(array_node), getitem_tuple, getitem_strs, array_dims))
                        
                    if array_dims is None:
                        array_dims = len(loop_vars)
                        
                    getitem_new = []
                    last_loop_var_index = -1
                    for dim in range(len(getitem_strs)):
                        if self.is_slice_py_ast(getitem_tuple, dim): #dim in full_slice_dims:
                            last_loop_var_index += 1
                            getitem_new.append('(' + loop_vars[last_loop_var_index] + ') + ' + self.slice_lo_bound_py_ast(getitem_tuple, dim))
                        else:
                            getitem_new.append(getitem_strs[dim])
                    if verbose:
                        print('=> getitem_new (before adding extra loop vars at end):', getitem_new)
                        print('last_loop_var_index:', last_loop_var_index, 'lhs_dims:', lhs_dims)
                    for index in range(last_loop_var_index+1, len(loop_vars)):
                        if len(getitem_new) >= array_dims:
                            break
                        if verbose:
                            print('  loop_vars[{index}]'.format(**locals()))
                        getitem_new.append(loop_vars[index])
                    if verbose:
                        print('=> getitem_new (after adding extra loop vars at end):', getitem_new)
                        
                    getitem_str = '[' + ','.join(getitem_new) + ']'
                    if isinstance(array_node, ast.Name):
                        array_node.id = array_node.id + getitem_str
                    elif isinstance(array_node, ast.Subscript):
                        getitem_to_replace = py_ast.get_ast(array_node.value.id + getitem_str).body[0].value
                        py_ast.replace_node(assignnode_new, array_node, getitem_to_replace)
                        
                    if verbose:
                        print('* LoopImplicit new assignnode: ', py_ast.dump_ast(assignnode_new))
                        
                for namenode in py_ast.find_all(assignnode_new, ast.Name):
                    name = namenode.id
                    if name in macros.macro_to_scalar:
                        py_ast.replace_node(assignnode_new, namenode, py_ast.get_ast(macros.macro_to_scalar[name]).body[0].value)
                
                if verbose:
                    print('* LoopImplicit final assignnode: ', py_ast.dump_ast(assignnode_new))
                    print()
                    
                #new_code.append((assignnode.col_offset + indent) * ' ' + py_ast.dump_ast(assignnode_new))
                new_code.append((indent) * ' ' + py_ast.dump_ast(assignnode_new))
                
                if verbose:
                    util.print_header('new_code:', new_code)
                    util.print_header('DefNode contents before replacing assignment:', py_ast.dump_ast(defnode))
                    
                py_ast.replace_node(rootnode, assignnode, py_ast.get_ast(''.join(new_code)))
                
                if verbose:
                    util.print_header('DefNode contents after replacing assignment:', py_ast.dump_ast(defnode))
            
            # Introduce cdef for new variables.
            if len(loop_vars):
                cdefL = []
                defnode_dumps = py_ast.dump_ast(defnode)
                for varname in loop_vars:
                    decl = '{}(cdef int {})'.format(cython_str, varname)
                    if decl not in defnode_dumps:
                        cdefL.append(decl)
                        
                if verbose:
                    util.print_header('DefNode contents before add_to_top_of_func:', py_ast.dump_ast(defnode))
                if len(cdefL) > 0:
                    for i in range(len(cdefL)):
                        py_ast.add_before_node(rootnode, defnode.body[0], py_ast.get_ast(cdefL[i] + '\n').body[0])
            
        else:
            raise TransformError('Unknown target node type {} {}'.format(type(assignnode), assignnode))
        
        if verbose:
            print('LoopImplicit lhs_name:', lhs_name, 'defnode:', defnode.name, 'local_types:', local_types, 'full_slice_dims:', full_slice_dims, 'full_slice_shapes:', full_slice_shapes)
            
        if len(initialized_var_strL):
            introduce_new_variables_py_ast(defnode, initialized_var_strL)
                
        ans = py_ast.dump_ast(rootnode)
        
        if verbose:
            util.print_header('LoopImplicit result:', ans)

        return ans             

    def can_apply_extract_types_py_ast(self, r, lineno):
        """
        Helper function used by can_apply(), returns (can_be_applied, typesL, assignnode, no_slice).
        """
        try:
            assignnode = self.get_next_node_py_ast(r, (ast.Assign, ast.AugAssign), lineno)
            defnode = self.get_previous_node_py_ast(r, ast.FunctionDef, lineno)
        except TransformError:
            return (False, None, None, None)
        
        if util.is_test_funcname(defnode.name):
            return (False, None, assignnode, None)
        
        try:
            typesL = self.program_info.types[defnode.name]
        except KeyError:
            return (False, None, assignnode, None)
        #assert len(typesL) > 0, 'expected typesL to have non-zero length'
        if len(typesL) == 0:
            print('cannot apply TypeSpecialize to line {} because typesL has zero length'.format(lineno))
            raise ValueError('expected typesL to have non-zero length')
        
        targets = assignnode.targets if isinstance(assignnode, ast.Assign) else [assignnode.target]
        no_slice = len(targets) == 1 and isinstance(targets[0], ast.Name)

        if not ((len(targets) == 1 and isinstance(targets[0], ast.Subscript)) or no_slice):
            return (False, None, assignnode, None)
        
        return (True, typesL, assignnode, no_slice)

    def can_apply(self, r, lineno):
        """
        Determine whether we can apply LoopImplicit to the given line number (return True or False).
        """
        (can_be_applied, typesL, assignnode, no_slice) = self.can_apply_extract_types_py_ast(r, lineno)
        if not can_be_applied:
            return False
        
        any_ok = False
        
        for local_types in typesL:
            try:
                (full_slice_dims, full_slice_dim_to_index, lhs_type, lhs_name, lhs_dims, lhs_shape, full_slice_shapes) = self.get_lhs_info_py_ast(assignnode, local_types, no_slice)
            except ArrayParseFailed:
                continue
            if len(full_slice_dims):
                any_ok = True
#        lhs_name = self.get_lhs_name(assignnode)
#
#        lhs_types = [types[lhs_name] for types in typesL]
#        lhs_has_shape = [len(lhs_type.shape) > 0 for lhs_type in lhs_types]
#
#        ans = any(lhs_has_shape)
        return any_ok

    def mutate(self):
        r = py_ast.get_ast(self.program_info.s_orig)
        py_ast.add_line_info(r)
        
        def is_valid(node):
            return self.can_apply(r, py_ast.get_line(r, node))

        (line, node) = self.get_line_for_mutation((ast.Assign, ast.AugAssign), is_valid)
        self.line = line
        self.orig_num = node.orig_lineno

class VectorizeInnermost(LoopImplicit):
    """
    Vectorize innermost dimension.
    """
    supported_shapes = [2, 3, 4]
    supported_shape_to_hardware = {2: 2, 3: 4, 4: 4}        # Map one of the previous supported shapes to a hardware (power of 2) vectorized size
    
    def __eq__(self, other):
        return LoopImplicit.__eq__(self, other)

    # name, getitem_tuple, local_types[name]
    
    def calc_flat_index_py_ast(self, name, nd_index, array_type):
        """
        Calculates the equivalent 1D index for an ndarray, returning (index_string, last_dim_sliced)

        Args:
            name, string, array name
            nd_index, list-like, list of string representations of the indices into the array
            array_type, CythonType, the CythonType representation of the array
        """
        nd_index = list(nd_index)
        verbose = get_verbose()
        
        dims = array_type.shape
        vec_length = dims[-1]
        
        if verbose:
            print("nd index: %s" % (nd_index, ))
            print("dimensions: %s" % (dims, ))
        if len(nd_index) == len(dims) - 1:
            # Assume last dimension was sliced. Add a slice (':') to the nd_index list
            nd_index.append(py_ast.get_ast('a[:]').body[0].value.slice)
        else:
            assert len(nd_index) == len(dims), "len(nd_index) = %d, len(dims) = %d" % (len(nd_index), len(dims))
        
        index_string = ""
        last_dim_sliced = False
        
        for i in range(len(dims)):
            dims_to_use = dims[i + 1:len(dims)]
            dims_to_use = [dval if util.CythonType.dim_has_small_constant_shape(dval) else None for dval in dims_to_use]
            
            if verbose:
                print("i: %d, dims_to_use: %s" % (i, dims_to_use))
                
            if len(dims_to_use) > 0:
                if any([d is None for d in dims_to_use]):
                    dims_multiply_string = "%s.strides[%d] // sizeof(%s) " % (name, i, array_type.primitive_type())
                else:
                    dims_multiply_string = "(" + str(dims_to_use[0])

                    for j in range(1, len(dims_to_use)):
                        dims_multiply_string += ("*" + str(dims_to_use[j]))

                    # add trailing dimension
                    dims_multiply_string += ")"
                dims_multiply_string += "* (" + py_ast.dump_ast(nd_index[i]) + ")"
                
                index_string += dims_multiply_string
            else:
                if isinstance(nd_index[i], ast.Slice):
                    index_string += "(0)"
                    last_dim_sliced = True
                else:
                    index_string += '(' + py_ast.dump_ast(nd_index[i]) + ')'
                    
            if i < len(dims) - 1:
                index_string += " + "
            if verbose:
                print(index_string)

        index_string = "(" + index_string + ") // %d" % vec_length

        if verbose:
            print("------")
            print(index_string)
            print("------")

        # exit()

        return (index_string, last_dim_sliced)

    def apply(self, s):
        verbose = get_verbose()
        if verbose:
            T0 = time.time()
            util.print_header('VectorizeInnermost input, line={}:'.format(self.annotated_line), s, linenos=True)
        
        # Check if given line is cached. Parse some lines with ast to get greater speed than if we use AST.
        line0 = line = self.annotated_line - 1
        lines = s.split('\n')
        while line < len(lines):
            if verbose:
                print('VectorizeInnermost, line={}, text={}'.format(line, lines[line].strip()))
            try:
                ast_node = ast.parse(lines[line].strip())
            except:
                if verbose:
                    print(' => Failed to parse, continuing to next line')
                line += 1
                continue
            if hasattr(ast_node, 'body') and len(ast_node.body) >= 1 and isinstance(ast_node.body[0], (ast.Assign, ast.AugAssign)):
                if verbose:
                    print(' => Success parse, assignment node, exiting loop')
                break
            if verbose:
                print(' => Not assignment node, going to next line')
            line += 1
        
            #while not isinstance(redbaron.RedBaron(lines[line])[0], redbaron.AssignmentNode) and line < len(lines):
        assignnode_s = lines[line].strip()
        while not lines[line0].lstrip().startswith('def') and line0 > 0:
            line0 -= 1
        defnode_L = lines[line0].strip().split()
        if len(defnode_L) < 2:
            raise TransformError('cannot parse def node in VectorizeInnermost')
        defnode_name = defnode_L[1]
#        assignnode_s = assignnode.dumps().strip()
        cache_key = (assignnode_s + '_' + defnode_name, self.program_info.arraystorage_cache_key())
        if cache_vectorize_innermost:
            if cache_key in transform_cache:
                indent = ' ' * (len(lines[line]) - len(lines[line].lstrip()))
                lines[line] = indent + transform_cache[cache_key]
                cached_ans = '\n'.join(lines)
                if verbose:
                    print('VectorizeInnermost, cached: {} => {}, (took {} secs)'.format(assignnode_s, transform_cache[cache_key], time.time()-T0))
                return cached_ans
        
        s = fix_comment_indentation_no_deletion(s)
        rootnode = py_ast.get_ast(s)
        
        try:
            assignnode1 = self.get_next_node_py_ast(rootnode, ast.Assign)
        except:
            assignnode1 = None
        try:
            assignnode2 = self.get_next_node_py_ast(rootnode, ast.AugAssign)
        except:
            assignnode2 = None
        
        if assignnode1 is not None:
            if assignnode2 is not None:
                if assignnode1.lineno > assignnode2.lineno:
                    assignnode = assignnode2
                else:
                    assignnode = assignnode1
            else:
                assignnode = assignnode1
        else:
            if assignnode2 is not None:
                assignnode = assignnode2
            else:
                raise TransformError("Could not get next node")
        
        defnode = self.get_previous_node_py_ast(rootnode, ast.FunctionDef)
        
        if verbose:
            print('defnode:', defnode.name, 'annotated_line:', self.annotated_line)
            if verbose:
                util.print_header('DefNode contents at beginning of VectorizeInnermost:', py_ast.dump_ast(defnode))
                
        local_types = chosen_typespec_loads_py_ast(self.program_info, defnode)
        
        if local_types is None:         # In the non-type-specialized function, do nothing
            return s
        
        cython_id_to_code_function = {}
        
        if isinstance(assignnode, ast.Assign):
            no_slice = len(assignnode.targets) == 1 and isinstance(assignnode.targets[0], ast.Name)
            is_index = len(assignnode.targets) == 1 and isinstance(assignnode.targets[0], ast.Subscript)
        else:
            no_slice = isinstance(assignnode.target, ast.Name)
            is_index = isinstance(assignnode.target, ast.Subscript)
            
        if is_index or no_slice:
            # Find LHS with array getitem or slice syntax, and determine which dimensions to loop over as well as their shape (or None to use array shape)
            (full_slice_dims, full_slice_dim_to_index, lhs_type, lhs_name, lhs_dims, lhs_shape, full_slice_shapes) = self.get_lhs_info_py_ast(assignnode, local_types, no_slice)
            if verbose:
                print('VectorizeInnermost, lhs_type:', lhs_type)
            
            # Add exactly one vectorize over the last dimension
            if len(full_slice_dims):
                lhs_scalar_type = lhs_type.primitive_type()
                if len(full_slice_dims) != 1:
                    raise TransformError('can only vectorize one dimension')
                if full_slice_dims[0] != lhs_dims-1:
                    raise TransformError('can only vectorize last dimension')
                
                if verbose:
                    util.print_header('before vectorizing, assignment statement is:', assignnode.dumps())
                    
                # Vectorize all slice operations over the last dimension for both LHS and RHS
                assignnode_new = py_ast.get_ast(py_ast.dump_ast(assignnode)).body[0]
                py_ast.add_parent_info(assignnode_new)
                
                """for array_node in (list(py_ast.find_all(assignnode_new, ast.Subscript))[::-1] + 
                                   list(py_ast.find_all(assignnode_new, ast.Call))[::-1] +
                                   list(py_ast.find_all(assignnode_new, ast.Name)) + 
                                   list(py_ast.find_all(assignnode_new, ast.Num))):"""
                for array_node in list(py_ast.find_all(assignnode_new, (ast.Call, ast.Subscript, ast.Name, ast.Num))):
                    
                    skip = False
                    parent = array_node.parent
                    while parent is not None:
                        if hasattr(parent, 'vectorize_visited') and parent.vectorize_visited:
                            skip = True
                            break
                        parent = parent.parent
                    
                    if verbose:
                        print('VectorizeInnermost, skip={}, array_node={}'.format(skip, array_node.dumps()))
                    if skip:
                        continue
                    
                    array_node.vectorize_visited = True
                    
                    is_array = False
                    if isinstance(array_node, ast.Name):
                        name = array_node.id
                        is_array = True
                    elif isinstance(array_node, ast.Subscript):
                        name = py_ast.dump_ast(array_node.value)
                        is_array = True
                    elif isinstance(array_node, ast.Call):
                        name = '_nonvar_' + macros.remove_whitespace(py_ast.dump_ast(array_node))
                    else:
                        name = None
                    
                    if not isinstance(array_node,ast.Call):
                        has_shape = (name in local_types and local_types[name].shape is not None and
                                    len(local_types[name].shape))
                        is_scalar = (name in local_types and local_types[name].shape is not None and
                                    len(local_types[name].shape) == 0)
                    else:
                        has_shape = (name in local_types and local_types[name].shape is not None and
                                    len(local_types[name].shape))
                        if has_shape:
                            is_scalar = (name in local_types and local_types[name].shape is not None and
                                        len(local_types[name].shape) == 0)
                        else:
                            funcname = py_ast.dump_ast(array_node.func)
                            if funcname in scalar_macros:
                                is_scalar = True
                            else:
                                """
                                if function call is not understood or is not scalar, 
                                call loop implicit instead
                                """
                                return super(VectorizeInnermost, self).apply(s)
                        
                    if is_array and is_scalar:
                        is_array = False
                    if name is None:
                        is_scalar = True
                        
                    if is_array:
                        try:
                            (getitem_tuple, getitem_strs, ignore_dims, ignore_name) = self.parse_array_slice_py_ast(array_node, local_types)
                        except ArrayParseFailed:
                            if verbose:
                                print('VectorizeInnermost: skipping', array_node)
                            continue
                        if verbose:
                            print('VectorizeInnermost: getitem_tuple={}'.format(getitem_tuple))
                        if len(getitem_tuple) == 0 and not has_shape:
                            if verbose:
                                print('VectorizeInnermost: getitem_tuple is empty or lacking shape (has_shape={}), skipping'.format(has_shape))
                            continue
                        
                    if verbose:
                        print('VectorizeInnermost: name={}'.format(name))
                        print('VectorizeInnermost: local_types={}'.format(local_types))
                        
                    if is_scalar or has_shape:
                        if not is_scalar:
                            if local_types[name].shape[-1] is None:
                                raise TransformError('Could not vectorize because last dimension is non-constant')
                            
                            last_element = len(local_types[name].shape) - 1
                            (unraveled_array_index, last_dim_sliced) = self.calc_flat_index_py_ast(name, getitem_tuple, local_types[name])
                            
                            vector_dims = local_types[name].shape[last_element]
                            if vector_dims not in self.supported_shapes:
                                raise TransformError("Could not vectorize because array does not have a length of {}".format(self.supported_shapes))
                            if lhs_scalar_type not in ['double', 'float']:
                                raise TransformError('Could not vectorize because LHS was neither double nor float: {}'.format(lhs_scalar_type))
                            hardware_dims = self.supported_shape_to_hardware[vector_dims]
                            vector_type = 'v{}{}'.format(hardware_dims, lhs_scalar_type)
                            scalar_type = lhs_scalar_type #local_types[name].primitive_type()
                        else:
                            last_dim_sliced = False
                            scalar_type = lhs_scalar_type
                            
                        if not is_scalar and last_dim_sliced:
                            # Load vector into SIMD vector
                            data_part = '' if is_array_rewrite_var_py_ast(defnode, name) else '.data'
                            cython_code = pointer_cast + '(' + vector_type + ', ' + name + '{})[{}]'.format(data_part, unraveled_array_index)
                        else:
                            # Load scalar into SIMD vector
                            scalar_arg_str = py_ast.dump_ast(array_node)
                            cython_code = type_cast + '[' + scalar_type + ', ' + scalar_arg_str + ']'   
                    else:
                        raise TransformError("Array could not be vectorized because it has not been type-specialized")
                    
                    py_ast.replace_node(assignnode_new, array_node, py_ast.get_ast(cython_code).body[0].value)
                    
                    if verbose:
                        print('Current assignnode_new: ', py_ast.dump_ast(assignnode_new))
                   
                if verbose:
                    util.print_header('after vectorizing, assignment statement is:', py_ast.dump_ast(assignnode_new))
            
                #replace_str = '{}({})'.format(cython_replace_str, assignnode_new_s)
                
                if verbose:
                    util.print_header('after vectorizing, before assignnode insert, defnode is:', py_ast.dump_ast(defnode))
                
                #re-parse assignnode_new to eliminate parent info
                py_ast.replace_node(defnode, assignnode, py_ast.get_ast(py_ast.dump_ast(assignnode_new)).body[0])
                py_ast.add_line_info(rootnode)
                
                transform_cache[cache_key] = py_ast.dump_ast(assignnode_new)
                
                # Previous vectorization could cause an array variable to be uninitialized, e.g. a = b could be replaced by ((vector *) a.data)[0] = ...
                # Therefore reintroduce array initialization if needed. See test_programs/blur_one_stage_no_parallel.py for a test case.
                argnames = [arg.arg for arg in defnode.args.args if isinstance(arg, ast.arg)]
                if lhs_name not in argnames:
                    is_init = False
                    assignnode_line = py_ast.get_line(defnode, assignnode)
                    if verbose:
                        print()
                    for assignnode_p in py_ast.find_all(defnode, ast.Assign):
                        if len(assignnode_p.targets) == 1 and isinstance(assignnode_p.targets[0], ast.Name) and assignnode_p.targets[0].id == lhs_name and py_ast.get_line(defnode, assignnode_p) < assignnode_line:
                            if verbose:
                                print('lhs_name: {}, init_line: {}, assignnode_new_s: {}'.format(lhs_name, py_ast.dump_ast(assignnode_p), assignnode_new_s))
                            is_init = True
                            break
                    if verbose:
                        print('lhs_name: {}, is_init: {}'.format(lhs_name, is_init))
                        util.print_header('defnode contents:', py_ast.dump_ast(defnode))
                    
                    if not is_init:
                        # Add array initialization
                        if verbose:
                            print('lhs_shape: {}, all_constant: {}'.format(lhs_shape, all(isinstance(subsize, int) for subsize in lhs_shape)))
                        if all(isinstance(subsize, int) for subsize in lhs_shape):
                            lhs_shape_str = str(lhs_shape)
                            lhs_primitive_str = lhs_type.primitive_type(is_numpy = True)
                            new_declare = '{lhs_name} = numpy.empty({lhs_shape_str}, "{lhs_primitive_str}")'.format(**locals())
                            if verbose:
                                print('new_declare: {}'.format(new_declare))
                            introduce_new_variables_py_ast(defnode, [new_declare])
                            
                if verbose:
                    util.print_header('after vectorizing, defnode is:', defnode.dumps())           
        else:
            raise TransformError('Unknown target node type {} {}'.format(type(assignnode), assignnode))
        
        if verbose:
            print('VectorizeInnermost lhs_name:', lhs_name, 'defnode:', defnode.name, 'local_types:', local_types, 'full_slice_dims:', full_slice_dims, 'full_slice_shapes:', full_slice_shapes)
        
        ans = py_ast.dump_ast(rootnode)
        
        if verbose:
            util.print_header('after vectorizing, return value is:', ans)

        return ans             

    def can_apply(self, r, lineno):
        if LoopImplicit.can_apply(self, r, lineno):
            (can_be_applied, typesL, assignnode, no_slice) = LoopImplicit.can_apply_extract_types_py_ast(self, r, lineno)
            if not can_be_applied:
                return False
            
            any_ok = False
            
            for local_types in typesL:
                (full_slice_dims, full_slice_dim_to_index, lhs_type, lhs_name, lhs_dims, lhs_shape, full_slice_shapes) = self.get_lhs_info_py_ast(assignnode, local_types, no_slice)
                if len(lhs_shape) and lhs_shape[-1] in self.supported_shapes and len(full_slice_dims):
                    any_ok = True
                
            return any_ok
        return False

    def mutate(self):
        return LoopImplicit.mutate(self)