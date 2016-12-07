from transforms_util import *
from transforms_base import BaseTransform

class TypeSpecialize(BaseTransform):
    """
    Type specialization code transform: adds cdef statements.
    """

    def __init__(self, program_info, line=None, typesL=None, checks=None):
        """
        Initialize with line-number 'line' which should precede a def statement, and 'types' which is a list of all type configurations used for that function. Each type configuration is a dict mapping variable name strings to their Cython type strings (e.g. {'a': 'int', 'x': 'double'}).
        
        If typesL is None then it is read from the type information in program_info.
        
        If checks is False then disables some checks for the function (bounds checking, index wrap-around, divide by zero).
        """
        """Constructor (py_ast version) for TypeSpecialize. This should be 
        applied before a def statement.

        Args:
            program_info, compiler.ProgramInfo, compilation context
            line, int, line of the program for this transform to be applied to
            typesL, list of ???, ???
            checks, boolean, ???
        """

        """
        print('**********************')
        print(program_info.s_orig)
        print(line)
        print(typesL)
        print(checks)
        print('**********************')
        """

        BaseTransform.__init__(self, program_info, line)

        if checks is None:
            checks = random.choice([False, True])

        self.typesL = typesL
        self.checks = checks

        if self.line is None:
            self.mutate()

        if self.typesL is None:
            source_str = program_info.s_orig

            if program_info.s_current:
                source_str = program_info.s_current

            a = py_ast.get_ast(source_str)
            defnode = [node for node in py_ast.find_all(a, ast.FunctionDef)
                       if node.lineno >= self.line][0]
            funcname = defnode.name

            try:
                self.typesL = program_info.types[funcname]
            except KeyError:
                raise TransformError(
                    '%s not found in types, self.line=%d, line=%d' % (
                        funcname, self.line, line))

        assert self.line is not None
        assert self.typesL is not None

        self.typesL = list(self.typesL)

        for typeconfig in self.typesL:
            for (varname, vartype) in typeconfig.items():
                assert isinstance(varname, str), \
                    'expected varname to be str: %s' % varname

                if not isinstance(vartype, util.CythonType):
                    typeconfig[varname] = util.CythonType.from_cython_type(
                        vartype, program_info)
                    vartype = typeconfig[varname]

                assert isinstance(vartype, util.CythonType), \
                    'expected vartype to be CythonType: %s' % str(vartype)

    def __str__(self):
        args = list(self.args())
        args[1] = None
        return transform_repr(self, tuple(args))
    
    def args(self):
        return (self.line, self.typesL, self.checks)

    def no_check_lines(self):
        lines = ['@cython.cdivision(True)', '@cython.boundscheck(False)']
        if not self.program_info.safe:
            lines.append('@cython.wraparound(False)')
        return lines

    def apply(self, s):
        # verbose = True
        verbose = get_verbose()

        if verbose:
            util.print_header('TypeSpecialize input:', s)
            util.print_header('TypeSpecialize typesL:', self.typesL)

        specialized_nameL = []
        arg_names_and_typesL = []

        # for (i, types) in enumerate(self.typesL):
        for types in self.typesL:
            a = py_ast.get_ast(s)

            if verbose:
                util.print_header("TypeSpecialize types:", types)
                util.print_header("ast initial:", py_ast.dump_ast(a))
                print("self.line: ", self.line)

            defnode = self.get_next_node_py_ast(a, ast.FunctionDef)
            #defnode = [node for node in \
             #          py_ast.get_all_nodes_in_lineno_order(a, ast.FunctionDef)
              #         if typespecialize_header not in node.name and \
               #           ((len(node.args.args) > 0 and \
                #            node.args.args[0].arg in types) or 
                 #          (len(node.args.args) == 0))][0]
            defnode_str = py_ast.dump_ast(defnode)

            specialized_defnode = py_ast.get_ast(defnode_str).body[0]

            # I'm not sure if this works with type-hints in the args...
            specialized_args = [py_ast.dump_ast(arg) for arg in \
                                specialized_defnode.args.args]

            if verbose:
                util.print_header("selected defnode:", 
                    py_ast.dump_ast(defnode))
                print("specialized_args (%d):" % len(specialized_args),
                    specialized_args)
                print("types:", types)
                for arg in specialized_args:
                    print(arg, '-->', types[arg].cython_nickname)


            specialized_defnode_name_str = "%s%s%s%s" % (
                specialized_defnode.name,
                typespecialize_header,
                ''.join(('_' + types[arg].cython_nickname) for arg in \
                        specialized_args),
                typespecialize_trailer)
            specialized_defnode.name = specialized_defnode_name_str
            specialized_nameL.append(specialized_defnode_name_str)

            if verbose:
                print("Specialized name:", specialized_defnode_name_str)
                util.print_header('Specialized defnode:',
                    py_ast.dump_ast(specialized_defnode))
                print("lineno:", specialized_defnode.lineno)

            py_ast.add_after_node(a, defnode, specialized_defnode)

            if verbose:
                util.print_header('Program after adding specialized defnode:',
                    py_ast.dump_ast(a))

            # reparse the tree to update the lineno of specialized_defnode, then
            # search through the AST to find specialized_defnode, to update its
            # reference
            a = py_ast.get_ast(py_ast.dump_ast(a))
            specialized_defnode = [
                n for n in py_ast.find_all(a, ast.FunctionDef) \
                if n.name == specialized_defnode_name_str
            ][0]

            # We need to process the function definition and it's arguments.
            # Fortunately, even if the args are on multiple lines, py_ast's 
            # dump_ast() function will put them all on one line.
            func_def_str = \
                py_ast.dump_ast(specialized_defnode).strip().split("\n")[0]

            if verbose:
                print("Specialized defnode definition:", func_def_str)

            # change def to cdef
            if func_def_str.startswith('def'):
                func_def_str = 'c' + func_def_str

            # remove spacing between arguments:
            while '  ' in func_def_str:
                func_def_str = func_def_str.replace('  ', ' ')

            while ' ,' in func_def_str:
                func_def_str = func_def_str.replace(' ,', ',')

            while ', ' in func_def_str:
                func_def_str = func_def_str.replace(', ', ',')

            while '= ' in func_def_str:
                func_def_str = func_def_str.replace('= ', '=')

            while ' =' in func_def_str:
                func_def_str = func_def_str.replace(' =', '=')

            # try to add return type:
            return_key = util.types_non_variable_prefix + 'return'

            if return_key in types and func_def_str.startswith('cdef'):
                func_def_str = "cdef %s %s" % (
                    types[return_key].cython_type[0].cython_type_str(),
                    func_def_str[4:]) # include string after 'cdef'

            if verbose:
                print('Specialized defnode definition, after replacements:',
                    func_def_str)

            cdefL = []
            argdefL = []
            arg_names_and_types = []
            varL = []
            var_set = set()

            for argnode in specialized_defnode.args.args:
                varname = argnode.arg
                varL.append(varname)
                var_set.add(varname)

            for varname in types:
                if not util.is_type_variable(varname):
                    continue

                if varname not in var_set:
                    varL.append(varname)
                    var_set.add(varname)

            for varname in varL:
                vartype = types[varname]

                if varname in specialized_args:
                    arg_names_and_types.append((varname, vartype))

                    for prefix in [',', '(']:
                        if (prefix + varname + ',') in func_def_str:
                            match = "%s%s," % (prefix, varname)
                            replace = "%s%s %s," % (
                                prefix, 
                                vartype.cython_type_str(), 
                                varname)
                            func_def_str = func_def_str.replace(match, replace)

                        elif (prefix + varname + ')') in func_def_str:
                            match = "%s%s)" % (prefix, varname)
                            replace = "%s%s %s)" % (
                                prefix, 
                                vartype.cython_type_str(),
                                varname)
                            func_def_str = func_def_str.replace(match, replace)

                        elif (prefix + varname + '=') in func_def_str:
                            match = "%s%s=" % (prefix, varname)
                            replace = "%s%s %s=" % (
                                prefix,
                                vartype.cython_type_str(),
                                varname)
                            func_def_str = func_def_str.replace(match, replace)

                        if verbose:
                            print(' => TypeSpecialize, func_def_str after variable {}, prefix={}: {}'.format(varname, prefix, func_def_str))
                else:
                    cdefL.append('%s(cdef %s %s)' % (cython_str, 
                        vartype.cython_type_str(), varname))

            arg_names_and_typesL.append(arg_names_and_types)

            # Find dictionaries on RHS of assignment, and replace with 
            # CythonType dict class constructor
            for assignnode in py_ast.find_all(specialized_defnode, ast.Assign):
                lhs = py_ast.dump_ast(assignnode.targets[0]).strip()

                if lhs in types:
                    lhs_type = types[lhs]

                    if lhs_type.is_dict():
                        rhs = assignnode.value

                        if isinstance(rhs, ast.Dict):
                            rhs_L = zip(rhs.keys, rhs.values)
                            success = True
                            rhs_d = {}

                            for (key, value) in rhs_L:
                                if isinstance(key, ast.Str):
                                    key_str = py_ast.dump_ast(key).strip()
                                    rhs_key = eval(key_str)
                                    rhs_d[rhs_key] = value
                                else:
                                    success = False
                                    break

                            if success:
                                lhs_type_str = lhs_type.cython_type_str()
                                lhs_key_list = lhs_type.sorted_key_list()

                                if set(lhs_key_list) != set(rhs_d.keys()):
                                    warnings.warn('key list mismatch, not using dict cdef class constructor: {}, {}'.format(lhs_key_list, rhs_d.keys()))
                                else:
                                    replace_str = "%s(%s)" % (
                                        lhs_type_str,
                                        ','.join(rhs_d[key].dumps() for key in lhs_key_list))
                                    if verbose:
                                        print("replace_str:", replace_str)
                                    assignnode.value.replace(replace_str)

            if verbose:
                print('cdefL:')
                print(cdefL)
                util.print_header('AST before replacing node.value:',
                    py_ast.dump_ast(a))

            chosen_typespec_dumps_str = chosen_typespec_dumps(types)

            if verbose:
                print("chosen_typespec_dumps_str:", chosen_typespec_dumps_str)

            type_spec_comment_str = "%s%s\n%s" % (
                chosen_typespec_dumps_str,
                '\n'.join(cdefL), 
                after_cdef_trailer)
            type_spec_comment_nodes = py_ast.get_ast(type_spec_comment_str).body

            for node in reversed(type_spec_comment_nodes):
                specialized_defnode.body.insert(0, node)

            # Replace array shapes with constants: replace a.shape[m] for constant m with constant shape if known
            for sub_node in py_ast.find_all(specialized_defnode, ast.Subscript):
                if isinstance(sub_node.value, ast.Attribute) and isinstance(sub_node.value.value, ast.Name) and sub_node.value.attr == 'shape' and isinstance(sub_node.slice, ast.Index) and isinstance(sub_node.slice.value, ast.Num):
                    sub_var = sub_node.value.value.id
                    sub_dim = sub_node.slice.value.n
                    if sub_var in types:
                        sub_type = types[sub_var]
#                        print('sub_type.shape:', sub_type.shape, type(sub_type.shape))
#                        print('sub_dim:', sub_dim, type(sub_dim))
                        if sub_type.is_array() and sub_type.shape is not None and 0 <= sub_dim < len(sub_type.shape) and util.CythonType.dim_has_small_constant_shape(sub_type.shape[sub_dim]):
                            sub_shape = sub_type.shape[sub_dim]
                            py_ast.replace_node(a, sub_node, ast.parse(str(sub_shape)).body[0].value)

            if verbose:
                util.print_header('AST after replacing node.value:', 
                    py_ast.dump_ast(a))

            if not self.checks:
                for line in self.no_check_lines():

                    comment_str = "%s(%s)" % (cython_str, line)
                    comment_str_node = py_ast.get_ast(comment_str).body[0]
                    py_ast.add_before_node(a, specialized_defnode, 
                        comment_str_node)

            if verbose:
                print(' => TypeSpecialize, func_def_str final: {}'.format(func_def_str))
                print()

            comment_str = "%s(%s)" % (cython_replace_str, func_def_str)
            comment_str_node = py_ast.get_ast(comment_str).body[0]
            py_ast.add_before_node(a, specialized_defnode, comment_str_node)
            
            s = py_ast.dump_ast(a)

            if verbose:
                util.print_header('TypeSpecialize partial result:', s)

        # Add type-specialized dispatch to original function
        a = py_ast.get_ast(s)
        stop_index = specialized_defnode_name_str.index(typespecialize_header)
        original_func_name = specialized_defnode_name_str[:stop_index]
        defnode = [node for node in py_ast.find_all(a, ast.FunctionDef) 
                   if node.name == original_func_name][0]


        if verbose:
            util.print_header("TypeSpecialize partial result dumped:",
                py_ast.dump_ast(a))
            util.print_header('TypeSpecialize defnode found:', 
                py_ast.dump_ast(defnode))
            print(defnode.lineno)

        if_switch = []

        for i in range(len(specialized_nameL)):
            andL = []

            for (argname, argtype) in arg_names_and_typesL[i]:
                andL.append(argtype.isinstance_check(argname))

            if len(andL) == 0:
                andL = ['True']     # No argument functions always succeed in type specializing

            if_statement = 'if' if i == 0 else 'elif'
            func_args = ', '.join([argname for (argname, argtype) in arg_names_and_typesL[i]])
            if_switch.append('{} {}:\n    return {}({})'.format(if_statement, ' and '.join(andL), specialized_nameL[i], func_args))

        str_to_add = chosen_typespec_dumps(None) + '\n'.join(if_switch)
        nodes_to_add = py_ast.get_ast(str_to_add).body

        if verbose:
            util.print_header('TypeSpecialize result before add_to_top_of_func:', s)
            util.print_header("if_switch:", if_switch)
            util.print_header("str_to_add", str_to_add)
            print("nodes to add:", nodes_to_add)

        for node in reversed(nodes_to_add):
            defnode.body.insert(0, node)

        s = py_ast.dump_ast(a)

        if verbose:
            util.print_header('TypeSpecialize final result:', s)

        return s

    def mutate(self, set_all=False, checks=False):
        """
        If set_all is True, then creates TypeSpecialize instances for all functions, with checks set to the argument checks.
        """
        def is_valid(node):
            return node.name in self.program_info.types

        if self.line is not None:
            before_line = self.line
            before_orig_line = self.orig_num
            before_checks = self.checks
            before_typesL = self.typesL
        else:
            before_line = None
            before_orig_line = None
            before_checks = False
            before_typesL = None
        
        if before_line is None:
            (line, node) = self.get_line_for_mutation(ast.FunctionDef, is_valid)
            self.line = line
            self.orig_num = node.orig_lineno
            self.typesL = list(self.program_info.types[node.name])
            self.checks = random.choice([False, True])

        if set_all:
            case3 = True
        else:
            randval = random.random()
            
            case1 = randval < 1.0/3.0
            case2 = 1.0/3.0 <= randval < 2.0/3.0
            case3 = 2.0/3.0 <= randval
            
            if before_line is not None:
                if before_line == self.line or case1:
                    # Toggle checks without changing line mutation
                    self.line = before_line
                    self.orig_num = before_orig_line
                    self.checks = not before_checks
                    self.typesL = list(before_typesL)
            if case2:
                # Turn off checks for all TypeSpecialize instances mutation, and do not change current line
                if before_line is not None:
                    self.line = before_line
                    self.orig_num = before_orig_line
                    self.typesL = list(before_typesL)
                self.checks = False
                for transform in self.program_info.transformL:
                    if isinstance(transform, TypeSpecialize):
                        transform.checks = False

        if case3 or set_all:
            # TypeSpecialize all functions mutation
            self.checks = random.choice([False, True]) if not set_all else checks
            (lines, nodes) = self.get_line_for_mutation(ast.FunctionDef, is_valid, get_all=True)
            for i in range(len(lines)):
                line = lines[i]
                node = nodes[i]
                if line != self.line:
                    self.program_info.transformL.append(TypeSpecialize(self.program_info, line, None, self.checks))
                    
