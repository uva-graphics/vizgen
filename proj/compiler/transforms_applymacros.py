from transforms_util import *
from transforms_base import BaseTransform

class ApplyMacros(BaseTransform):
    """
    Special all-program transform that applies macros to all functions that have been type-specialized (apply this after type specialization).
    """
    def apply(self, s, extra_info=None):
        """
        Apply transform. If extra_info is not None then stores number of macros applied in extra_info['macro_count'].
        """
        verbose = get_verbose()
        macro_count = 0
        
        if verbose:
            util.print_header("ApplyMacros input:", s)

        r = py_ast.get_ast(s)
        py_ast.add_line_info(r)

        # Add additional macros for all the function calls made by the programmer
        func_call_macros = []
        for defnode in py_ast.find_all(r, (ast.FunctionDef,)):
            if typespecialize_header in defnode.name:
                orig_name = defnode.name[:defnode.name.index(typespecialize_header)]
            else:
                continue
            local_types = chosen_typespec_loads_py_ast(self.program_info, defnode)

            if verbose:
                print("local_types:", local_types)

            args = []
            for argnode in defnode.args.args:
                if isinstance(argnode, ast.arg) and argnode.arg in local_types:
                    args.append(argnode.arg)
            for nargs in range(len(args)+1):
                arg_parens = '(' + ','.join(args[:nargs]) + ')'
                current_macro = (orig_name + arg_parens, tuple([macros.IsCythonType(local_types[arg]) for arg in args[:nargs]]), defnode.name + arg_parens)
#                print(current_macro)
                func_call_macros.append(current_macro)
        if verbose:
            print('s_original:')
            print(s)
        
        L = macros.find_macros_ast(r, macros.macros + func_call_macros)
        if verbose:
            print('All macros:')
            pprint.pprint(L)

        replace_d = {}      # Map node IDs to nodes that they have been replaced with
        
        for macro_match in L:
            if verbose:
                print('macro:', macro_match)
            defnode = self.get_previous_node_py_ast(r, ast.FunctionDef, py_ast.get_line(r, macro_match.node))
            try:
                local_types = chosen_typespec_loads_py_ast(self.program_info, defnode)
                if local_types is None:
                    if verbose:
                        print('ApplyMacros.apply(): local_types is None')
                    continue
            except TransformError:
                if verbose:
                    print('skipping: did not find {} for macro_match={}'.format(chosen_typespec, macro_match))
                continue
            
            if verbose:
                print('local_types:', local_types)
            
            key = util.types_non_variable_prefix + macro_match.node_str
            try:
                arg_types = local_types[key]
            except KeyError:
                if verbose:
                    print('skipping: did not find key={} for macro_match={}'.format(key, macro_match))
                continue
            if verbose:
                print('arg_types before unpacking:', arg_types)
            arg_types = tuple([util.CythonType.from_cython_type(str(arg_type_str), self.program_info) for arg_type_str in arg_types.cython_type])
            if verbose:
                print('arg_types after unpacking:', arg_types, (type(arg_types[0])) if len(arg_types) else None)
            
            
            for (idx, (source_pattern, arg_types_func, dest_pattern)) in enumerate(macro_match.macroL):
                if verbose:
                    print('macro_match:', idx, '/', len(macro_match.macroL), (source_pattern, arg_types_func, dest_pattern))
                arg_func = arg_types_func
                if isinstance(arg_func, (list, tuple)):
                    def arg_func(program_info, type_args, expr_args):
                        wrong_lengths = len(type_args) != len(expr_args) or len(arg_types_func) != len(type_args)
                        if verbose:
                            print('type_args:', type_args, 'expr_args:', expr_args, 'arg_types_func:', arg_types_func, 'macro_match:', macro_match)
                        if wrong_lengths:
                            raise WrongMacroArgsError
                        success = True
                        ans_args = []
                        Xlen = None
                        for i in range(len(type_args)):
                            if arg_types_func[i].__name__.endswith('same'):
                                (current_success, current_arg) = arg_types_func[i](program_info, type_args[i], expr_args[i], Xlen)
                                Xlen = current_success
                            else:
                                (current_success, current_arg) = arg_types_func[i](program_info, type_args[i], expr_args[i])
                                if 'VecX' in arg_types_func[i].__name__:
                                    Xlen = current_success
                            if current_success is False:
                                success = False
                                break
                            ans_args.append(current_arg)
                        if success is False:
                            return (success, tuple(ans_args))
                        else:
                            return (Xlen, tuple(ans_args))
#
#                        return all( for i in range(len(args)))
                in_args = [astor.to_source(c_arg if id(c_arg) not in replace_d else replace_d[id(c_arg)]) for c_arg in macro_match.arg_nodes]
                if verbose:
                    print(' => macro match, in_args={}'.format(in_args))
                try:
                    (ok, ans_args) = arg_func(self.program_info, arg_types, in_args)
                except WrongMacroArgsError:
                    if verbose:
                        print(' => mismatched argument lengths, skipping')
                    ok = False
                if ok is not False:
                    Xlen = ok
                    if verbose:
                        print(' => macro types match: node_str={macro_match.node_str}, macro=({source_pattern}, {arg_types}, {dest_pattern})'.format(**locals()))
                
                    if hasattr(dest_pattern, '__call__'):
                        dest_pattern = dest_pattern(*arg_types)
                    
                    (lparen_count, rparen_count) = (dest_pattern.count('('), dest_pattern.count(')'))
                    if (lparen_count, rparen_count) == (1, 1):
                        replace_r = ast.parse(dest_pattern)
                        
                        try:
                            replace_func_name = replace_r.body[0].value.func.id
                        except:
                            replace_func_name = ''
                        if '_vecX' in replace_func_name:
                            assert isinstance(Xlen, int)
                            new_replace_func_name = replace_func_name.replace('_vecX', '_vec'+str(Xlen))
                            py_ast.replace_node(replace_r, replace_r.body[0].value.func, py_ast.get_ast(new_replace_func_name).body[0].value)
                            if new_replace_func_name not in macro_funcs_templated.templated_func:
                                macro_funcs_templated.add_templated_func(Xlen)
                        
                        call_args = py_ast.find_all(replace_r, ast.Call)[0].args
                        call_args = [_v for _v in call_args if isinstance(_v, ast.Name)]
                        
                        if len(call_args) != len(arg_types):
                            if verbose:
                                print('skipping: call_args length ({}) != arg_types length ({})'.format(len(call_args), len(arg_types)))
                                continue
                        if verbose:
                            print('ApplyMacros: arg_types={}, len(call_args)={}, call_args={}, ans_args={}'.format(arg_types, len(call_args), call_args, ans_args))
                        for i in range(len(call_args)):
                            py_ast.replace_node(replace_r, call_args[i], ast.parse(ans_args[i]).body[0].value)
#                            call_args[i].replace(ans_args[i]) #macro_match.arg_nodes[i])
                        replace_s = astor.to_source(replace_r) #.dumps()
                    elif (lparen_count, rparen_count) == (0, 0):
                        replace_s = dest_pattern
                    else:
                        assert False, 'expected zero or one set of parentheses in dest_pattern, found {}, {}'.format(lparen_count, rparen_count)

                    node_replace = ast.parse(replace_s).body[0].value
                    replace_d[id(macro_match.node)] = node_replace
                    py_ast.replace_node(r, macro_match.node, node_replace)
                    #macro_match.node.replace(replace_s)
                    macro_count += 1
                    if verbose:
                        print('macro replacement:', replace_s)
            if verbose:
                print()
        
        if extra_info is not None:
            extra_info['macro_count'] = macro_count
        
        return py_ast.dump_ast(r) #r.dumps()

    def mutate(self):
        self.line = 1
        self.orig_num = 1

