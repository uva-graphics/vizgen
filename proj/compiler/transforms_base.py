from transforms_util import *

class BaseTransform:
    """
    Abstract base class for code transform.
    
    Code transforms should implement apply(), mutate(). If the number of arguments is changed, they should implement __init__() and args().
    Optionally, they can also implement dependencies(), is_consistent(), can_apply().
    """
    def __init__(self, program_info, line=None):
        """
        Build transform with arguments provided by the #transform*(cls(line, *args)) line in the Python source.
        
        The line argument (an int) should be stored in the instance attribute 'line'. If 'line' is None or not provided
        then the instance should be initialized (ordinarily by calling mutate(): this is what the default constructor does).
        Here program_info is an instance of compiler.ProgramInfo. Raises MutateError if the transform cannot be created.
        """
        self.program_info = program_info
        #if "Line " not in self.program_info.s_orig:    
        #   self.program_info.s_orig = preprocess.preprocess_input_python(self.program_info.s_orig)
        self.line = line
        if line is None:
            self.mutate()
        self.orig_num = get_orig_line_from_s_orig(program_info.s_orig, line)
    
    def args(self):
        """
        Return arguments to constructor so we can build a repr().
        """
        return (self.line,)
    
    def __deepcopy__(self, memo):
        """
        Default deep copy implementation: shares ProgramInfo instance, and uses args() to create a copy.
        """
        return self.__class__(self.program_info, *self.args())
    
    def __hash__(self):
        """
        Default hash implementation, used by resolve_dependencies(). Compares class and line only. Do not override in subclasses.
        """
        return hash((self.__class__, self.line))
    
    def __eq__(self, other):
        """
        Default equality operator, used by resolve_dependencies(). Compares class and line by default.
        
        Two transforms are 'equal' if they are on the same line and mutually exclusive.
        """
        return self.__class__ == other.__class__ and self.line == other.line
    
    def __repr__(self):
        """
        repr() method. Do not override in subclasses.
        """
        return transform_repr(self, self.args())
    
    def repr_no_line_number(self):
        """
        repr() with no line numbers. Do not override in subclasses.
        """
        return transform_repr(self, self.args()[1:])
    
    def __str__(self):
        """
        Shorter, potentially more human-readable repr(). Feel free to override in subclasses.
        """
        return self.__repr__()
    
    def apply(self, s):
        """
        Apply to (transform-annotated) Python source string s given list of transform instances transformL, returning new source string.
        
        The output source string should preserve any #transform() lines, since those will have apply() called in turn.
        """
        raise NotImplementedError
    
    def mutate(self):
        """
        Mutate parameters of current instance.
        
        Ordinarily one implements this by checking if the current instance is initialized (by checking if self.line is None or not).
        If the current instance is initialized (i.e. it has a line number), then its line number or parameters should be changed randomly.
        If not, then the line number and parameters should be initialized randomly.
        
        Raises MutateError if mutation cannot be done (most likely because the current transformation cannot be applied).
        """
        raise NotImplementedError

    def can_apply(self, r, lineno):
        """
        Return bool, whether transform can be applied to given line number in RedBaron object r.
        """
        raise NotImplementedError
    
    def dependencies(self):
        """
        List of BaseTransform instances that we depend on, i.e. that should also be applied if the current transform is applied.
        
        Dependency resolution is used by the compiler to make sure all the other dependencies are also applied.
        
        If multiple transforms are specified for a given line than any of them can be chosen by the dependency resolver.
        """
        return []
    
    def is_consistent(self):
        """
        Checks list of all transforms for consistency (i.e. they are suitable for being applied). Return a bool for whether consistent.
        """
        return True
    
    def apply_auto_cache(self, s, **kw):
        """
        Calls apply() if caching is disabled otherwise apply_cached().
        """
        if enable_cache:
            return self.apply_cached(s, **kw)
        else:
            return self.apply(s, **kw)
        
    def apply_cached(self, s, *args, **kw):
        """
        Apply transform, with results cached in all_transform_cache.
        """
        key = (s, repr(self), self.program_info.arraystorage_cache_key())
        if key in all_transform_cache:
            (ans, extra_info) = all_transform_cache[key]
            if 'extra_info' in kw:
                kw['extra_info'].clear()
                kw['extra_info'].update(extra_info)
            return ans
        
        ans = self.apply(s, *args, **kw)
        extra_info = kw.get('extra_info', None)
        all_transform_cache[key] = (ans, copy.deepcopy(extra_info))
        
        return ans
        
    def apply_and_delete(self, s):
        """
        Apply and delete current transform from annotated string s.
        """
        L = s.split('\n')
        line = self.annotated_line - 1
        assert 0 <= line < len(L)
        while not L[line].lstrip().startswith(transform_str) and line < len(L):
            line += 1
        if line >= len(L):
            util.print_header('BaseTransform.apply_and_delete, line={} does not start with {}'.format(line, transform_str), s, linenos=True)
            raise ValueError
        del L[line]
        s = '\n'.join(L)
        return self.apply_auto_cache(s)
    
    def get_line_for_mutation(self, nodecls, filter_func=None, get_all=False, outermost=False):
        """
        Helper function for mutation: returns (line, node) where line is an int and node is a RedBaron node matching given nodename.
        
        If filter_func is defined then only nodes with filter_func(node) returning True will be considered.
        
        If outermost is True then only return outer-most instances of nodename.
        """
        r = py_ast.get_ast(self.program_info.s_orig)
        py_ast.add_line_info(r)
        lines = []
        nodes = []
        existing_lines = [transform.line for transform in self.program_info.transformL if transform is not self and isinstance(transform, self.__class__)]
        for node in py_ast.find_all(r, nodecls):
#            print('------------------ Begin check loop ---------------------')
            in_defnode = False
            in_test_func = False
            is_outermost = True
            for parent in [node] + py_ast.parent_list(r, node):
#                print(parent, 'is defnode:', isinstance(parent, redbaron.DefNode))
                if isinstance(parent, nodecls) and parent is not node:
                    is_outermost = False
                if isinstance(parent, ast.FunctionDef):
                    in_defnode = True
#                    print('is indeed defnode, name=', parent.name)
                    if util.is_test_funcname(parent.name):
                        in_test_func = True
                        break
            if outermost and not is_outermost:
                continue
            if not in_defnode or (in_defnode and in_test_func):
                continue
            if filter_func is None or filter_func(node):
                line = py_ast.get_line(r, node)
                if line not in existing_lines:
                    lines.append(line)
                    nodes.append(node)
        if get_all:
            return (lines, nodes)
        if len(lines):
            i = random.randrange(len(lines))
            return (lines[i], nodes[i])
        else:
            raise MutateError

    def get_previous_node_py_ast(self, a, node_type, lineno=None):
        """
        Helper function to get previous node matching string name nodename from RedBaron instance r (or raise TransformError).
        """
        """
        Same as get_previous_node, but for py_ast

        Args:
            a, ast.AST, ast root created by py_ast
            node_type, class, type (not string) of the node to find
            lineno, int, line number to begin search at
        """
        if lineno is None:
            lineno = self.annotated_line

        all_nodes = py_ast.get_all_nodes_in_lineno_order(a, node_type=node_type)
        L = [node for node in reversed(all_nodes) if node.lineno <= lineno]

        if len(L):
            return L[0]

        raise TransformError("Could not get next node")

    def get_next_node_py_ast(self, a, node_type, lineno=None):
        """
        Helper function to get next node matching string name nodename from RedBaron instance r (or raise TransformError).
        """
        """Same as get_next_node, but for py_ast

        Args:
            a, ???, ast root created by py_ast
            node_type, class, type (not string) of the node to find
            lineno, int, line number to begin search at
        """

        if lineno is None:
            lineno = self.annotated_line

        all_nodes = py_ast.get_all_nodes_in_lineno_order(a, node_type=node_type)
        # all_nodes = py_ast.find_all(a, node_type)
        L = [node for node in all_nodes if node.lineno >= lineno]

        if len(L):
            return L[0]

        raise TransformError("Could not get next node")