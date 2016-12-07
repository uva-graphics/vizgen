import ast
import astor
import json
# import redbaron
import sys
import time
import copy

STR_COMMENT_PREFIX = "PY_AST_COMMENT>"

TYPE_ANNOTATION_LIST = ["#type:", "# type:"]

TRANS_PREFIXES = (
    "#cython",
    "#cython_replace",
    "#c_rewrite_cython",
    "#preallocate_intermediate_cython",
    "#openmp_add_private",
    "#_chosen_typespecialize",
    "#_after_cdefs_",
    "#transform",
    "#deleted_comment") + tuple(TYPE_ANNOTATION_LIST)

TYPE_INDENT = 100

class BottomUpVisitor:
    """
    Similar to ast.NodeVisitor but always recurse into children: visit in bottom-up order by default.
    """
    def __init__(self, bottom_up=True, strict_line_order=False, verbose=False, make_unique=False):
        """
        If bottom_up is False then use top-down order.
        If strict_line_order is True then sort and visit every node in order by line number, and only within
        nodes with the same line number, proceed from bottom-up or top-down order.
        If make_unique is True then ensure that all nodes in the returned tree are unique (have different ids)
        by copying them if necessary.
        """
        self.bottom_up = bottom_up
        self.strict_line_order = strict_line_order
        self.verbose = verbose
        self.make_unique = make_unique
    
    def generic_visit(self, node):
        if self.verbose:
            print("Couldn't find visit() for node", node)

    def visit_one_node(self, node, lineno=None):
        method = 'visit_' + node.__class__.__name__
        getattr(self, method, self.generic_visit)(node)

    def visit(self, node):
        if self.strict_line_order:
            seen_ids = set()
            
            all_nodes = []
            last_lineno = [1]
            def visit_recurse(current, depth):
                if hasattr(current, 'lineno'):
                    last_lineno[0] = current.lineno
                all_nodes.append((last_lineno[0], -depth if self.bottom_up else depth, current))
                self.visit_child_nodes(current, lambda child: visit_recurse(child, depth+1), seen_ids)
            visit_recurse(node, 0)
#            print('done visiting all nodes')
#            print('all_nodes:', all_nodes)
            
            all_nodes.sort(key=lambda tup_key: tup_key[:2])
            
            for (lineno, depth, current) in all_nodes:
                self.visit_one_node(current, lineno)
            return
        
        if self.bottom_up:
            self.visit_child_nodes(node)
        self.visit_one_node(node)
        if not self.bottom_up:
            self.visit_child_nodes(node)
    
    def visit_child_nodes(self, node, visit_func=None, seen_ids=None):
        if visit_func is None:
            visit_func = self.visit
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i in range(len(value)):
                    item = value[i]
                    if isinstance(item, ast.AST):
                        if self.make_unique:
                            if id(item) in seen_ids:
                                if hasattr(item, 'parent'):
                                    item.parent = None
                                if hasattr(item, 'parent_functiondef'):
                                    item.parent_functiondef = None
                                if hasattr(item, 'parent_functionname'):
                                    item.parent_functionname = None
                                item = copy.deepcopy(item)
                                value[i] = item
                                setattr(node, field, value)
                            seen_ids.add(id(item))
                        visit_func(item)
#            elif isinstance(value, ast.expr_context):
#                print('found context', value)
#                if self.make_unique:
#                    print('cloning context', value)
#                    value = copy.deepcopy(value)
#                    setattr(node, field, value)
#                visit_func(value)
            elif isinstance(value, ast.AST):
                if self.make_unique:
                    if id(value) in seen_ids:
                        if hasattr(value, 'parent'):
                            value.parent = None
                        if hasattr(value, 'parent_functiondef'):
                            value.parent_functiondef = None
                        if hasattr(value, 'parent_functionname'):
                            value.parent_functionname = None
                        value = copy.deepcopy(value)
                        seen_ids.add(id(value))
                        setattr(node, field, value)
                    else:
                        seen_ids.add(id(value))
                visit_func(value)

class TopDownVisitor(BottomUpVisitor):
    """
    Similar to ast.NodeVisitor but always recurse into children: visit in top-down order by default.
    """
    def __init__(self, bottom_up=False, strict_line_order=False, verbose=False):
        BottomUpVisitor.__init__(self, bottom_up=bottom_up, strict_line_order=strict_line_order, verbose=verbose)

def isStr(node):
    if(node == None):
        return -1
    if(node.__class__.__name__ == 'Expr' and hasattr(node, 'value') and hasattr(node.value, 's') and len(node.value.s.split()) >= 2 and node.value.s.split()[0] == 'Line'):
        return (int)(node.value.s.split()[1])
    return -1

def z3_div_solver(s):
    return s

def str_for_loop_fusion(s):
    s = s.replace('int', '')
    rootnode = get_ast(s)
    for node in find_all(rootnode, ast.Call):
        if hasattr(node, 'func') and hasattr(node.func, 'id') and node.func.id == 'pow':
            new_str = '(' + dump_ast(node.args[0]) + ' ** ' + dump_ast(node.args[1]) + ')'
            new_node = get_ast(new_str).body[0].value
            replace_node(rootnode, node, new_node)
    return dump_ast(rootnode)



#Assuming every node has the parent attribute
def getLineNum(target):
    if(hasattr(target, 'parent') == False):
        return -1
    parent = target.parent
    isFound = False
    if(parent == None or hasattr(parent, '_fields') == False):
        return -1
    for list in parent._fields:
        if(getattr(parent, list).__class__.__name__ == 'list' and (target in getattr(parent, list))):
            for element in getattr(parent, list):
                if(element == target):
                    isFound = True
                if(isFound and isStr(element) != -1):
                    return isStr(element)
    return -1

def add_parent_info(root_node):
    """
    Add parent attribute for all nodes at and under root_node, recursively.
    """
    class AddParents(TopDownVisitor):
        def generic_visit(self, node):
            if not hasattr(node, 'parent'):
                node.parent = None
            for child in ast.iter_child_nodes(node):
                child.parent = node
    AddParents().visit(root_node)


def add_line_info(root_node):
    """
    Add lineno attribute for all nodes at and under root_node, recursively.
    """
    class AddLineNumbers(BottomUpVisitor):
        def __init__(self):
            BottomUpVisitor.__init__(self, strict_line_order=True, make_unique=True)
        def visit_one_node(self, node, lineno=None):
#            print(node, lineno, getattr(node, 'lineno', None))
            if not hasattr(node, 'lineno'):
                node.lineno = lineno
            else:
                if node.lineno != lineno:
                    print(node, lineno, node.lineno)
                    print(astor.dump(root_node))
                    assert False
            BottomUpVisitor.visit_one_node(self, node, lineno)
    AddLineNumbers().visit(root_node)

def get_line(root_node: ast.AST, node: ast.AST):
    """
    Get line number for given node 'node', which has ancestor root_node. Always succeeds, unlike node.lineno.
    """
    try:
        return node.lineno
    except AttributeError:
        add_line_info(root_node)
        return node.lineno

def get_parent(root_node: ast.AST, node: ast.AST):
    """
    Get parent node of 'node', which has an ancestor root_node.
    
    If parent information is missing then it is added for all nodes.
    """
    try:
        return node.parent
    except AttributeError:
        add_parent_info(root_node)
    return node.parent

def parent_function(root_node: ast.AST, node: ast.AST):
    """
    Return parent FunctionDef of given node 'node' in ast, or if not found raise KeyError.
    """
    for parent in parent_list(root_node, node):
        if isinstance(parent, ast.FunctionDef):
            return parent
    raise KeyError

def parent_list(root_node: ast.AST, node: ast.AST):
    """
    List of parents of given node, which has an ancestor root_node.
    """
    ans = []
    current = node
    while current is not None:
        current = get_parent(root_node, current)
        if current is not None:
            ans.append(current)
    return ans

def get_all_transform_comments(a):
    """Given the AST, this returns a list of every node in the AST that is a 
    transform comment (encoded as a string). This function is analogous to 
    redbaron_util.find_all_transforms().
    """

    return [n for n in find_all(a, ast.Str)
        if n.s.startswith(STR_COMMENT_PREFIX + "#transform")]

def get_all_transform_comments_str(s):
    """Given a string-representation of a program, this will return a list of 
    all of the #transform() comments. This does a similar job as 
    get_all_transform_comments(), but this preserves the line number of the 
    comment in the input program string. Line number preservation becomes an 
    issue when mixing the use of RedBaron and py_ast, because py_ast doesn't
    preserve vertical whitespace.
    """

    program_lines = s.split("\n")
    result = []

    for i in range(len(program_lines)):
        line = program_lines[i]

        if line.lstrip(' ').startswith("#transform"):
            transform_node = get_ast(line).body[0].value
            transform_node.lineno = i + 1
            result.append(transform_node)

    return result

def count_whitespace_at_beginning(line):

    return len(line) - len(line.lstrip(' '))

def comment_to_str(source_prog, transform_prefix, typing_list = TYPE_ANNOTATION_LIST):
    """This finds compiler comments in the source program and then converts them
    into a string so we can keep them in the AST. This function was made because
    Python's built-in AST library ignores comments, but keeps string constants
    in the tree.
    We use json.dumps when encoding the string because the compiler comments may
    include arbitrary single- or double-quotes. json.dumps() lets you encode
    arbitrary characters into a string.
    Args:
        source_prog, string, the input source code, as a string
        transform_prefix, tuple of strings, all of the compiler comment prefixes
            we need to wrap
    """

    lines = source_prog.split("\n")

    for i in reversed(range(len(lines))):
        for typing in typing_list:
            if typing in lines[i] and not lines[i].strip().startswith(typing):
                ind = lines[i].find(typing)
                typing_comment = lines[i][ind : len(lines[i])]
                lines[i] = lines[i][0 : ind]
                indentation = len(lines[i]) - len(lines[i].lstrip())
                if lines[i].rstrip().endswith(':'):
                    indentation += 4
                typing_comment = indentation * ' '  + json.dumps(STR_COMMENT_PREFIX + typing_comment)
                lines.insert(i + 1, typing_comment)
            
        if lines[i].lstrip(' ').startswith(transform_prefix):

            # we should indent the current line (i.e. the transform we're 
            # wrapping) so it matches the amount of indent of the next line
            if i < len(lines) - 1:
                j = i + 1
                while j < len(lines) - 1:
                    if lines[j].strip() == '':
                        j += 1
                    else:
                        break
                preceding_whitespace_on_next_line = \
                    count_whitespace_at_beginning(lines[j])

                if preceding_whitespace_on_next_line > 0:
                    lines[i] = (' ' * preceding_whitespace_on_next_line) + \
                        json.dumps(STR_COMMENT_PREFIX + lines[i])

                else:
                    lines[i] = json.dumps(STR_COMMENT_PREFIX + lines[i])

            else:
                lines[i] = json.dumps(STR_COMMENT_PREFIX + lines[i])

    return "\n".join(lines)

def get_ast(source_prog):
    """Returns the ast of the program, with comments converted into string 
    literals.
    Args:
        source_prog, string, string version of the source code
    """
    wrapped_str = comment_to_str(source_prog, TRANS_PREFIXES)
    node = ast.parse(wrapped_str)
    add_parent_info(node)
    nodeList = [i for i in ast.walk(node) if (isinstance(i, ast.stmt))]
    for i in nodeList:
        if(hasattr(i,'lineno')):
            #i.orig_lineno = 1
            temp = getLineNum(i)
            if(temp != -1):
                i.orig_lineno = temp
                #a = 1
    nodeList = [i for i in ast.walk(node)]
    for i in nodeList:
        if(hasattr(i,'parent')):
            delattr(i, 'parent')
    return node

def to_source_any(n):
    """
    Convert AST node to string, handling all node types, without fixing comments.
    """
    try:
        return astor.to_source(n)
    except AttributeError:
        pass
    cls = n.__class__
    if cls in astor.misc.all_symbols:
        return astor.misc.all_symbols[cls]
    def wrap(s):
        return '___' + s + '___'
    extra_d = {ast.Load: wrap('load'),
               ast.Store: wrap('store'),
               ast.Del: wrap('del'),
               ast.AugLoad: wrap('augload'),
               ast.AugStore: wrap('augstore'),
               ast.Param: wrap('param'),
               ast.keyword: wrap('keyword')}
    if cls in extra_d:
        return extra_d[cls]
    raise AttributeError('unknown node type {}'.format(cls))

def dump_ast(a):
    """
    Convert AST node to string and fix comments
    """
    dump_str = astor.to_source(a)
    return str_to_comment(dump_str)

def str_to_comment(source_prog, typing_list = TYPE_ANNOTATION_LIST):
    """This performs the inverse of comment_to_str. We use eval() to perform the
    inverse of json.dumps().
    """

    lines = source_prog.split("\n")

    for i in range(len(lines)):
        line = lines[i]

        if line.lstrip(' ').startswith(("\"" + STR_COMMENT_PREFIX,
                "\'" + STR_COMMENT_PREFIX)):
            whitespace_at_beginning = count_whitespace_at_beginning(line)
            prefix_len = len(STR_COMMENT_PREFIX) + 1 # +1 for quote
            whitespace_after_prefix = count_whitespace_at_beginning(
                line[(whitespace_at_beginning + prefix_len):])

            start_index = whitespace_at_beginning + prefix_len + \
                whitespace_after_prefix
            stop_index = len(line)
            newline = line[start_index:stop_index]

            # match closing quote:
            if newline[-1] == "\"":
                lines[i] = (' ' * whitespace_at_beginning) + \
                    eval("\"%s" % line[start_index:stop_index])
            else:
                lines[i] = (' ' * whitespace_at_beginning) + \
                    eval("\'%s" % line[start_index:stop_index])
            
            for typing in typing_list:
                if lines[i].strip().startswith(typing):
                    line_len = len(lines[i - 1].rstrip())
                    if line_len < TYPE_INDENT:
                        lines[i - 1] = lines[i - 1].rstrip() + ' ' * (TYPE_INDENT - line_len) + lines[i].strip()
                    else:
                        lines[i - 1] = lines[i - 1].rstrip() + ' ' + lines[i].strip()
                    lines[i] = ''

    return "\n".join(lines)

def nodes_are_equal(node1, node2):
    """Returns true if node1 and node2 are equal
    This is kinda a hack to solve a problem where (for example) node1 is an 
    expression node and node2 is a terminal for an expression (e.g. a Call()
    node). In both cases, the string representations of the nodes are the same 
    and they also have the same line numbers and column offsets. However, 
    because they are different objects, normal == equality would return false.
    """

    try:
        return dump_ast(node1).strip() == dump_ast(node2).strip() and \
            node1.lineno == node2.lineno and \
            node1.col_offset == node2.col_offset
    except:
        return False

def find_node_recursive(node, goal):
    """Given a root node 'node' and a goal node, return (goal, parent).
    If goal cannot be found, this returns None, parent. Otherwise this returns
    node, parent.
    """
    return (goal, get_parent(node, goal))

def replace_node(ast_root, node_to_replace, replacement_node):
    """Replaces node_to_replace with replacement_node in the ast.
    """

    # first, search for the node
    #node, parent = find_node_recursive(ast_root, node_to_replace)
    if not hasattr(node_to_replace, 'parent'):
        add_parent_info(ast_root)

    # if you can't find the node you want to replace, raise an error
    if not hasattr(node_to_replace, 'parent'):
        raise ValueError("Node %s not found in ast: %s" % (
            str(node_to_replace),
            dump_ast(node_to_replace)))
    
    parent = node_to_replace.parent
    
    # otherwise, find the node, within its parent, and replace it
    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], ast.AST) and \
                   nodes_are_equal(value[i], node_to_replace):
                    value[i] = replacement_node
                    return

        elif isinstance(value, ast.AST) and nodes_are_equal(value, node_to_replace):
            setattr(parent, field, replacement_node)
            setattr(replacement_node, 'parent', parent)
            return

def get_col_offset(node):
    if hasattr(node, "col_offset"):
        return node.col_offset

    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            result = get_col_offset(value)

            if result is not None:
                return result

    return None

def add_before_node(ast_root, before_node, node_to_add):
    """Attempts to add node_to_add before before_node
    For example, if you had the code:
        def foo(j):
            for i in range(j):
                print(i)
    and before_node was "for i in range(j):" and node_to_add was "print(2)",
    the result would be:
        def foo(i):
            print(2)
            for i in range(j):
                print(i)
    """

    node, parent = find_node_recursive(ast_root, before_node)

    if node is None:
        raise ValueError("Node %s not found in ast: %s" % (
            str(before_node),
            dump_ast(before_node)))

    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], ast.AST) and \
                   nodes_are_equal(value[i], node):
                    value.insert(i, node_to_add)
                    return

def add_after_node(ast_root, after_node, node_to_add):
    """Same idea as add_before_node, but in this case add it after after_node
    """

    node, parent = find_node_recursive(ast_root, after_node)

    if node is None:
        raise ValueError("Node %s not found in ast: %s" % (
            str(after_node),
            dump_ast(after_node)))

    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], ast.AST) and \
                   nodes_are_equal(value[i], node):
                    value.insert(i + 1, node_to_add)
                    return

def find_all(m, cls):
    """
    Find all nodes in ast.AST instance m that are instances of cls (a class or tuple of classes).
    
    (Replacement for redbaron find_all() function).
    """
    return [node for node in ast.walk(m) if isinstance(node, cls)]

def get_binary_op_str(bin_op_node):
    """Returns the string representation of the binary operator node (e.g. +, -,
    etc.). For some reason astor doesn't implement this???
    """

    if isinstance(bin_op_node, ast.Add):
        return "+"

    elif isinstance(bin_op_node, ast.Sub):
        return "-"

    elif isinstance(bin_op_node, ast.Mult):
        return "*"

    elif isinstance(bin_op_node, ast.Div):
        return "/"

    elif isinstance(bin_op_node, ast.Mod):
        return "%"

    elif isinstance(bin_op_node, ast.Pow):
        return "**"

    elif isinstance(bin_op_node, ast.LShift):
        return "<<"

    elif isinstance(bin_op_node, ast.RShift):
        return ">>"

    else:
        raise ValueError("No string defined for binary operator node %s" % \
            bin_op_node.__class__.__name__)

def get_all_nodes_in_lineno_order(ast_root, node_type=None):
    """Returns all nodes in the ast_root subtree, sorted in order of their line 
    number. This also includes an option to only return nodes of a certain type
    or types (i.e. node_type could be a tuple)
    """

    all_nodes = []

    if node_type is not None:
        for node in ast.walk(ast_root):
            if isinstance(node, node_type) and \
               hasattr(node, 'lineno') and \
               hasattr(node, 'col_offset'):
                all_nodes.append(node)

    else:
        for node in ast.walk(ast_root):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                all_nodes.append(node)

    # this sorts the nodes by their line number and their column offset
    # (i.e. nodes that are on the same line are sorted from left to right)
    return sorted(all_nodes, key=lambda node: (node.lineno, node.col_offset))

def get_all_nodes_in_bfs_order(ast_root):
    q = [ast_root]
    result = []

    while len(q) > 0:
        top = q.pop(0)
        result.append(top)

        for field, value in ast.iter_fields(top):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        q.append(item)
            elif isinstance(value, ast.AST):
                result.append(value)

    return result

def get_comment_strings(ast_root):
    """Similar idea to get_all_nodes_in_lineno_order, but this returns all
    comments (which have been encoded as string-literals from comment_to_str)
    """

    comment_nodes = []

    for node in ast.walk(ast_root):
        if isinstance(node, ast.Str) and \
           dump_ast(node).lstrip(" ").startswith("#"):
            comment_nodes.append(node)

    return sorted(comment_nodes, key=lambda n: (n.lineno, n.col_offset))

def is_int_constant_py_ast(s):
    """
    py_ast version on is_int_constant in redbaron_util.py
    """
    s = s.strip()
    rootnode = get_ast(s).body
    if len(rootnode) == 1 and isinstance(rootnode[0], ast.Expr):
        node = rootnode[0].value
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return isinstance(node.operand, ast.Num)
        else:
            return isinstance(node, ast.Num)
    return False

class FindDefVisitor(ast.NodeVisitor):
    """
    an ast visitor that finds defnodes with corresponding name
    """
    
    def __init__(self, defname):
        self.defnode = []
        self.defname = defname
    
    def visit_Def(self, node):
        if node.name == self.defname:
            self.defnode.append(node)
            
    def visit_FunctionDef(self, node):
        if node.name == self.defname:
            self.defnode.append(node)
            
class FindAssignVisitor(ast.NodeVisitor):
    """
    an ast visitor that finds assignment nodes
    """
    
    def __init__(self):
        self.assignnode = []
    
    def visit_Assign(self, node):
        self.assignnode.append(node) 
