
"""
Preprocess input Python before doing any tuning/compiling.

Replace (x)**(y) with pow(x, y) when the second argument is not an integer. This is because Cython's x**y operator
does not return the same type as Python (e.g. 2**(-2) => 0 in Cython, whereas in Python 2**(-2) => 0.25). Also
we can apply the macro facility in macros.py when pow(x, y) is called because it has a function form.
"""

import py_ast
import ast

def add_str_node(root):
    nodeList = [i for i in ast.walk(root) if (isinstance(i, ast.stmt))]
    for i in nodeList:
        if(hasattr(i,'value') and hasattr(i.value,'s') and hasattr(i.value,'s')):
            continue
        if(hasattr(i,'lineno')):
            s = ast.Str('Line ' + str(i.lineno) + '')
            new_node = ast.Expr()
            new_node.value = s
            py_ast.add_after_node(root, i, new_node)
    return root

def preprocess_input_python(s):
    rootnode = py_ast.get_ast(s)
    rootnode = add_str_node(rootnode)

    for node in py_ast.find_all(rootnode, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            if not py_ast.is_int_constant_py_ast(py_ast.dump_ast(node.right)):
                new_str = 'pow(' + py_ast.dump_ast(node.left) + ', ' + py_ast.dump_ast(node.right) + ')'
                new_node = py_ast.get_ast(new_str).body[0].value
                py_ast.replace_node(rootnode, node, new_node)
    
    for node in py_ast.find_all(rootnode, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.attr in ['floow', 'ceil']:
                if node.func.value.id == 'math':
                    py_ast.replace_node(node, node.func.value, py_ast.get_ast('numpy').body[0].value)
    
    ans = py_ast.dump_ast(rootnode)
    ans = 'import numpy\n' + ans
    
    lines = ans.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#'):
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                del lines[j]
        i += 1
        
    ans = '\n'.join(lines)
    
    return ans