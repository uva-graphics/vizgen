import transforms
import sys
import ast
import py_ast
import util

import z3_util
import z3

def insert_before_parent_list_fixed(node_list, s):
    """
    insert a string before a certain ast node if the node's parent has a field which is a list that contains the node
    """
    
    for i in range(len(node_list)):
        parent = node_list[i].parent
        
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                try:
                    index = value.index(node_list[i])
                    value[index:index] = ast.parse(s).body
                except:
                    continue
    return

class FornodeVisitor(ast.NodeVisitor):
    """
    an ast visitor that documents every outermost fornode
    """
    
    def __init__(self):
        self.fornode = []

    def visit_For(self, node):
        self.fornode.append(node)
        
def find_fornodes(node):
    """
    helper function to use FornodeVisitor to find outermost fornodes
    """
    
    fornodevisitor = FornodeVisitor()
    fornodevisitor.visit(node)
    return fornodevisitor.fornode
        
class NamenodeVisitor(ast.NodeVisitor):
    """
    This is a visitor that reads all the namenodes on either side of the assignment node, sort them according to read or write, then return the reads and writes of namenodes that represents arrays
    """
    def __init__(self):
        self.read = {}
        self.write = {}
        self.array_name = []
        #self.index = {}
        self.flag = None
        #self.index_name = None
        
    def visit_For(self, node):
        
        for field, value in ast.iter_fields(node):
            
            flag_cache = self.flag
            if field == 'target':
                self.flag = 'lhs'
            
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
            
            self.flag = flag_cache
    
    def visit_Assign(self, node):
        
        for field, value in ast.iter_fields(node):
            
            if field == 'targets':
                self.flag = 'lhs'
            elif field == 'value':
                self.flag = 'rhs'
                
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
            
            self.flag = None
            
    def visit_AugAssign(self, node):
        
        for field, value in ast.iter_fields(node):
            
            if field == 'target':
                self.flag = 'lhs'
            elif field == 'value':
                self.flag = 'rhs'
                
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
            
            self.flag = None

    def visit_Subscript(self, node):
        """
        tag variables that are followed by index as array
        """
        try:
            if node.value.id not in self.array_name:
                self.array_name.append(node.value.id)
        except:
            pass
        
        for field, value in ast.iter_fields(node):
            
            flag_cache = self.flag
            #index_cache = self.index_name
            #self.index_name = node.value
            
            if field == 'slice':
                self.flag = None
            elif field == 'ctx':
                self.flag = None
                
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
                
            self.flag = flag_cache
            #self.index_name = index_cache
    
    def visit_Name(self, node):
        """
        sets namenode as either read or write
        """
        if self.flag is not None:
            
            if self.flag == 'lhs':
                self.write.setdefault(node.id, []).append(node)
            
            elif self.flag == 'rhs':
                self.read.setdefault(node.id, []).append(node)
                
            #elif self.flag == 'index':
                #self.index.setdefault(self.index_name, []).append(node)
    
    def find_array_namenodes(self):
        """
        stores write namenode in self.lhs_namenodes, their names in self.lhs_names
        stores read namenode in self.rhs_namenodes, their names in self.rhs_names
        """
        self.lhs_namenodes = {}
        self.lhs_names = []
        self.rhs_namenodes = {}
        self.rhs_names = []
        
        for name in self.array_name:
            try:
                self.lhs_namenodes.setdefault(name, self.write[name])
                self.lhs_names.append(name)
            except:
                pass
            
            try:
                self.rhs_namenodes.setdefault(name, self.read[name])
                self.rhs_names.append(name)
            except:
                pass
            
class IndexVisitor(ast.NodeVisitor):
    """
    this visitor is used to track the namenodes inside the index
    """
    
    def __init__(self, write_nodes):
        self.write = []
        self.value = []
        self.possible_write = write_nodes
        
    def reset_class(self):
        self.write = []
        self.value = []
        
    def visit_Name(self, node):

        if node.id in self.possible_write:
            for i in range(1, len(self.possible_write[node.id]) + 1):
                if is_before(self.possible_write[node.id][-i], node) or is_iterator(self.possible_write[node.id][-i], node):
                    try:
                        self.write[-1].append(self.possible_write[node.id][-i])
                        break
                    except:
                        pass
                    
        
    def visit_Subscript(self, node):
        
        ind_node = node.slice
        for field, value in ast.iter_fields(ind_node):
            if field == 'value':
                if isinstance(value, ast.Tuple):
                    try:
                        for i in range(len(value.elts)):
                            ind_value = py_ast.dump_ast(value.elts[i])
                            self.value.append(ind_value)
                            self.write.append([])
                            self.visit(value.elts[i])
                    except:
                        pass
                else:
                    try:
                        ind_value = py_ast.dump_ast(value)
                        self.value.append(ind_value)
                        self.write.append([])
                        self.visit(value)
                    except:
                        pass
                        
            elif field == 'dims':
                try:
                    for i in range(len(value)):
                        ind_value = py_ast.dump_ast(value[i])
                        self.value.append(ind_value)
                        self.write.append([])
                        self.visit(value[i])
                except:
                    pass

def is_iterator(node1, node2):
    
    if not isinstance(node1.parent, ast.For):
        return False
    
    try:
        if not node1.parent.target == node1:
            return False
    except:
        return False      
    
    #if node1.parent_field != 'target':
     #   return False
    
    current = node2.parent
    
    while current is not None:
        if node1.parent == current:
            return True
        current = current.parent
    return False

def get_index(node):
    """
    get the index followed after the given namenode if possible
    """
    
    parent = node.parent
    if isinstance(parent, ast.Subscript):
        try:
            ind_node = parent.slice.value
            #if there's multiple dimensions in index
            if isinstance(ind_node, ast.Tuple):
                value = []
                for i in range(len(ind_node.elts)):
                    ind_value = py_ast.dump_ast(ind_node.elts[i])
                    value.append(ind_value)
                return value
            else:
                value = py_ast.dump_ast(ind_node)
                return value
        except:
            try:
                value = []
                ind_node = parent.slice.dims
                for i in range(len(ind_node)):
                    ind_value = py_ast.dump_ast(ind_node[i])
                    value.append(ind_value)
                return value
            except:
                return []
    else:
        return []
    
def is_before(node1, node2):
    """
    checks if definately appears before node2
    """
    
    parents1 = []
    current = node1.parent
    
    #first get parent_list of node1
    while current is not None:
        if isinstance(current, ast.Assign) or isinstance(current, ast.AugAssign):
            parent1 = current
            break
        parents1.append(current)
        current = current.parent
    
    current = node2
    
    if current in parents1:
        return False
    
    while current is not None:
        try:
            """if current.parent == parent1.parent:
                parent_field = current.parent_field
                field_list = getattr(current.parent, parent_field)
                list_index1 = field_list.index(parent1)
                list_index2 = field_list.index(current)
                if list_index2 > list_index1:
                    return True"""
            if current.parent == parent1.parent:
                for field, value in ast.iter_fields(parent1.parent):
                    if value == current or value == parent1:
                        return False
                    elif isinstance(value, list) and current in value and parent1 in value:
                        list_index1 = value.index(parent1)
                        list_index2 = value.index(current)
                        if list_index2 > list_index1:
                            return True
        except:
            return False
        
        current = current.parent
    return False
    
def match_ind(lhs, rhs, write_nodes):
    """
    lhs, rhs are namenodes that represents the occurence of a single array on lhs and rhs
    we'll see if the index in lhs contains every index in rhs, in this situation, the array is used simply as a dummy variable
    and return True
    if not, then the loop cannot be parallelized, thus return False
    """
    
    """
    eliminate several false alarms
    case 1:
    when the variable is only used for caching
    """
    
    indexvisitor = IndexVisitor(write_nodes)
    
    lhs_inds = []
    lhs_iterators = []
    
    for tnode in lhs:
        if isinstance(tnode.parent, ast.Subscript):
            indexvisitor.reset_class()
            indexvisitor.visit(tnode.parent)
            lhs_inds.append(indexvisitor.value)
            lhs_iterators.append(indexvisitor.write)
        else:
            lhs_inds.append([])
            lhs_iterators.append([])
        
    rhs_inds = []
    rhs_iterators = []
    
    for tnode in rhs:
        if isinstance(tnode.parent, ast.Subscript):
            indexvisitor.reset_class()
            indexvisitor.visit(tnode.parent)
            rhs_inds.append(indexvisitor.value)
            rhs_iterators.append(indexvisitor.write)
        else:
            rhs_inds.append([])
            rhs_iterators.append([])
    
    not_resolved = []
    
    """
    first round check:
    see if there's write which is exactly the same format, ie. input[a, b] = input[a, b]
    """
    for i in range(len(rhs_inds)):
        if rhs_inds[i] not in lhs_inds:
            not_resolved += [i]
        else:
            possible_writes = [j for j in range(len(lhs)) if lhs_inds[j] == rhs_inds[i] and is_before(lhs[j], rhs[i])]
            if len(possible_writes) == 0:
                not_resolved += [i]

    """
    second round check:
    see if there exist a[:] or a on lhs
    """
    if len(not_resolved) == 0:
        return True
    
    else:
        not_resolved2 = []
        
        for i in range(len(not_resolved)):
            ind = not_resolved[i]
            lhs_possible = [x for j, x in enumerate(lhs_inds) if len(lhs_inds[j]) <= len(rhs_inds[ind]) and is_before(lhs[j], rhs[ind])]
            
            for j in range(len(rhs_inds[ind])):
                
                lhs_possible_new = []
                for k in range(len(lhs_possible)):

                    if len(lhs_possible[k]) <=j:
                        lhs_possible_new.append(lhs_possible[k])
                    else:
                        try:
                            if lhs_possible[k][j] == ':':
                                lhs_possible_new.append(lhs_possible[k])
                            elif lhs_possible[k][j] == rhs_inds[ind][j]:
                                lhs_possible_new.append(lhs_possible[k])
                        except:
                            pass
                        
                lhs_possible[:] = lhs_possible_new[:]
        
            if len(lhs_possible) == 0:
                not_resolved2 += [ind]
            
    if len(not_resolved2) == 0:
        return True
    else:
        not_resolved_pair = []
        
        for i in range(len(not_resolved2)):
            ind = not_resolved2[i]
            possible_constant = [j for j, x in enumerate(rhs_iterators[ind]) if x == []]
            
            if len(possible_constant) == 0:
                for j in range(len(lhs_inds)):
                    not_resolved_pair.append([ind, j])
            else:
                for j in range(len(lhs_inds)):
                    can_resolved = False
                    for k in possible_constant:
                        if lhs_iterators[j][k] == []:
                            try:
                                z3_expr_lhs, z3_var_lhs = rewrite_expr_z3(lhs_inds[j][k], False)
                                z3_expr_rhs, z3_var_rhs = rewrite_expr_z3(rhs_inds[ind][k], False)
                                current_var = {}
                                for var in z3_var_lhs:
                                    if var not in current_var:
                                        current_var[var] = z3.Int(var)
                                for var in z3_var_rhs:
                                    if var not in current_var:
                                        current_var[var] = z3.Int(var)
                                        
                                solver = z3.Solver()
                                solver.add(eval(z3_expr_lhs, current_var) != eval(z3_expr_rhs, current_var))
                                if solver.check() == z3.sat:
                                    can_resolved = True
                                    break
                            except:
                                pass
                    if can_resolved == False:
                        not_resolved_pair.append([ind, j])
                        
    if len(not_resolved_pair) == 0:
        return True
    
    not_resolved_pair2 = []
    for inds in not_resolved_pair:
        lhs_node = lhs[inds[1]]
        rhs_node = rhs[inds[0]]
        
        try:
            lhs_parent = lhs_node.parent
            while not isinstance(lhs_parent, ast.Assign):
                lhs_parent = lhs_parent.parent
            lhs_parent = lhs_parent.parent
        
            rhs_parent = rhs_node.parent
            while not isinstance(rhs_parent, ast.Assign):
                rhs_parent = rhs_parent.parent
            rhs_parent = rhs_parent.parent
        
            if lhs_parent == rhs_parent and lhs_node.lineno < rhs_node.lineno:
                continue
            else:
                not_resolved_pair2.append(inds)
        except:
            not_resolved_pair2.append(inds)
            
    if len(not_resolved_pair2) == 0:
        return True
                                

    return False

def rewrite_expr_z3(r, is_py_ast=True):
    
    # Rewrites py_ast expression to a str expression that could be used in z3
    # Return (z3_expr_str, z3_varnames)
    z3_expr_str = (py_ast.dump_ast(r) if is_py_ast else r).strip()
    z3_expr_str = z3_expr_str.replace('.', '_').replace('[', '_').replace(']', '_')
 
    rp = py_ast.get_ast(z3_expr_str).body[0].value
    for node in py_ast.find_all(rp, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            if rp == node:
                rp = py_ast.get_ast('z3.Not(' + py_ast.dump_ast(node.operand) + ')').body[0].value
            else:
                py_ast.replace_node(rp, node, py_ast.get_ast('z3.Not(' + py_ast.dump_ast(node.operand) + ')').body[0].value)
    for node in py_ast.find_all(rp, ast.BoolOp):
        if isinstance(node.op, ast.And):
            if rp == node:
                rp = py_ast.get_ast('z3.And(' + py_ast.dump_ast(node.values[0]) + ',' + py_ast.dump_ast(node.values[1]) + ')')
            else:
                py_ast.replace_node(rp, node, py_ast.get_ast('z3.And(' + py_ast.dump_ast(node.values[0]) + ',' + py_ast.dump_ast(node.values[1]) + ')'))
        elif isinstance(node.op, ast.Or):
            if rp == node:
                rp = py_ast.get_ast('z3.Or(' + py_ast.dump_ast(node.values[0]) + ',' + py_ast.dump_ast(node.values[1]) + ')')
            else:
                py_ast.replace_node(rp, node, py_ast.get_ast('z3.Or(' + py_ast.dump_ast(node.values[0]) + ',' + py_ast.dump_ast(node.values[1]) + ')'))
    z3_expr_str = py_ast.dump_ast(rp)
            
    z3_vars = set()
    for node in py_ast.find_all(rp, ast.Name):
        z3_vars.add(node.id)
    return (z3_expr_str, z3_vars)

def ignore_shape(rhs_namenodes, rhs_names):
    """
    ignore rhs expression such as img.shape
    """
    for rhs_name in rhs_names:
        rhs_namenode = rhs_namenodes[rhs_name]
        for rhs_node in rhs_namenode:
            try:
                if rhs_node.parent.attr == 'shape':
                    rhs_namenodes[rhs_name].remove(rhs_node)
            except:
                pass
        if len(rhs_namenodes[rhs_name]) == 0:
            del rhs_namenodes[rhs_name]
            rhs_names.remove(rhs_name)
    
    return (rhs_namenodes, rhs_names)

def check_loop_parallel(node):
    """
    given a fornode, check if it can be parallelized
    """
    py_ast.add_parent_info(node)
    namenodevisitor = NamenodeVisitor()
    namenodevisitor.visit(node)
    namenodevisitor.find_array_namenodes()
    
    lhs_namenodes = namenodevisitor.lhs_namenodes
    rhs_namenodes = namenodevisitor.rhs_namenodes
    lhs_names = namenodevisitor.lhs_names
    rhs_names = namenodevisitor.rhs_names
    
    (rhs_namenodes, rhs_names) = ignore_shape(rhs_namenodes, rhs_names)
    
    for j in range(len(lhs_namenodes)):
        #if lhs_namenodes[j] in rhs_namenodes:
        if lhs_names[j] in rhs_names:
            """
            eliminate some false alarm situations
            """
            if not match_ind(lhs_namenodes[lhs_names[j]], rhs_namenodes[lhs_names[j]], namenodevisitor.write):
                return False
        else:
            """
            then prove that the iterator in the parallelized loop appears in the index of lhs array
            (so that writes won't be overlapping with each other
            """
            target_nodes = py_ast.find_all(node.target, ast.Name)
            target_index_mapping = {}
            ok = False
            
            indexvisitor = IndexVisitor(namenodevisitor.write)
            inds = []
            iterators = []
            for tnode in lhs_namenodes[lhs_names[j]]:
                if isinstance(tnode.parent, ast.Subscript):
                    indexvisitor.reset_class()
                    indexvisitor.visit(tnode.parent)
                    inds.append(indexvisitor.value)
                    iterators.append(indexvisitor.write)
                else:
                    inds.append([])
                    iterators.append([])
            
            for li in range(len(lhs_namenodes[lhs_names[j]])):
                lhs_node = lhs_namenodes[lhs_names[j]][li]
                if not isinstance(lhs_node.parent, ast.Subscript):
                    return False
                
                for target in target_nodes:
                    for k in range(len(inds[li])):
                        namenodes = iterators[li][k]
                        iter_index = [node for node in namenodes if node.id == target.id]
                        if len(iter_index):
                            ok = True
                            target_index_mapping.setdefault(target.id, []).append(k)       
                if not ok:
                    return False
                
            ok = False    
            for key, mapping in target_index_mapping.items():
                if not len(mapping) == len(lhs_namenodes[lhs_names[j]]):
                    continue
                is_identical = [ind == mapping[0] for ind in mapping]
                if all(is_identical):
                    ok = True
                    break
            
            if ok:
                return True
            
            """
            if there're multiple writes, can we prove 
            1. if they're non-overlapping?
            or
            2. if the loop iterator is on the same position of index?
            """     
            for li in range(len(lhs_namenodes[lhs_names[j]])):
                for lj in range(li + 1, len(lhs_namenodes[lhs_names[j]])):
                    
                    node1 = lhs_namenodes[lhs_names[j]][li]
                    node1_inds = inds[li]
                    node1_iterators = iterators[li]
                        
                    node2 = lhs_namenodes[lhs_names[j]][lj]
                    node2_inds = inds[lj]
                    node2_iterators = iterators[lj]
                    
                    ok = False
                    for k in range(min(len(node1_inds), len(node2_inds))):
                        try:
                            z3_expr_node1, z3_var_node1 = rewrite_expr_z3(node1_inds[k], False)
                            z3_expr_node2, z3_var_node2 = rewrite_expr_z3(node2_inds[k], False)
                            current_var = {}
                            for var in set(z3_var_node1) | set(z3_var_node2):
                                current_var[var] = z3.Int(var)
                            solver = z3.Solver()
                            solver.add(eval(z3_expr_node1, current_var) == eval(z3_expr_node2, current_var))
                            if solver.check() == z3.unsat:
                                ok = True
                                break
                        except:
                            pass
                        
                    if not ok:
                        return False
    return True

def main(path='test_programs/dependency_test.py'):
    
    s_orig = open(path, 'rt').read()
    node = ast.parse(s_orig)
    py_ast.add_parent_info(node)
    fornodes = find_fornodes(node)
    
    fornode = fornodes[1].body[2]
    result = check_loop_parallel(fornode)
    
    for i in range(len(fornodes)):
        result = check_loop_parallel(fornodes[i])
        
        if result:
            insert_before_parent_list_fixed(fornodes[i], '"""transform(parallel())"""')

    print(node.dumps())

def find_parallel_depth(node, base_depth):
    """
    helper function to determine whether the outermost of the inner loops are labeled parallel
    """
    result = check_loop_parallel(node)
    if result:
        return base_depth
    
    depth = []
    base_depth += 1
    for child in node.body:
        if isinstance(child, ast.For):
            child_depth = find_parallel_depth(child, base_depth)
            depth.append(child_depth)
    return depth
    
def unit_test():
    
    testnames = ['bilateral_grid/bilateral_grid_clean.py', 
                 'blur_one_stage/blur_one_stage.py', 
                 'blur_two_stage/blur_two_stage.py', 
                 'camera_pipe/camera_pipe_fcam.py', 
                 'composite/composite.py', 
                 'harris_corner_circle/harris_corner_circle.py', 
                 'interpolate/interpolate.py', 
                 'local_laplacian/local_laplacian_fuse.py', 
                 'mandelbrot/mandelbrot_animate.py', 
                 'optical_flow/optical_flow_patchmatch.py', 
                 'pacman/pacman_clean.py', 
                 'raytracer/raytracer_short_simplified_animate.py']
    
    testfiles = ['../apps/' + name for name in testnames]

    expected_depth = [[[[]], [], 0, 0, 0, 0], 
                      [0], 
                      [0, 0], 
                      [0, 0, 0, 0], 
                      [0], 
                      [0, 0, [[]], [1]], 
                      [0, [1, 1], [1, 1], 0], 
                      [0, [1], 0, [1], 0],
                      [0],
                      [[], 0, [[[3]]], 0],
                      [0, 0, 0, 0],
                      [0]]
    
    for i in range(len(testfiles)):
        
        filename = testfiles[i]
        s_orig = open(filename, 'rt').read()
        node = ast.parse(s_orig)
        py_ast.add_parent_info(node)
        fornodes = find_fornodes(node)

        for j in range(len(fornodes)):
            ind = 0
            depth = find_parallel_depth(fornodes[j], 0)
            assert depth == expected_depth[i][j], ('Parallel analysis failed', {'file': testnames[i], 'loop': j})
            
    util.print_twocol('static parallel analysis:', 'OK')

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        unit_test()
    else:
        main(sys.argv[1])
    
 
