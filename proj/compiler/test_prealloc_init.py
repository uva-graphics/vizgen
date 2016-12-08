import transforms
import ast
import py_ast
import util

def test_prealloc_init():
    fnames = ['blur_two_stage.py',
              'bilateral_grid.py',
              'camera_pipe_fcam.py',
              'harris_corner_annotated.py']
    
    array_maps = [{'two_stage_blur': ['temp_img', 'output_img']},
                  {'bilateral_grid': ['blurx', 'blury', 'blurz', 'interpolated', 'output_img']},
                  {'demosaic': ['colorMatrix', 'inBlock', 'linear']},
                  {'harris_corner': ['Ix', 'Iy', 'Ixy', 'Iy', 'Iy2', 'R']}]
    
    for i in range(len(fnames)):
        fname = fnames[i]
        array_map = array_maps[i]
        file_name = 'test_programs/' + fname
        s_orig = open(file_name).read()
        rootnode = py_ast.get_ast(s_orig)
        py_ast.add_parent_info(rootnode)
        defnodes = py_ast.find_all(rootnode, ast.FunctionDef)
        
        for funcname in array_map:
            defnode = [node for node in defnodes if node.name == funcname][0]
            for array_var in array_map[funcname]:
                (assignnode, call_args, numpy_func) = transforms.preallocate_find_assignnode_py_ast(rootnode, funcname, array_var)
                init_str = transforms.preallocate_find_init(defnode, array_var, assignnode, call_args, rootnode)
                assert init_str == ''
                
    util.print_twocol('prealloc_init:', 'OK')
             
if __name__ == '__main__':
    test_prealloc_init()
