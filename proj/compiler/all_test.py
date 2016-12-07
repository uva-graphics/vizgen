
"""
Runs all unit tests.
"""

import os
import util
import time

def run_test(py_filename):
    util.print_header(py_filename, '')
    assert os.system('python ' + py_filename) == 0
    print()

def main(all=True):
    T0 = time.time()
    run_test('loop_parallel_ast.py')
    run_test('test_prealloc_init.py')
    run_test('test_macros.py')
    run_test('test_py_ast.py')
    run_test('test_z3_util.py')
    run_test('test_type_infer.py')
    run_test('test_type_annotation.py')
    run_test('test_transforms.py')
    run_test('test_z_compiler.py')
    if all:
        run_test('test_validate.py')
    print()
    print('All unit tests passed in {} minutes'.format((time.time()-T0)/60.0))
    
if __name__ == '__main__':
    main()
