import sys; sys.path += ['../../..']
import compiler
import preprocess

if __name__ == "__main__":
    
    filename = '../apps/' + sys.argv[1] + '/' + sys.argv[2]
    path = '/home/yuting/annotating_compiler/proj/apps/' + sys.argv[1]
    full_filename = path + sys.argv[1] + '/' + sys.argv[2]
    
    s_orig = open(filename, 'rt').read()
    
    program_info = compiler.ProgramInfo(preprocess.preprocess_input_python(s_orig), path, types=None, log_transforms=True, compile_dir=None, is_verbose=False, filename=filename, max_types=None, apply_macros=True, verbose_level=1, out_file=None, preallocate=True, preallocate_verbose=True, full_filename=full_filename, quiet=False, use_4channel=False)
    
    run_kw_args = dict(clean=True, verbose=False, cython=True, extra_info={})
    run_kw_args['use_4channel'] = False
    initial_guess = program_info.get_initial_guesses((), run_kw_args)


"""
filename = '../apps/blur_two_stage/blur_two_stage.py'
s_orig = open(filename, 'rt').read()

program_info = compiler.ProgramInfo(preprocess.preprocess_input_python(s_orig), '/home/yuting/annotating_compiler/proj/apps/blur_two_stage', types=None, log_transforms=True, compile_dir=None, is_verbose=False, filename='../apps/blur_two_stage/blur_two_stage.py', max_types=None, apply_macros=True, verbose_level=1, out_file=None, preallocate=True, preallocate_verbose=True, full_filename='/home/yuting/annotating_compiler/proj/apps/blur_two_stage/blur_two_stage.py', quiet=False, use_4channel=False)

run_kw_args = dict(clean=True, verbose=False, cython=True, extra_info={})
run_kw_args['use_4channel'] = False
initial_guess = program_info.get_initial_guesses((), run_kw_args)

print(initial_guess)
"""