import transforms
import py_ast
import compiler

s_orig = open('test_programs/blur_two_stage.py').read()
program_info = compiler.ProgramInfo(s_orig)
fusion = transforms.LoopFusion(program_info, 'two_stage_blur', 'output_img')
s_apply = fusion.apply(s_orig)
file=open('test_fusion_example.py', 'w+')
file.write(s_apply)
file.close()