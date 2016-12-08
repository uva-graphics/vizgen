
import glob
import ast
import py_ast
import util

def test_get_line():
    filename_L = """
../apps/composite_gray/composite.py
../apps/blur_one_stage_gray/blur_one_stage.py
../apps/blur_two_stage_gray/blur_two_stage.py
../apps/camera_pipe/camera_pipe.py
../apps/composite/composite_rgb.py
../apps/local_laplacian/local_laplacian.py
../apps/interpolate/interpolate.py
../apps/bilateral_grid/bilateral_grid.py
../apps/blur_two_stage/blur_two_stage_rgb.py
../apps/raytracer/raytracer.py
../apps/optical_flow_patchmatch/optical_flow_patchmatch.py
../apps/pacman/pacman.py
../apps/blur_one_stage/blur_one_stage_rgb.py
../apps/harris_corner_circle/harris_corner_circle.py
../apps/mandelbrot/mandelbrot.py
../apps/composite/composite_rgb.py
""".split()
    #filename_L = [filename for filename in glob.glob('*.py') if not filename.startswith('_')]
    s_L = [open(filename, 'rt').read() for filename in filename_L]
    tup_L = [(filename_L[i], s_L[i]) for i in range(len(filename_L))]
    tup_L.sort(key = lambda tup: len(tup[1]))
    for (filename, s) in tup_L:
#        print(filename)
#        print(s)
        try:
            r = ast.parse(s)
            ok = True
        except:
            ok = False
        if not ok:
            continue
        for node in ast.walk(r):
            lineno = py_ast.get_line(r, node)
            assert lineno == node.lineno
    util.print_twocol('py_ast.get_line:', 'OK')

if __name__ == '__main__':
    test_get_line()

