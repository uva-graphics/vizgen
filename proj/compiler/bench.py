
"""
Benchmark all apps (overnight run).
"""

import os
import shutil
import argparse
import time

def configure_app(appdir, appname, do_rgb, do_gray):
    filename = '../apps/{appdir}/{appname}_config.py'.format(**locals())

    s = """
do_rgb  = {do_rgb}     # Whether to run RGB test
do_gray = {do_gray}    # Whether to run gray test
""".format(**locals())

    with open(filename, 'wt') as f:
        f.write(s)

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks of example programs.')
    parser.add_argument('--no-tune', dest='tune', action='store_false', help='do not run tuner. only find speedups vs comparison baselines')
    parser.add_argument('--print', dest='do_print', action='store_true', help='do not run tuner or find speedups: only print compiler commands')
    parser.add_argument('--safe', dest='safe', action='store_true', help='supply --safe option to compiler')
    parser.add_argument('--check', dest='check', action='store_true', help='only check existence of input files; do not run compiler')
    parser.add_argument('--float32', dest='float32', action='store_true', help='use applications variants that use 32-bit floats')
    parser.add_argument('--args', dest='args', help='extra args to supply to compiler: use --args "\-\-blah"; escape dashes using \-')
    parser.add_argument('--no-validate', dest='validate', action='store_false', help='disable profiling and validation of speedups')
    parser.set_defaults(tune=True)
    parser.set_defaults(do_print=False)
    parser.set_defaults(safe=False)
    parser.set_defaults(check=False)
    parser.set_defaults(float32=False)
    parser.set_defaults(validate=True)
    parser.set_defaults(args='')
    args = parser.parse_args()
    args.args = args.args.replace('\-', '-')
    cmd_options = ''
    if not args.tune:
        cmd_options = ' --no-tune'
    if args.safe:
        cmd_options += ' --safe'
    else:
        if args.validate:
            cmd_options += ' --profile --validate-speedups-after'

    if len(args.args):
        cmd_options += ' ' + args.args

    def system(s, add_options=True, check=True):
        if args.float32 and check:
            L = s.split()
            filename = L[2]
            filename = filename.replace('.py', '_float32.py')
            L[2] = filename
            s = ' '.join(L)
        if args.check and check:
            filename = s.split()[2]
            if not os.path.exists(filename):
                print('*** File missing: {}'.format(filename))
            else:
                print('    File found:   {}'.format(filename))
            return
        if add_options:
            s += cmd_options
        print(s)
        if not args.do_print:
            os.system(s)

    outdir = 'out_' + time.strftime('%Y_%m_%d')
    if os.path.exists(outdir):
        outdir0 = outdir
        counter = 2
        while True:
            outdir = outdir0 + '_' + str(counter)
            if not os.path.exists(outdir):
                break
            counter += 1

#    if os.path.exists('out'):
#        shutil.rmtree('out')

#    system('python compiler.py ../apps/composite/composite_4channel.py --out-dir {}/composite_4channel'.format(outdir))

#    system('python compiler.py ../apps/bilateral_grid/bilateral_grid_clean_small.py --out-dir {}/bilateral_grid_clean_small'.format(outdir))

#    system('python compiler.py ../apps/interpolate/interpolate_float.py --out-dir {}/interpolate_float'.format(outdir))

    system('python compiler.py ../apps/mandelbrot/mandelbrot.py --out-dir {}/mandelbrot'.format(outdir))

    system('python compiler.py ../apps/composite/composite_rgb.py --out-dir {}/composite_rgb'.format(outdir))

    system('python compiler.py ../apps/composite_gray/composite.py --out-dir {}/composite_gray'.format(outdir))

    system('python compiler.py ../apps/blur_one_stage/blur_one_stage_rgb.py --out-dir {}/blur_one_stage_rgb'.format(outdir))

    system('python compiler.py ../apps/blur_one_stage_gray/blur_one_stage.py --out-dir {}/blur_one_stage_gray'.format(outdir))

    system('python compiler.py ../apps/blur_two_stage/blur_two_stage_rgb.py --out-dir {}/blur_two_stage_rgb'.format(outdir))

    system('python compiler.py ../apps/blur_two_stage_gray/blur_two_stage.py --out-dir {}/blur_two_stage_gray'.format(outdir))

    system('python compiler.py ../apps/interpolate/interpolate.py --out-dir {}/interpolate'.format(outdir))

    system('python compiler.py ../apps/optical_flow_patchmatch/optical_flow_patchmatch.py --out-dir {}/optical_flow_patchmatch'.format(outdir))

    system('python compiler.py ../apps/pacman/pacman.py --out-dir {}/pacman'.format(outdir))

#    system('python compiler.py ../apps/raytracer/raytracer.py --out-dir out/raytracer')

#    system('python compiler.py ../apps/blur_two_stage/blur_two_stage_4channel.py --out-dir {}/blur_two_stage_4channel'.format(outdir))
#    system('python compiler.py ../apps/blur_one_stage/blur_one_stage_4channel.py --out-dir {}/blur_one_stage_4channel'.format(outdir))

#    system('python compiler.py ../apps/harris_corner/harris_corner.py --out-dir {}/harris_corner'.format(outdir))
    system('python compiler.py ../apps/harris_corner_circle/harris_corner_circle.py --out-dir {}/harris_corner_circle'.format(outdir))

    system('python compiler.py ../apps/raytracer/raytracer.py --out-dir {}/raytracer'.format(outdir))

    system('python compiler.py ../apps/bilateral_grid/bilateral_grid.py --out-dir {}/bilateral_grid'.format(outdir))

    system('python compiler.py ../apps/local_laplacian/local_laplacian.py --out-dir {}/local_laplacian'.format(outdir))

    system('python compiler.py ../apps/camera_pipe/camera_pipe.py --out-dir {}/camera_pipe'.format(outdir))

#    system('python compiler.py --validate {} --out-dir validate_{} --validate-speedups > {}/validate.csv'.format(outdir, outdir, outdir))

    if not args.do_print and not args.check:
    
        system('python bench_stats.py {}'.format(outdir), add_options=False, check=False)

if __name__ == '__main__':
    main()
