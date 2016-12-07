
import sys
import glob
import os.path
import json
import argparse_util

import numpy
import util
import summarize_profile

def load_csv_cached(filename='../apps/naive_c_stats.csv', cache={}):
    """
    Get C statistics numpy record list, or return None if the file does not exist.
    """
    if filename in cache:
        return cache[filename]
    if not os.path.exists(filename):
        ans = None
    else:
        ans = numpy.recfromcsv(filename)
    cache[filename] = ans
    return ans

def get_c_stats(filename):
    return load_csv_cached(filename)

def get_pythran_stats():
    return load_csv_cached('../apps/pythran_stats.csv')

def lines(filename, exclude_imports=True, exclude_comments=True, exclude_tests=True, exclude_globals=True, exclude_blank=True, verbose=False, is_c=False, s=None):
    """
    Get the lines of main program logic, excluding various less important information such as imports/comments/tests, and globals (typically used for tests).
    """
    if s is None:
        s = open(filename, 'rt').read()

    L = s.split('\n')
    
    # Hack to strip out triple and single quote string lines in a heuristic (unreliable) way, which avoids parsing Cython
    if not is_c:
        for i in range(len(L)):
            if L[i].strip().startswith("'") and L[i].strip().endswith("'"):
                L[i] = ''
        i = 0
        while i < len(L):
            found = False
            for triple_quote in ['"""', "'''"]:
                if L[i].strip().startswith(triple_quote):
                    L[i] = L[i].strip()[3:]
                    for j in range(i, len(L)):
                        if triple_quote in L[j]:
                            found = True
                        L[j] = ''
                        if found:
                            break
                    i = j+1
            if not found:
                i += 1
    else:
        begin_comment = '/*'
        end_comment = '*/'
        i = 0
        while i < len(L):
            found = False
            if begin_comment in L[i]:
                rest = L[i][L[i].index(begin_comment)+len(begin_comment):]
                L[i] = L[i][:L[i].index(begin_comment)]
                if end_comment in rest:
                    found = True
                    i += 1
                else:
                    for j in range(i+1, len(L)):
                        if end_comment in L[j]:
                            found = True
                            L[j] = L[j][L[j].index(end_comment)+len(end_comment):]
                        else:
                            L[j] = ''
                        if found:
                            break
                    i = j + 1
            if not found:
                i += 1

#    util.print_header('Lines before exclude_tests:' + filename, '\n'.join(L))

    # Hack to strip out def test() and other methods in a heuristic (unreliable) way, which avoids parsing Cython
    if exclude_tests:
        # Also exclude makeColorMatrix so that our camera pipe is apples-to-apples comparable with reported lines in Halide paper
        if not is_c:
            methods = 'test run_test_all mandelbrot_gray mandelbrot_color composite_numpy composite_numexpr makeColorMatrix'.split()
        else:
            methods = ['int main', 'void main']
        i = 0
        while i < len(L):
            L_i_strip = L[i].strip()
            if ((not is_c and (any(L_i_strip.startswith('def ' + method) for method in methods) or
                              any(L_i_strip.startswith('cdef ' + method) for method in methods))) or
                (is_c and (any(L_i_strip.startswith(method) for method in methods)))):
                L[i] = ''
                for j in range(i+1, len(L)):
                    L_j_strip = L[j].strip()
                    c_ok = True
                    if is_c:
                        c_ok = L_j_strip != '{' and L_j_strip != '}'
                    if not L[j].startswith(' ') and not L[j].startswith('\t') and not len(L[j].strip()) == 0 and c_ok:
                        break
                    else:
                        L[j] = ''
                i = j
            elif (L[i].strip().startswith('test(') or L[i].strip().startswith('run_test_all(')) and not is_c:
                L[i] = ''
                i += 1
            else:
                i += 1

#    util.print_header('Lines before exclude_imports:' + filename, '\n'.join(L))
    if exclude_imports:
        if not is_c:
            L = [x for x in L if not x.lstrip().startswith('import') and not x.lstrip().startswith('cimport') and not x.startswith('cdef extern')]
        else:
            L = [x for x in L if not x.lstrip().startswith('#include')]
#    util.print_header('Lines before exclude_comments:' + filename, '\n'.join(L))
    if exclude_comments:
        if not is_c:
            L = [x for x in L if not x.lstrip().startswith('#') and not x.strip() == 'pass']
        else:
            L = [x for x in L if not x.lstrip().startswith('//')]
#    util.print_header('Lines before exclude_globals:' + filename, '\n'.join(L))
    if exclude_globals and not is_c:
        L = [x for x in L if (x.startswith(' ') or x.startswith('\t') or x.startswith('def') or x.startswith('cdef')) and (not x.lstrip().startswith('has_'))]
#    util.print_header('Lines before exclude_blank:' + filename, '\n'.join(L))

    if is_c:
        # Also exclude makeColorMatrix so that C camera pipe is apples-to-apples comparable with reported lines in Halide paper
        L = [x for x in L if not x.lstrip().startswith('matrix_3200') and not x.lstrip().startswith('matrix_7000')]
    if exclude_blank:
        L = [x for x in L if not len(x.strip()) == 0]

    if verbose:
        util.print_header('Final lines for:' + filename, '\n'.join(L))

    return len(L)

def main():
    parser = argparse_util.ArgumentParser(description='Calculate statistics table from tuner runs.')
    parser.add_argument('outdir', help='Loops through all subdirs in out and prints final speedup for each subdir')
    parser.add_argument('--csv', action='store_false', dest='latex', help='Output old CSV format')
    parser.add_argument('--transforms-used', action='store_true', dest='transforms_used', help='Output transforms used table in Latex format')
    parser.add_argument('--names', action='store_true', dest='names', help='Use longer and more descriptive names for applications')
    parser.add_argument('--safe-dir', dest='safe_dir', help='Use tuner directory that was run with --safe for the non-approximated ours times')
    parser.add_argument('--alt-times', dest='alt_times', help='Source from backup tuner output directory if times are missing in the table (can be a comma-separated list)')
    parser.add_argument('--cstats', dest='cstats', help='Use naive C run-time file with .csv extension')
    parser.add_argument('--changed', dest='changed', action='store_true', help='Wrap all Latex table entries in \\changed{} macro')
    parser.add_argument('--no-safe', dest='safe', action='store_false', help='Remove comparison with non-approximated (--safe mode) results')
#    parser.add_argument('--unpython', action='store_true', dest='unpython', help='Include unPython in comparison table')
    parser.set_defaults(changed=False)
    parser.set_defaults(latex=True)
    parser.set_defaults(pythran=True)
    parser.set_defaults(transforms_used=False)
    parser.set_defaults(names=False)
    parser.set_defaults(safe=True)
    parser.set_defaults(safe_dir='out_2016_08_23_float64')
    parser.set_defaults(alt_times='out_january')
    parser.set_defaults(cstats='../apps/naive_c_stats.csv')
#    parser.set_defaults(unpython=True)
    
    args = parser.parse_args()
    outdir = args.outdir
    latex = args.latex
    transforms_used = args.transforms_used
    unpython = True #args.unpython
    
    log = open(os.path.join(outdir, 'stats.csv'), 'wt')

    def print_log(s):
        print(s)
        print(s, file=log)

    def read_float(filename, dirname=None):
        if dirname is None:
            dirname = subdir
        ans = 0.0
        filename = os.path.join(dirname, filename)
        if os.path.exists(filename):
            ans = float(open(filename, 'rt').read())
        return ans

    subdirL = []

    for subdir in glob.glob(os.path.join(outdir, '*')):
        if os.path.isdir(subdir):
            if len(glob.glob(os.path.join(subdir, 'stats/*_speedup.txt'))):
                subdirL.append(subdir)

    # Map our benchmark name to name used in the paper
    name_aliases = dict(keyvalue.split(' ', 1) for keyvalue in """
bilateral_grid_clean Bilateral grid
blur_one_stage_4channel One stage blur (RGBA)
blur_one_stage_rgb One stage blur (RGB)
blur_one_stage_gray One stage blur (gray)
blur_two_stage_4channel Two stage blur (RGBA)
blur_two_stage_rgb Two stage blur (RGB)
blur_two_stage_gray Two stage blur (gray)
camera_pipe_fcam Camera pipeline
composite_gray Composite (gray)
composite_4channel Composite (RGBA)
composite_rgb Composite (RGB)
harris_corner Harris corner
harris_corner_circle Harris corner
interpolate Interpolate
local_laplacian_fuse Local Laplacian
mandelbrot_animate Mandelbrot
optical_flow_one_module Optical flow
optical_flow_patchmatch Optical flow
pacman_clean Pac-Man
raytracer_short_simplified_animate Raytracer
""".strip().split('\n'))

    # Map our benchmark name to name used in C comparison script benchmark_naive_c.py
    name_ours_to_c = dict(keyvalue.split() for keyvalue in """
bilateral_grid_clean bilateral_grid
blur_one_stage_4channel blur_one_stage
blur_one_stage_gray blur_one_stage_gray
blur_one_stage_rgb blur_one_stage
blur_two_stage_4channel blur_two_stage
blur_two_stage_rgb blur_two_stage
blur_two_stage_gray blur_two_stage_gray
camera_pipe_fcam camera_pipe
composite_gray composite_gray
composite_4channel composite
composite_rgb composite
harris_corner harris_corner
harris_corner_circle harris_corner_circle
interpolate interpolate
local_laplacian_fuse local_laplacian
mandelbrot_animate mandelbrot
optical_flow_one_module optical_flow
optical_flow_patchmatch optical_flow_patchmatch
pacman_clean pacman
raytracer_short_simplified_animate raytracer
""".strip().split('\n'))

    # Map our benchmark name to C filename (without .c prefix) in /c/ subdirectory of app directory.
    name_ours_to_c_filename = dict(keyvalue.split() for keyvalue in """
bilateral_grid_clean bilateral_grid.c
blur_one_stage_4channel blur_one_stage.c
blur_one_stage_rgb blur_one_stage.c
blur_one_stage_gray blur_one_stage.c
blur_two_stage_4channel blur_two_stage.c
blur_two_stage_rgb blur_two_stage.c
blur_two_stage_gray blur_two_stage.c
camera_pipe_fcam camera_pipe.c
composite_gray composite.c
composite_4channel composite.c
composite_rgb composite.c
harris_corner harris_corner.c
harris_corner_circle harris_corner.c
interpolate interpolate.c
local_laplacian_fuse local_laplacian.c
mandelbrot_animate mandelbrot.c
optical_flow_one_module optical_flow.c,color_wheel.h,draw_line.h
optical_flow_patchmatch optical_flow_patchmatch.c
pacman_clean pacman.c
raytracer_short_simplified_animate raytracer_short_simplified.c
""".strip().split('\n'))
    #if latex or transforms_used:
    subdirL = sorted(subdirL, key=lambda subdir: name_aliases.get(os.path.split(subdir)[1], os.path.split(subdir)[1]).lower())

    stats_dict = dict(keyvalue.split(' ', 1) for keyvalue in """
ApplyMacros API call, rewriting
ArrayStorage Array storage, alteration
LoopImplicit Loop over, implicit variables
LoopRemoveConditionals Remove loop, conditionals
Parallel Parallelize, loop
Preallocate Preallocate, arrays
TypeSpecialize Type, specialization
VectorizeInnermost Vectorize, innermost variable
""".strip().split('\n'))

    statsL = []
    stats_keys = set()

    for subdir in subdirL:
        stats_filename = os.path.join(subdir, 'stats/final_transform_stats.txt')
        if os.path.exists(stats_filename):
            statsL.append(json.loads(open(stats_filename, 'rt').read()))
            stats_keys |= set(statsL[-1].keys())
        else:
            statsL.append({})
    stats_keys = sorted(stats_keys)
    stats_count = len(stats_keys)
    stats_names = []

    if transforms_used:
        stats_keys.sort(key=lambda stats_key: stats_dict[stats_key])

    for stats_key in stats_keys:
        stats_names.append(stats_dict[stats_key])


    def print_our_header():
        if not csv:
            print(r'% Tabular region automatically generated by "bench_stats.py ' + ' '.join(sys.argv[1:]) + '". Do not hand edit!')

    amp = '&'
    csv = False
    if not latex:
        csv = True
        amp = ','
    latex = True

    begin_changed = end_changed = ''
    if args.changed:
        begin_changed = '\\changed{'
        end_changed = '}'

    def remove_latex(s):
        if csv:
            return s.replace(r'\textbf{', '').replace(r'\multicolumn{1}{|c|}{', '').replace(r'\multicolumn{7}{|c|}{', '').replace(r'\multicolumn{2}{|c|}{', '').replace(r'\multirow{2}{*}{', '').replace(r' \cline{3-9} \cline{11-12}', '').replace(r'\\', '').replace(r'\hline', '').replace('{', '').replace('}', '').replace(r'\X', '')
        else:
            return s

    if not latex and not transforms_used:
        print_log('Application,Our Speedup,Numba JIT Speedup,PyPy JIT Speedup,unPython* Speedup,Ours Shorter,Python Time,Ours Time,Numba Time,PyPy Time,unPython Time,Pythran Time,Tune Iters,Tune Time [min],Profiling Count,Profiling Time [min],Our Codegen Time [min],C Compiler Time [min],' + ','.join('Has' + key for key in stats_keys))
    elif latex:
        print_our_header()
        if not csv:
            print(r'\begin{tabular}{|l|r|r|r|r|r|r|r|r|' + ('r|' if args.safe else '') + ('r|' if unpython else '') + 'r|}')
            print(r'\hline')
        header = remove_latex(r"""
 & \multicolumn{1}{|c|}{ \textbf{Ours} }  & \multicolumn{7}{|c|}{ \textbf{Ours Speedup vs} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{2}{|c|}{ \textbf{Ours Shorter vs} } \\
 \cline{3-9} \cline{11-12}
\multirow{2}{*}{\textbf{Application}}  & \multicolumn{1}{|c|}{ \textbf{Approx.} } & \multicolumn{1}{|c|}{ \textbf{} } &  \multicolumn{1}{|c|}{ \textbf{Ours} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{1}{|c|}{ \textbf{Ours} } & \multicolumn{1}{|c|}{ \textbf{} } & \multicolumn{1}{|c|}{ \textbf{} } \\
 &  \multicolumn{1}{|c|}{\textbf{Time}} & \multicolumn{1}{|c|}{\textbf{Python}} & \multicolumn{1}{|c|}{\textbf{Non-}} & \multicolumn{1}{|c|}{\textbf{Numba}} & \multicolumn{1}{|c|}{\textbf{PyPy}} & \multicolumn{1}{|c|}{\textbf{Pythran}} & \multicolumn{1}{|c|}{\textbf{unPython*}} & \multicolumn{1}{|c|}{\textbf{C code}} & \multicolumn{1}{|c|}{\textbf{Lines}} & \multicolumn{1}{|c|}{\textbf{vs C}} & \multicolumn{1}{|c|}{\textbf{vs Cython}} \\
 &  \multicolumn{1}{|c|}{\textbf{[ms]}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{approx.}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} & \multicolumn{1}{|c|}{\textbf{}} \\ \hline
""".replace('&', amp))
        if not args.safe:
            header = header.replace(r'\multicolumn{7}{|c|}{', r'\multicolumn{6}{|c|}{').replace(r'&  \multicolumn{1}{|c|}{ \textbf{Ours} }', r'').replace(r'& \multicolumn{1}{|c|}{\textbf{Non-}} ', '').replace(r'& \multicolumn{1}{|c|}{\textbf{approx.}}', '').replace('Approx.', 'Ours').replace(r'\cline{3-9} \cline{11-12}', r'\cline{3-8} \cline{10-11}').replace(r'& \multicolumn{1}{|c|}{ \textbf{Ours} }  &', '& &')
        print(header)
    elif transforms_used:
        print_our_header()
        print(r'\begin{tabular}{|l' + '|c' * stats_count + '|}')
        print(r'\hline')
        print(r'Application & ' + ' & '.join([r'\multicolumn{1}{|c|}{' + stats_name.split(',')[0] + '}' for stats_name in stats_names]) + r'\\')# \hline')
        print(r' & ' +  ' & '.join([r'\multicolumn{1}{|c|}{' + stats_name.split(',')[1] + '}' for stats_name in stats_names]) + r'\\ \hline')
#        sys.exit(1)
#        for rowstr in [r'Application & Ours & Ours Speedup & Ours Speedup & Ours Speedup & ' + ('Ours Speedup & ' if unpython else '') + r'Ours Speedup & Ours Lines & Ours Shorter & Ours Shorter\\',
#                       r'            & Time [ms] & vs Python    & vs Numba JIT & vs PyPy JIT  & ' + ('vs unPython & ' if unpython else '') + r'vs naive C   & of Code    & vs C         & vs Cython\\ \hline']:
#            rowstr = rowstr.replace('&', '} & \multicolumn{1}{|c|}{')
#            rowstr = rowstr.replace(r'\\', r'} \\')
#            i = rowstr.index('}')
#            rowstr = rowstr[:i] + rowstr[i+1:]
#            print(rowstr)

    speedupL = []
    ours_vs_numbaL = []
    ours_vs_pypyL = []
    ours_vs_unpythonL = []
    ours_vs_naive_cL = []
    ours_vs_pythranL = []
    ours_vs_safeL = []
    shorterL = []
    c_shorterL = []

    for (i, subdir) in enumerate(subdirL):
        stats = statsL[i]
        subdir_only = os.path.split(subdir)[1]
    
        if latex and subdir_only in ['bilateral_grid_clean_small', 'blur_one_stage_4channel', 'blur_two_stage_4channel', 'composite_4channel', 'optical_flow_one_module', 'harris_corner']:
            continue
    
        speedup          = read_float('stats/final_speedup.txt')
        numba_speedup    = read_float('stats/numba_speedup.txt')
        pypy_speedup     = read_float('stats/pypy_speedup.txt')
        unpython_speedup = read_float('stats/unpython_speedup.txt')
        
        py_time = read_float('stats/final_py_time.txt')
        ours_time = read_float('stats/final_ours_time.txt')
        numba_time = read_float('stats/numba_time.txt')
        pypy_time = read_float('stats/pypy_time.txt')
        unpython_time = read_float('stats/unpython_time.txt')
        tune_time = read_float('stats/tune_time.txt')
        run_time = read_float('stats/run_time.txt')
        run_count = int(read_float('stats/run_count.txt'))

        ours_safe_time = read_float('stats/final_ours_time.txt', os.path.join(args.safe_dir, subdir_only))
#        print(subdir_only, numba_time)

        alt_timesL = args.alt_times.split(',')
        for alt_times in alt_timesL:
            if py_time == 0.0:
                py_time = read_float('stats/final_py_time.txt', os.path.join(alt_times, subdir_only))
                if py_time == 0.0:
                    raise ValueError('missing py_time for', subdir_only)

            if numba_time == 0.0:
                numba_time = read_float('stats/numba_time.txt', os.path.join(alt_times, subdir_only))
    #            if numba_time == 0.0:
    #                raise ValueError('missing numba_time for', subdir_only)
    #        print('=>', subdir_only, numba_time)

            if pypy_time == 0.0:
                pypy_time = read_float('stats/pypy_time.txt', os.path.join(alt_times, subdir_only))
    #            if pypy_time == 0.0:
    #                raise ValueError

            if unpython_time == 0.0:
                unpython_time = read_float('stats/unpython_time.txt', os.path.join(alt_times, subdir_only))
    #            if unpython_time == 0.0:
    #                raise ValueError('missing unpython_time for', subdir_only)

        headers = os.path.join(subdir, 'final', 'cython_headers.pyx')
        if os.path.exists(headers):
            program_python = os.path.join(subdir, 'final', 'program.py')
            program_cython = os.path.join(subdir, 'final', 'program.pyx')
            lines_ours = lines(program_python)
            shorter = (lines(program_cython) - lines(headers, exclude_imports=True)) * 1.0 / lines_ours
        else:
            lines_ours = 0
            shorter = 0.0
        
        tune_iters = 0
        gen_subdir = os.path.join(subdir, 'gen')
        for program_filename in glob.glob(os.path.join(gen_subdir, 'program*.pyx')):
            filename_only = os.path.split(program_filename)[1]
            gen_str = filename_only.strip('program.pyx')
            gen = int(gen_str)
            tune_iters = max(tune_iters, gen)

        speedup          = py_time / ours_time
        ours_vs_numba    = numba_time / ours_time
        ours_vs_pypy     = pypy_time / ours_time
        ours_vs_unpython = unpython_time / ours_time
        ours_vs_safe     = ours_safe_time / ours_time
#        print(subdir, ours_time*1000.0)

        def get_time_from_stats(stats):
            ret_time = 0.0
            for row in stats:
                app_name_strip = row['app_name'].decode('ascii').strip()
                if app_name_strip == c_appname:
                    found = True
                    ret_time = row['time_seconds']
            return ret_time

        c_appname = subdir_only
        if c_appname.endswith('_float32'):
            c_appname = c_appname[:-len('_float32')]
        subdir_only_strip = c_appname
        c_appname = name_ours_to_c.get(c_appname, c_appname)

        ours_vs_pythran = 0.0
        pythran_time = 0.0
        if args.pythran:
            pythran_stats = get_pythran_stats()
            pythran_time = get_time_from_stats(pythran_stats)
            ours_vs_pythran = pythran_time / ours_time

        def format_speedup(val, method='', problem=None, fmt=None):
            if csv:
                return str(val)
            if isinstance(val, str):
                return begin_changed + val + end_changed
            for digits_after in range(2):
                if fmt is None:
                    fmt_p = '{:8.' + str(digits_after) + 'f}'
                else:
                    fmt_p = fmt
                numfmt = (fmt_p).format(val)
#                print(digits_after, numfmt)
                if len(numfmt.strip()) == 1 and fmt is None:
                    continue
                return begin_changed + remove_latex('{}\X'.format(numfmt) if val else ('Error' if problem is None else problem)) + end_changed
#                return '{}\X'.format(numfmt) if val else (method + ' Error' if problem is None else problem)

        descriptive_name = name_aliases.get(subdir_only, subdir_only)

        try:
            profile = summarize_profile.read_profile(subdir)
        except:
            profile = {}
        profile_tune_time = profile.get('ProgramInfo: tune', 0.0)
        profile_compile_run_time = profile.get('run_code', 0.0)
        profile_compile_time = profile.get('run_code: compile', 0.0)
        our_codegen_time = profile_tune_time - profile_compile_run_time
        c_compiler_time = profile_compile_time
        
        if not latex and not transforms_used:
            print_log('{:30s}{:8.3f},{:8.3f},{:8.3f},{:8.3f},{:8.3f},{:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f},{:6d},{:6.1f},{:6d},{:6.3f},{:6.3f},{:6.3f},'.format((subdir_only if not args.names else descriptive_name) + ',', speedup, numba_speedup, pypy_speedup, unpython_speedup, shorter, py_time, ours_time, numba_time, pypy_time, unpython_time, pythran_time, tune_iters, tune_time/60.0, run_count, run_time/60.0, our_codegen_time/60.0, c_compiler_time/60.0) + ','.join(str(int(bool(stats.get(key,0)))) for key in stats_keys))
        elif transforms_used:
            print(descriptive_name + ' & ' + ' & '.join(r'\checkmark' if stats.get(key, 0) else '' for key in stats_keys) + r'\\')
        else:
            c_stats = get_c_stats(args.cstats)
#            print(c_appname)
#            print(c_stats)
            c_time = get_time_from_stats(c_stats)
#            if c_time == 0.0:
#                raise ValueError
#            print()
            ours_vs_naive_c = c_time / ours_time
            descriptive_name = name_aliases.get(subdir_only, subdir_only)

            c_lines = 0
            c_sub_appnameL = name_ours_to_c_filename.get(subdir_only_strip, c_appname)
            c_filenameL = []
            for c_sub_appname in c_sub_appnameL.split(','):
                c_filename = os.path.join('../apps/', c_appname, 'c', c_sub_appname)
                c_filenameL.append(c_filename)
                if os.path.exists(c_filename):
                    c_lines += lines(c_filename, verbose=False, is_c=True)
            if c_lines == 0:
                raise ValueError('missing C comparison file(s):', c_sub_appnameL, c_filenameL)
            c_shorter = c_lines * 1.0 / lines_ours
#            print(c_appname, c_filename, os.path.exists(c_filename))

            if ours_vs_numba:
                ours_vs_numbaL.append(ours_vs_numba)
            if ours_vs_pypy:
                ours_vs_pypyL.append(ours_vs_pypy)
            if ours_vs_unpython:
                ours_vs_unpythonL.append(ours_vs_unpython)
            if ours_vs_naive_c:
                ours_vs_naive_cL.append(ours_vs_naive_c)
            if ours_vs_pythran:
                ours_vs_pythranL.append(ours_vs_pythran)
            if ours_vs_safe:
                ours_vs_safeL.append(ours_vs_safe)
            if shorter:
                shorterL.append(shorter)
            if c_shorter:
                c_shorterL.append(c_shorter)

            def print_formatted_row(print_name):
#                print(shorter, type(shorter))
                time_fmt = '{:8.1f}'
                if csv:
                    time_fmt = '{}'
                if args.changed:
                    time_fmt_changed = begin_changed.replace('{', '{{') + time_fmt + end_changed.replace('}', '}}')
                else:
                    time_fmt_changed = time_fmt
                format_tuple = (print_name, (time_fmt_changed.format(ours_time*1000.0) if ours_time else ''), format_speedup(speedup, 'Ours')) + ((format_speedup(ours_vs_safe, 'Ours Safe'),) if args.safe else ()) + (format_speedup(ours_vs_numba, 'Numba'), format_speedup(ours_vs_pypy, 'PyPy'), format_speedup(ours_vs_pythran, 'Pythran')) + ((format_speedup(ours_vs_unpython, 'unPython'),) if unpython else ()) + (format_speedup(ours_vs_naive_c, 'Naive C', r'\textemdash'), begin_changed + str(lines_ours) + end_changed, format_speedup(c_shorter, fmt=time_fmt, problem=r'\textemdash'), shorter)
                print(remove_latex(('{:30s} & ' + '{} & ' * (6 + int(args.safe) + int(unpython)) + r'{} & ' + r'{} & ' + time_fmt_changed + r'\X \\').format(*format_tuple).replace('&', amp)))
            print_formatted_row(descriptive_name)

    if latex:
        if not csv:
            print(r'\hline')
        else:
            print()
        ours_vs_numba = numpy.median(ours_vs_numbaL)
        ours_vs_pypy = numpy.median(ours_vs_pypyL)
        ours_vs_unpython = numpy.median(ours_vs_unpythonL)
        ours_vs_naive_c = numpy.median(ours_vs_naive_cL)
        ours_vs_pythran = numpy.median(ours_vs_pythranL)
        ours_vs_safe = numpy.median(ours_vs_safeL)
        shorter = numpy.median(shorterL)
        c_shorter = numpy.median(c_shorterL)
        ours_time = 0.0
#                print(shorter, shorterL)
        lines_ours = ''
        print_formatted_row(remove_latex(r'\textbf{Median}'))

    if latex or transforms_used:
        if not csv:
            print(r'\hline')
            print(r'\end{tabular}')
            print(r'% End automatically generated tabular region')

if __name__ == '__main__':
    main()
