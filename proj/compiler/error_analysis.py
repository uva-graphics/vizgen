
import glob
import os.path
import argparse_util
import util
import hashlib
import shutil
import importlib
import sys
import traceback
import numpy

def main():
    import compiler

    parser = argparse_util.ArgumentParser(description='Perform error analysis for each application with respect to the original ground truth output.')
    parser.add_argument('tune_dir', help='Analyzes each application in the tune directory and prints error.')
    parser.add_argument('--apps', dest='apps', help='List of app subdirectories separated by commas, e.g. --apps a,b')
    parser.add_argument('--save', dest='save', help='Save output to filenames starting with given prefix')
    parser.set_defaults(save='')
    parser.set_defaults(apps='')
    args = parser.parse_args()
    if len(args.save):
        args.save = os.path.abspath(args.save)
    
    app_error = {}
    
    def print_summary():
        print()
        print('Application, Error')
        for app_short in sorted(app_error.keys()):
            print(app_short + ', ' + str(app_error[app_short]))
        print()
    if len(args.apps):
        appL = args.apps.split(',')
    else:
        appL = []
    
    for subdir in glob.glob(os.path.join(args.tune_dir, '*')): #[::-1][:7]:
        if len(appL):
            subdir_app = os.path.split(subdir)[1]
            if subdir_app not in appL:
                continue
        subdir = os.path.abspath(subdir)
        try:
            #if 'pacman' not in subdir and 'raytracer' not in subdir and 'optical_flow' not in subdir:
            #    continue
            filename_locator = os.path.join(subdir, 'stats', 'tune_filename.txt')
            if not os.path.exists(filename_locator):
                continue

            print()
            print('-'*70)
            print(subdir)
            print('-'*70)

            tuned_pyx = os.path.join(subdir, 'final', 'program.pyx')
            if not os.path.exists(tuned_pyx):
                print(' * Application has no final tuner output, skipping')

            app_filename = open(filename_locator, 'rt').read()
            app_path = os.path.split(app_filename)[0]
            orig_dir = os.getcwd()
            os.chdir(app_path)
            try:
                source = open(app_filename, 'rt').read()
                d = {}
                exec(source, d, d)
                if 'test' not in d:
                    print(' * Application has no test method, skipping')
                    continue
                util.is_initial_run = True
                res = d['test']()
                if 'output' not in res:
                    print(' * Application has test method but no output, skipping')
                    continue
                ground_truth = res['output']
                print(' * Ground truth shape:', ground_truth.shape)

                tuned_path = os.path.split(tuned_pyx)[0]
                os.chdir(tuned_path)
                program_str = subdir + '_' + open('program.pyx', 'rt').read()
                hash = hashlib.md5(program_str.encode('utf-8')).hexdigest()
                program_filename = 'error_analysis_program_{}'.format(hash)
                shutil.copyfile('program.pyx', os.path.join(orig_dir, program_filename + '.pyx'))
                os.chdir(orig_dir)
                util.compile_cython_single_file(program_filename, compiler.c_edit_func)

                d_tuned = {}
                sys.path.append(tuned_path)
                os.chdir(app_path)
                importlib.invalidate_caches()           # If we do not clear the caches then module will sometimes not be found
                #exec('import ' + program_filename, d_tuned, d_tuned)
                d_tuned = importlib.import_module(program_filename)
                res_tuned = d_tuned.test()
                if 'output' not in res_tuned:
                    print(' * Tuned application has test method but no output, skipping')
                    continue
                tuned_image = res_tuned['output']
                print(' * Tuned image shape:', tuned_image.shape)

                if len(args.save):
                    numpy.save(args.save + '_ground.npy', ground_truth)
                    numpy.save(args.save + '_ours.npy', tuned_image)

                try:
                    diff = util.imdiff(ground_truth, tuned_image)
                except:
                    print(' * Error in computing diff, skipping')
                    continue
                print(' * Error:', diff)
                app_short = os.path.split(subdir)[1]
                app_error[app_short] = diff
                
                print_summary()
            #print(app_path, 'test' in d)
            #print(app_filename)
            finally:
                os.chdir(orig_dir)
        except:
            traceback.print_exc()
            print(' * Unhandled exception, skipping')
            
    print_summary()

if __name__ == '__main__':
    main()
