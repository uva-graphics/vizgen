
import os, os.path
import subprocess
import numpy

def get_subdirs(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

def main():
    out_filename = 'pythran_stats.csv'

    all_dirs = get_subdirs('.')
    subdirs = sorted([subdir for subdir in all_dirs if os.path.exists(os.path.join(subdir, 'test_pythran.py'))])

    print('Collected {} subdirs:'.format(len(subdirs)))
    for subdir in subdirs:
        print('  ' + subdir)
    print('')
    
#    subdirs = subdirs[:2]
    
    invalid_value = 0.0
    
    timeL = []
    orig_dir = os.getcwd()
    for subdir in subdirs:
        print('-'*80)
        print('Benchmarking {}'.format(subdir))
        print('-'*80)
        print
        os.chdir(orig_dir)
        os.chdir(subdir)
        try:
            process_output = subprocess.check_output('python test_pythran.py', shell=True)
            current_time = invalid_value
            lines = process_output.split('\n')
            for line in lines:
                if line.startswith('Fastest time:'):
                    current_time = float(line.split()[2])
            timeL.append(current_time)
        except:
            timeL.append(invalid_value)
        print('Time list: {}'.format(timeL))
        print('')

    os.chdir(orig_dir)
    assert len(timeL) == len(subdirs)

    with open(out_filename, 'wt') as f:
        f.write('app name, time (seconds)\n')
        for i in range(len(subdirs)):
            f.write('{}, {}\n'.format(subdirs[i], timeL[i]))

    print(open(out_filename, 'rt').read())

if __name__ == '__main__':
    main()
