import os
import subprocess
import argparse

def get_apps_with_c(app_dir="."):
    """Given the app directory, this returns a list of paths to the c subdir 
    in app folders which have naive c implementations
    """

    dirs_to_check = os.listdir(app_dir)
    c_implementations = []

    for d in dirs_to_check:
        path_to_d = os.path.join(app_dir, d)

        if os.path.isdir(path_to_d):
            path_to_c = os.path.join(path_to_d, "c")

            if os.path.isdir(path_to_c):
                c_implementations.append(path_to_c)

    print("-----\nFound C implementations for the following apps:")

    for c_impl in c_implementations:
        app_name = c_impl.split(os.sep)[-2]
        print("-->", app_name)

    print("-----")
#    print(c_implementations)
#    c_implementations = ['./harris_corner/c']
    
    return c_implementations

def compile_c_implementation(path_to_c_implementation, bits=32):
    """Runs make to compile the c implementation

    If there's an error when compiling, this will return False, otherwise, it
    will return True.
    """
#    print('compile_c_implementation', path_to_c_implementation, bits)
    
    starting_dir = os.getcwd()
    os.chdir(path_to_c_implementation)

    with open('Makefile', 'rt') as f:
        s = f.read()
    
    found = False
    target = '-Dreal=double' if bits == 64 else '-Dreal=float'
    for search in ['-Dreal=double', '-Dreal=float']:
        if search in s:
            s = s.replace(search, target)
            found = True
    if not found:
        if 'CFLAGS=' in s:
            s = s.replace('CFLAGS=', 'CFLAGS=' + target)
        else:
            raise ValueError('Could not find CFLAGS in {} Makefile'.format(path_to_c_implementation))
    with open('Makefile', 'wt') as f:
        f.write(s)
#    print('Edited Makefile:')
#    print(s)

    proc = subprocess.Popen(
        ["make"],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    output, error = proc.communicate()

    os.chdir(starting_dir)

    if "error" in str(error):
        print("Error when compiling %s:" % path_to_c_implementation)
        print("--> Output from make:\n%s" % bytes.decode(output))
        print("--> Error information:\n%s" % bytes.decode(error))
        return False

    return True

def run_c_implementation(path_to_c_implementation, times=10):
    """Runs the c implementation; we assume that it has already been compiled
    correctly.

    Returns the amount of time the program took to run (outputted by the
    program as a float to stdout); if there is a problem with this, returns -1.
    
    Repeat the timing 'times' number of times and take the minimum time overall.
    """
    
    ansL = []
    for i in range(times):
        starting_dir = os.getcwd()
        os.chdir(path_to_c_implementation)

        proc = subprocess.Popen(
            ["./a.out"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE)
        output, error = proc.communicate()

        os.chdir(starting_dir)

        try:
            result = float(bytes.decode(output))
        except ValueError:
            print("Error when attempting to capture timing output from %s" % 
                path_to_c_implementation)
            print("--> Output:\n%s" % bytes.decode(output))
            print("--> Error information:\n%s" % bytes.decode(error))
            result = -1
        ansL.append(result)

    return min(ansL)

def run_benchmark(output_csv_filename="naive_c_stats.csv", bits=32):
    """Runs the whole benchmark
    """
    apps_with_c_paths = get_apps_with_c()
    output_str = "app name,time (seconds)\n"

    for app_path in apps_with_c_paths:
        print("-----")
        app_name = app_path.split(os.sep)[-2]
        print("Compiling: %s (path = \"%s\")" % (app_name, app_path))

        if compile_c_implementation(app_path, bits=bits):
            print("Success!")
            print("Now attempting to run the program...")
            time = run_c_implementation(app_path)
            print("time = %f seconds" % time)

            if time >= 0:
                output_str += "%s,%f\n" % (app_name, time)

        print("-----")

    print("Saving results to: %s" % output_csv_filename)

    with open(output_csv_filename, "w") as f:
        f.write(output_str)

def main():
    parser = argparse.ArgumentParser(description='Run all C program benchmarks.')
    parser.add_argument('bits', type=int, help='Float bits, either 32 or 64')
    parser.add_argument('--csv', dest='csv', default='', help='Output CSV filename, defaults to naive_c_stats_[bits].csv')
    args = parser.parse_args()
    csv = args.csv
    if csv == '':
        csv = 'naive_c_stats_{}.csv'.format(args.bits)
    run_benchmark(output_csv_filename=csv, bits=args.bits)

if __name__ == "__main__":
    main()
