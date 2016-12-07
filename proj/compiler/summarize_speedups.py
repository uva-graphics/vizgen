
import sys
import os, os.path

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: python summarize_speedups.py outdir')
        sys.exit(1)
    outdir = args[0]
    
    subdirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    for (i, subdir) in enumerate(subdirs):
        start = 0 if i == 0 else 1
        validate_speedups_filename = os.path.join(outdir, subdir, 'stats', 'validate_speedups_after.txt')
        try:
            s = open(validate_speedups_filename, 'rt').read()
        except:
            print('Error for', subdir)
            continue
        s = s.replace('\nApplication', '\n' + subdir)
        print('\n'.join(s.split('\n')[start:]).strip())

if __name__ == '__main__':
    main()

