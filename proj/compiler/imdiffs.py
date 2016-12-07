
import sys
import util
import os.path
import glob

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print('imdiffs.py validate_outdir', file=sys.stderr)
        print('  Computes image differences on a validated output directory.', file=sys.stderr)
        print('', file=sys.stderr)
        print('  validate_outdir is a directory where previously the following command has been run:', file=sys.stderr)
        print('  python compiler.py outdir --validate-images --out-dir validate_outdir', file=sys.stderr)
        sys.exit(1)

    for filename in glob.glob(os.path.join(args[0], '*.png')):
        I = util.read_img(filename)
        ref_filename = os.path.splitext(filename)[0] + '-python.png'
        if os.path.exists(ref_filename):
            I_ref = util.read_img(ref_filename)
            print(os.path.split(filename)[1], util.imdiff(I_ref, I))

if __name__ == '__main__':
    main()
