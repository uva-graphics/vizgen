
"""
Utility to summarize profiling information by aggregating across all applications.
"""

import sys
import os, os.path
import pprint
import collections
import transforms

def read_profile(subdir_full_path):
    """
    Given full path to a tuner output directory (e.g. 'out/mandelbrot'), return profile dict mapping strs => floats.
    """
    ans = {}
    outfile = os.path.join(subdir_full_path, 'stats', 'output.txt')
#        print(outfile)
    try:
        output = open(outfile, 'rt').read()
        sub = 'Profile times [sec]:'
        i = output.index(sub)
    except:
        raise #continue
    L = output[i + len(sub):].strip().split('\n')
#        print(L)
    for x in L:
        if '(' in x:
            x = x[:x.index('(')]
        x = x.strip()
        subL = x.split(' ')
#            print(x)
#            print(subL)
        key = ' '.join(subL[:-1]).strip().rstrip(':')
        time = float(subL[-1])
        ans[key] = time
    return ans

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: python summarize_profile.py outdir', file=sys.stderr)
        sys.exit(1)
    
    outdir = args[0]
    transforms.do_profile = True
    
#    d = collections.defaultdict(lambda: 0.0)
    for subdir in next(os.walk(outdir))[1]:
        d = read_profile(os.path.join(outdir, subdir))
        for (key, time) in d.items():
            transforms.profile[key] += time

    transforms.profile.close()

        #print(subdir, L)

if __name__ == '__main__':
    main()