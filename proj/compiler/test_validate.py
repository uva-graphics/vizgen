
import os, os.path
import sys

def system(s):
    print(s)
    return os.system(s)

if not os.path.exists('../images'):
    system('ln -s apps/images ../images')

filename = 'vizgen_validate.zip'
try:
    T0 = os.stat(filename).st_mtime
except FileNotFoundError:
    T0 = 0.0
system('wget -N http://www.cs.virginia.edu/~connelly/project_pages/vizgen/' + filename)
T1 = os.stat(filename).st_mtime

changed = T1 > T0

outdir = 'out_2016_08_23_validate'

if changed:
    system('unzip -f -o ' + filename)

sys.exit(system('python compiler.py --validate ' + outdir))
