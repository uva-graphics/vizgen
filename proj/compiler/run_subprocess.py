
import binascii
import pickle
import hashlib
import sys
import subprocess
import os, os.path
import time
import random

pickle_protocol = 2         # For PyPy2 compatibility, use a Python 2.x compatible pickle protocol

def run_subprocess(output_dir, code_str, python_binary, *args, **kw0):
    """
    Run code_str in a subprocess. This should assign to 'result' variable, which is returned.
    """
    #print(output_dir)
    pickle_protocol_local = pickle_protocol
    parent_sys_path = [os.path.abspath(path) for path in sys.path]
    kw = dict(kw0)
    arg_s = binascii.hexlify(pickle.dumps((args, kw), pickle_protocol))
    runner_filename = os.path.join(output_dir, '_compile' + str(os.getpid()) + '_' + str(time.time()).replace('.','') + '_' + str(random.random()).replace('.','') + '_' + hashlib.md5(code_str.encode('utf-8')).hexdigest() + '.py')
    
    with open(runner_filename, 'wt') as f:
        f.write( ("""
from __future__ import print_function

import pickle, sys, shutil, binascii, time; sys.path += {parent_sys_path}
(args, kw) = pickle.loads(binascii.unhexlify(sys.argv[1]))
{code_str}
print()
print(binascii.hexlify(pickle.dumps(result, {pickle_protocol_local})).decode('utf-8'))
""").format(**locals()))
    cmd ='''%s %s %s''' % (python_binary, "\"" + runner_filename + "\"", arg_s.decode('utf-8'))
    
    ans = {}
    try:
        out_s = subprocess.check_output(cmd, shell=True).decode('utf-8')
#        print('raw out_s:', out_s)
#        print(type(out_s.strip()))
 #       print(type(out_s.strip().split('\n')))
        
        last_line = out_s.strip().split('\n')[-1]
#        print(last_line)
        ans = pickle.loads(binascii.unhexlify(last_line))
        #ans['_raw_out'] = out_s
    except Exception as err:
        raise
        #ans['_raw_out']=str(err)
    try:
        os.remove(runner_filename)
    except:
        pass
    return ans
