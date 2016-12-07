
from __future__ import print_function

import math
import sys
import platform
import time
import numpy
import os
import inspect
import subprocess
import io
import string
import copy
import functools
import warnings
warnings.simplefilter('ignore', UserWarning)    # Suppress UserWarning from Cython related to https://github.com/cython/cython/issues/1509

from distutils.core import setup
from distutils.extension import Extension

is_pypy = False
if platform.python_implementation().lower() == 'pypy':
    # PyPy does not support skimage, so use Pillow instead for speed comparisons with PyPy
    from PIL import Image
    is_pypy = True
else:
    import skimage
    import skimage.color
    import skimage.io
    from Cython.Distutils import build_ext

default_test_dtype = None

cython_type_check = True                # Enable debugging checks in CythonType
track_shape_list = False
util_verbose = False
is_initial_run = False                  # Whether the initial (ground truth) test run is being done. The compiler sets this global variable.
override_input_image = None             # Override input images passed to test_image_pipeline_filename() with this filename
override_n_tests = None                 # Override number of test runs for test_image_pipeline_filename() with this integer
override_output_image = None            # Override output image stored by test_image_pipeline_filename()
use_4channel = False                    # Turned on by ArrayStorage transform with use_4channel set to True. Causes 3 channel images to convert to 4 channel
                                        # when read and 4 channel images to convert to 3 channel when written.
nrepeats = 1                            # Repeat the experiment of finding minimum time over n_tests runs 'nrepeats' times for more accuracy.
                                        # Should be set to 1 for checked-in code.

types_non_variable_prefix = '_nonvar_'      # Special prefix for non-variable name returned by compiler.get_types()

DEFAULT_THRESHOLD = 1e-2        # Default image difference threshold
DEFAULT_DYNAMIC_MAX_TESTS = 10
DEFAULT_DYNAMIC_MAX_TIME  = 1.0

def is_test_funcname(funcname):
    """
    Returns True if the given function name str funcname is a test function.
    """
    return funcname.startswith('test')

def overwrite_line(line):
    """Clears the current line on the console and overwrites it with line.
    """

    sys.stdout.write("\r")
    sys.stdout.write(str(line))
    sys.stdout.flush()

def read_img(in_filename, grayscale=False, extra_info={}):
    """Returns the image saved at in_filename as a numpy array.
    
    If grayscale is True, converts from 3D RGB image to 2D grayscale image.
    """
    if is_pypy:
        ans = Image.open(in_filename)
        height = ans.height
        width = ans.width
        channels = len(ans.getbands())
        if ans.mode == 'I':
            numpy_mode = 'uint32'
            maxval = 65535.0
        elif ans.mode in ['L', 'RGB', 'RGBA']:
            numpy_mode = 'uint8'
            maxval = 255.0
        else:
            raise ValueError('unknown mode')
        ans = numpy.fromstring(ans.tobytes(), numpy_mode).reshape((height, width, channels))
        ans = ans/maxval
        if grayscale and (len(ans.shape) == 3 and ans.shape[2] == 3):
            ans = ans[:,:,0]*0.2125 + ans[:,:,1]*0.7154 + ans[:,:,2]*0.0721
        if len(ans.shape) == 3 and ans.shape[2] == 1:
            ans = ans[:,:,0]
        return ans
    else:
        ans = skimage.io.imread(in_filename)
        if ans.dtype == numpy.int32:    # Work around scikit-image bug #1680
            ans = numpy.asarray(ans, numpy.uint16)
        ans = skimage.img_as_float(ans)
        if grayscale:
            ans = skimage.color.rgb2gray(ans)
#        print('here', use_4channel, len(ans.shape) == 3, ans.shape[2] == 3)
        if use_4channel and len(ans.shape) == 3 and ans.shape[2] == 3:
            ans = numpy.dstack((ans,) + (numpy.ones((ans.shape[0], ans.shape[1], 1)),))
            extra_info['originally_3channel'] = True
    return ans

def write_img(out_img, out_filename, do_clip=True):
    """Writes out_img to out_filename
    """
    if use_4channel and len(out_img.shape) == 3 and out_img.shape[2] == 4:
        out_img = out_img[:,:,:3]
    
    assert out_img is not None, 'expected out_img to not be None'
    out_img = numpy.clip(out_img, 0, 1) if do_clip else out_img
    if is_pypy:
        out_img = numpy.asarray(out_img*255, 'uint8')
        if len(out_img.shape) == 2:
            mode = 'L'
        elif len(out_img.shape) == 3:
            if out_img.shape[2] == 3:
                mode = 'RGB'
            elif out_img.shape[2] == 4:
                mode = 'RGBA'
            else:
                raise ValueError('unknown color image mode')
        else:
            raise ValueError('unknown number of dimensions for image')
            
        I = Image.frombytes(mode, (out_img.shape[1], out_img.shape[0]), out_img.tobytes())
        I.save(out_filename)
    else:
        try:
            skimage.io.imsave(out_filename, out_img)
        except:
            print('Caught exception while saving to {}: image shape is {}, min: {}, max: {}'.format(out_filename, out_img.shape, out_img.min(), out_img.max()))
            raise

def imdiff(golden_img, test_img, ignore_last_element=False, originally_3channel=False):
    """Diffs the two images pixel-by-pixel

    Compares the golden image vs. the test image pixel-by-pixel, and returns mean Euclidean distance between RGB colors.

    Args:
        golden_img, np.ndarray, represents the ground truth image to Compare
            against
        test_img, np.ndarray, represents the test image to compare to the
            golden image
        diff_thresh, float, represents the percentage (i.e. if 10%, then 0.1)
            difference allowed for any pixel
        originally_3channel, bool, if True then discards any 4th channel before doing comparison
    Returns:
        result, float
    """

    if len(golden_img.shape) == 2:
        return numpy.mean(numpy.abs(golden_img - test_img))
    elif ignore_last_element:
        return numpy.mean(numpy.abs(numpy.sqrt(numpy.sum((golden_img[:, :, :3]-test_img[:, :, :3])**2, 2))))
    elif len(golden_img.shape) == 3:
        if len(test_img.shape) == 2:
            test_img = numpy.dstack((test_img,))
        if use_4channel and originally_3channel:
            if golden_img.shape[2] == 4:
                golden_img = golden_img[:,:,:3]
            if test_img.shape[2] == 4:
                test_img = test_img[:,:,:3]
        
        # TODO: Actually fix whatever bugs cause the sizes to mismatch
        if len(golden_img.shape) == 3 and len(test_img.shape) == 3 and set([golden_img.shape[2], test_img.shape[2]]) == set([3, 4]):
            golden_img = golden_img[:,:,:3]
            test_img = test_img[:,:,:3]
        return numpy.mean(numpy.abs(numpy.sqrt(numpy.sum((golden_img-test_img)**2, 2))))
    else:
        raise ValueError('bad shape: %r' % golden_img.shape)

def write_vectorization_header_file(path):
    """Writes the vectorization header file to path
    """

    with open(path, 'wt') as f:
        f.write("""
            typedef float v4sf __attribute__((vector_size(16)));
            typedef float v2sf __attribute__((vector_size(8)));
            """)

def cast_args(input_imgL, dtype):
    return [(numpy.asarray(x, dtype) if isinstance(x, numpy.ndarray) else x) for x in input_imgL]

def test_image_pipeline(image_func, input_imgL, n, ground_truth_output=None, verbose=True, threshold=DEFAULT_THRESHOLD, name=None, use_output_img=False, dynamic_max_tests=DEFAULT_DYNAMIC_MAX_TESTS, dynamic_max_time=DEFAULT_DYNAMIC_MAX_TIME, imdiff_ignore_last_element=True, dtype=None, additional_args=None, additional_kw_args=None, allow_kwargs=True, output_img_shape=None, originally_3channel=False, output_gain=1.0, output_bias=0.0, ground_truth_filename=''):
    """
    Times multiple runs of image_func(*input_imageL), which accepts zero or more images as argument and returns a single image.

    Args:
        blur_func, function, image pipeline function
        input_imgL, list of numpy.ndarray, zero or more input image arguments to image_func
        ground_truth_output, numpy.ndarray, ground truth image to compare output to (or None)
        n, int, number of times to test the function, or None, chooses number dynamically
           (keep increasing number of tests to dynamic_max_tests so long as the total takes less
            than dynamic_max_time seconds).
        threshold: float threshold to compare against
        name: name for method to print, if None, uses module and function name.
        use_output_img: if True, passes to image_func an output image array as same type and size as input_imgL[0], as keyword arg output_img.
        imdiff_ignore_last_element: tells imdiff() whether or not to ignore the last element in the last dimension (i.e. ignore the A in RGBA),
           but this is only applied if the global use_4channel is also True.
        
    Returns {'time': time_float_secs, 'error': error_mean_rgb, 'output': output_img, 'total_time': time_float_secs}.
    
    """
    if dtype is None:
        dtype = default_test_dtype
    if dtype is not None:
        input_imgL = cast_args(input_imgL, dtype)

    if verbose:
        print("\nTesting " + (str(image_func.__module__) + "." +
            str(image_func.__name__) + "():" if name is None else name))

    error = 0.0
    
    kw = {}
    if use_output_img:
        if output_img_shape is None:
            output_img = input_imgL[0].copy()
        else:
            output_img = numpy.zeros(output_img_shape, dtype if dtype is not None else 'float64')
    
    if override_n_tests is not None:
        n = override_n_tests

    nmax = n if (n is not None) else dynamic_max_tests

    if additional_args is not None:
        input_imgL = tuple(list(input_imgL) + list(additional_args))
    if additional_kw_args is None:
        additional_kw_args = {}

    if not allow_kwargs and len(additional_kw_args):
        warnings.warn('keyword arguments disabled by allow_kwargs=False')

    min_time_L = []

    for repeat in range(nrepeats):
        times = []
        for i in range(nmax):
            if verbose and (n is not None):
                overwrite_line("{:.1f}% done".format(float(i) * 100.0 / n))
        
            if not use_output_img:
                if allow_kwargs and len(additional_kw_args):
                    t1 = time.time()
                    output_img = image_func(*input_imgL, **additional_kw_args)
                    t2 = time.time()
                else:
                    t1 = time.time()
                    output_img = image_func(*input_imgL)
                    t2 = time.time()
            else:
                if allow_kwargs:
                    t1 = time.time()
                    image_func(*input_imgL, output_img=output_img, **additional_kw_args)
                    t2 = time.time()
                else:
                    arg_tuple = tuple(input_imgL) + (output_img,)
                    t1 = time.time()
                    image_func(*arg_tuple)
                    t2 = time.time()
            if output_gain != 1.0 or output_bias != 0.0:
                output_img = output_img*output_gain + output_bias

            times.append(t2 - t1)
            output_img = numpy.asarray(output_img)

            if ground_truth_output is not None and not is_initial_run:
                error = imdiff(ground_truth_output, output_img, ignore_last_element=imdiff_ignore_last_element, originally_3channel=originally_3channel)
                if error > threshold or math.isnan(error):
                    debug_ground_truth = 'debug_ground_truth.png'
                    debug_output_img = 'debug_output_img.png'
                    write_img(ground_truth_output, debug_ground_truth)
                    write_img(output_img, debug_output_img)
                    raise ValueError("Output from function failed diffing with ground truth image ({}): error={}. For debugging, wrote ground truth to {}, wrote output image to {}".format(ground_truth_filename, error, debug_ground_truth, debug_output_img))

            if verbose and (n is not None):
                overwrite_line("{:.1f}% done".format(float(i + 1) * 100.0 / n))

            if n is None and numpy.sum(times) >= dynamic_max_time:
                break

        if verbose:
            print("\nResults from running " + str(len(times)) + " time(s):")

        sum_time = float(sum(times))
        ave_time = sum_time / len(times)
        max_time = max(times)
        min_time = min(times)

        if verbose:
            #print("--- Running time ---")
            #print("Average time: " + str(ave_time) + " seconds")
            #print("Slowest time: " + str(max_time) + " seconds")
            print("Fastest time: " + str(min_time) + " seconds")
            print("Error: " + str(error))

        min_time_L.append(min_time)

    return {'time': numpy.mean(min_time_L), 'error': float(error), 'output': output_img, 'total_time': sum_time}

def test_image_pipeline_filename(image_func, in_filenameL, n=None, verbose=True, threshold=DEFAULT_THRESHOLD, grayscale=False, name=None, additional_args=None, numpy_dtype=None, use_output_img=False, dynamic_max_tests=DEFAULT_DYNAMIC_MAX_TESTS, dynamic_max_time=DEFAULT_DYNAMIC_MAX_TIME, imdiff_ignore_last_element=False, dtype=None, additional_kw_args=None, allow_kwargs=True, output_img_shape=None, output_gain=1.0, output_bias=0.0):
    """
    Test an image pipeline using the given input image(s), which are given in the list in_filenameL (this list may be empty if no input image is used).
    
    A ground truth output image is created if it does not exist.
    If it does exist then it is compared against.
    If the global variable override_input_image is a string and a single image filename is provided, then it will be overridden by override_input_image.
    """
    if len(in_filenameL) == 1 and override_input_image is not None:
        in_filenameL = [override_input_image]
    
    if len(in_filenameL):
        name_prefix = ''
#        print('hasattr:', hasattr(image_func, '__name__'))
#        if hasattr(image_func, '__name__'):
#            name_prefix = image_func.__name__ + '_'
#        print('name_prefix:', name_prefix)
        ground_truth_prefix = name_prefix + os.path.split(os.path.splitext(in_filenameL[0])[0])[1] + '_'
    else:
        ground_truth_prefix = ''
        
    ground_truth_filename = ground_truth_prefix + 'ground_' + image_func.__name__ + '.png'

    ground_truth_output = None
    if os.path.exists(ground_truth_filename):
        ground_truth_output = read_img(ground_truth_filename, grayscale=grayscale)

    if not additional_args:
        additional_args = tuple()

    extra_info = {}
    input_imgL = [read_img(in_filename, grayscale=grayscale, extra_info=extra_info) for in_filename in in_filenameL]
    originally_3channel = extra_info.get('originally_3channel', False)
    
    if numpy_dtype is not None:
        input_imgL = cast_args(input_imgL, numpy_dtype)

    res = test_image_pipeline(image_func,
                              tuple(list(input_imgL)),
                              n, 
                              ground_truth_output, 
                              verbose, 
                              threshold, 
                              name,
                              use_output_img,
                              dynamic_max_tests,
                              dynamic_max_time,
                              imdiff_ignore_last_element,
                              dtype,
                              additional_args=additional_args,
                              additional_kw_args=additional_kw_args,
                              allow_kwargs=allow_kwargs,
                              output_img_shape=output_img_shape,
                              originally_3channel=originally_3channel,
                              output_gain=output_gain,
                              output_bias=output_bias,
                              ground_truth_filename=ground_truth_filename)

    if override_output_image is not None:
        write_img(res['output'], override_output_image)
    
    if is_initial_run:
#        print('Read {}'.format(in_filenameL))
        print('Writing {}'.format(ground_truth_filename))
#        print('  min: {}, max: {}'.format(res['output'].min(), res['output'].max()))
        write_img(res['output'], ground_truth_filename, False) #skimage.io.imsave(ground_truth_filename, res['output'])

    return res

def combine_tests(L):
    """
    Combine the dictionaries returned by several tests, returning a dict with keys 'time', 'error', and 'total_time' mapping to floats.
    
    If L has length 1 then return just the single item (thus preserving also other keys such as 'output').
    """
    if len(L) == 1:
        return L[0]
    
    ans = {}
    for key in ['time', 'error', 'total_time']:
        ans[key] = sum([x[key] for x in L])
    return ans

def print_log(s, log_file=[None]):
    """
    Print to a log file log.txt, to work around SuppressOutput.
    """
    if log_file[0] is None:
        log_file[0] = open('log.txt', 'w')
    print(s, file=log_file[0])

class SuppressOutput:
    """
    Suppress output if verbose is False, unless there is an error: print when there are errors (so long as verbose_on_fail is True).
    """
    def __init__(self, verbose=False, verbose_on_fail=True):
        self.verbose = verbose
        self.verbose_on_fail = verbose_on_fail
    
    def __enter__(self):
        if not self.verbose:
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            sys.stdout = self.stdout = io.StringIO()
            sys.stderr = self.stderr = io.StringIO()

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.verbose:
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            if exc_type is not None and not self.verbose_on_fail:
                print(self.stdout.getvalue(), end='')
                print(self.stderr.getvalue(), end='')

class ProcessError(Exception):
    def __init__(self, returncode):
        self.returncode = returncode

    def __repr__(self):
        return 'util.ProcessError({})'.format(self.returncode)

def system_verbose(cmd, exit_if_error=True):
    """
    Similar to os.system(), but prints command executed and exits if there is an error (unless exit_if_error is False, in which case, returns code).
    """
    print(cmd)
    code = os.system(cmd)
    if code != 0:
        if exit_if_error:
            sys.exit(1)
    return code

def system(s):
    """
    Similar to os.system(), but works with SuppressOutput. Raises ProcessError if there is an error.
    """
    try:
        ans = subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT)
        print(ans.decode('utf-8'), end='')
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf-8'))
        raise ProcessError(e.returncode)

def compile_c_single_file(filename, vectorize=True):
    """
    Compile C source (excluding .c extension) in place to module.
    """
    c_filename = filename + '.c'

    # this is a temporary hack to force the use of GCC instead of clang
    if os.path.isfile("/usr/local/bin/gcc-5"):
        print("using gcc-5 compiler from homebrew...")
        os.environ["CC"] = os.environ["CXX"] = "gcc-5"
    else:
        os.environ["CC"] = os.environ["CXX"] = "gcc"

    if os.name == 'nt':
        extension = [
            Extension(
                filename,
                sources = [c_filename],
                include_dirs = [numpy.get_include()],
                extra_compile_args = '-openmp'.split(),
                extra_link_args = '-openmp'.split()
            )
        ]
    
    else:
        extension = [
            Extension(
                filename,
                sources = [c_filename],
                include_dirs = [numpy.get_include()],
                extra_compile_args = '-w -fopenmp'.split() + (['-fno-tree-vectorize'] if not vectorize else []),
                extra_link_args = '-fopenmp'.split()
            )
        ]

    setup(
        name = filename,
        cmdclass = {'build_ext' : build_ext}, 
        include_dirs = [numpy.get_include()],
        ext_modules = extension,
        script_args='build_ext --inplace'.split()
    )

def compile_cython_single_file(filename, edit_c_func=None, extra_info=None, vectorize=True, compile_c=True):
    """
    Compile a single Cython extension file (excluding the .pyx extension) in place using distutils.
    
    If edit_c_func is not None, then edit_c_func(source_c, source_cython) is called, which takes C and Cython sources as input and returns a modified C source string.
    """
    pyx_filename = filename + '.pyx'
    c_filename = filename + '.c'

    # Run Cython twice because -a appears to suppress line directives
    system('cython -a -3 --fast-fail {}'.format(pyx_filename))
    system('cython --line-directives -3 {}'.format(pyx_filename))
    
    if extra_info is not None:
        html = open(filename + '.html', 'rt').read()
        extra_info['html'] = html
    
    if edit_c_func is not None:
        source_c = open(c_filename, 'rt').read()
        source_cython = open(pyx_filename, 'rt').read()
        source_c = edit_c_func(source_c, source_cython)
        with open(c_filename, 'wt') as f:
            f.write(source_c)

    if compile_c:
        compile_c_single_file(filename, vectorize)

def print_header(header, s='', file=None, linenos=False, file_list=None):
    if file_list is not None:
        for file_p in file_list:
            print_header(header, s, file_p, linenos)
        return

    if file is None:
        file = sys.stdout
    print('-'*80, file=file)
    print(header, file=file)
    print('-'*80, file=file)
    if not len(s):
        return
    if linenos:
        for (i, line) in enumerate(s.split('\n')):
            print('{:04d} {}'.format(i+1, line), file=file)
    else:
        print(s, file=file)
    print('-'*80, file=file)

def parse_cython_array(s):
#    with open('t.txt','wt') as f:
#        f.write(repr(type(s)) + ' ' + repr(s))
#
    dot1 = s.index('.')
    dot2 = s.index('.', dot1+1)
    underscore = s.index('_')
    primitive_type = s[dot2+1:underscore]
    ndim = s.index('ndim=')
    endbracket = s.index(']')
    ndim_val = s[ndim + len('ndim='):endbracket]
    return (ndim_val, primitive_type)

def promote_numeric(a, b, get_index=False):
    """
    If a and b are numeric types, return the one that has the type of a + b. If not, simply return a.
    
    If get_index is True then return 0 if a was selected, otherwise 1.
    """
    # Promote numeric argument types to a common type
    numeric_types = (bool, int, float, complex, numpy.float32, numpy.float64, numpy.int64, numpy.complex64, numpy.complex128)
    if isinstance(a, numeric_types) and isinstance(b, numeric_types):
        promoted_value = a + b
        if type(a) == type(promoted_value):
            return a if not get_index else 0
        else:
            return b if not get_index else 1
    return a if not get_index else 0

class UnionFailure(Exception):
    pass

def value_to_nickname(x):
    """
    Convert arbitrary Python value with repr() to a nickname suitable for CythonType.cython_nickname.
    """
    r = repr(x)
    allowed_chars = string.ascii_letters
    return ''.join([(c if c in allowed_chars else '_') for c in r]).strip('_')

class CythonType:
    """
    Constructs both Python and Cython type information from a value.
    
    Instance properties:
        cython_type:     Full Cython type string or other object such as tuple/dict/list (call cython_type_str()
                         to always obtain a single string).
        cython_nickname: Nickname string which can also be used in a C identifier (e.g. a variable name).
        shape:           Numpy array shape. The length is None for any dimension that changes in length.
                         If scalar then shape is ().
        shape_list:      List of all shapes encountered during program run, where each shape is a tuple of ints.
                         If scalar then shape_list is empty. If global variable track_shape_list is False then
                         shape_list is always empty.
    
    CythonType can be constructed using CythonType.from_value() or CythonType.from_cython_type(). These constructors
    take a compiler.ProgramInfo instance (program_info), which has a is_rewrite_float32() method that is used to
    determine whether float64 types should be rewritten to float32 types.
    
    CythonType can be constructed for:
     - Float, int, bool, and string types.
     - Tuple types. In this case, cython_type is a tuple of CythonType instances, cython_nickname is a single nickname
       string that concatenates sub-type nicknames, shape is (n,), where n is the length of the tuple, and shape_list is [(n,)].
     - String type. In this case, cython_type is 'str', cython_nickname is str, shape is (), and shape_list is [()].
     - Dict types. In this case, cython_type is a dict mapping unmodified instances for the dictionary keys to values which
       are CythonType instances constructed from the dictionary values (e.g. CythonType.from_value({'a': 10}) has
       cython_type of {'a': CythonType.from_value(10)}). Also, cython_nickname is a single nickname string that concatenates
       sub-type nicknames, shape is (n,), where n is the dict len, and shape_list is [(n,)].
     - List types. In this case, this list is assumed to homogeneous (of identical type) but arbitrary length that varies
       at run-time. Thus, cython_type is a list of a single CythonType instance storing the element type, cython_nickname
       is a single nickname string which includes the element type, shape is (n,), where n is the known length of the list
       or None if unknown length, and shape_list is [(n,)].
     - Object type, constructed with CythonType.from_value(object(), ...), which indicates a type could not be inferred.
    """
    constant_shape_max_size = 30
    
    promote_primitive_types = {
        ('bool',   'bool' ):     'bool',
        ('int',    'bool' ):     'int',
        ('float',  'bool' ):     'float',
        ('double', 'bool' ):     'double',

        ('bool',   'int'   ):    'int',
        ('int',    'int'   ):    'int',
        ('float',  'int'   ):    'float',
        ('double', 'int'   ):    'double',

        ('bool',   'float' ):    'float',
        ('int',    'float' ):    'float',
        ('float',  'float' ):    'float',
        ('double', 'float' ):    'float',

        ('bool',   'double'):    'double',
        ('int',    'double'):    'double',
        ('float',  'double'):    'double',
        ('double', 'double'):    'double',
    
        ('str', 'str'):          'str',
    }
    
    def equal(self, other, flex_shape=True):
        """
        Equality operator that correctly compares all fields. If flex_shape is True then permit None to equal a known shape size.
        """
        # TODO: Replace comparison operators such as __eq__ with this impleemntation?
        # (But TypeSignature and other places may rely on the currently broken behavior of __eq__, etc, so this has to be done with some care).

        if self.cython_nickname != other.cython_nickname:
            return False
        if flex_shape:
            if len(self.shape) != len(other.shape):
                return False
            if any([self.shape[i] != other.shape[i] and (self.shape[i] is not None and other.shape[i] is not None) for i in range(len(self.shape))]):
                return False
        else:
            if self.shape != other.shape:
                return False
        self_cython_type = self.cython_type
        other_cython_type = other.cython_type
        if isinstance(self_cython_type, str) and isinstance(other_cython_type, str):
            return self_cython_type == other_cython_type
        elif isinstance(self_cython_type, (list, tuple)) and isinstance(other_cython_type, (list, tuple)):
            return self_cython_type[0].equal(other_cython_type[0], flex_shape)
        elif isinstance(self_cython_type, dict) and isinstance(other_cython_type, dict):
            if self_cython_type.keys() != other_cython_type.keys():
                return False
            for key in self_cython_type:
                if not self_cython_type[key].equal(other_cython_type[key], flex_shape):
                    return False
            return True
        return False

    
    def is_subtype(self, other, flex_shape=True):
        """
        Whether self is a subtype of CythonType other.  If flex_shape is True then permit None to equal a known shape size.
        """
#        print('entering is_subtype')
        union_type = union_cython_types(self, other, numeric_promotion=False)
#        print('(A) self.shape={}, other.shape={}, union_type.shape={}'.format(self.shape, other.shape, union_type.shape))
#        print('is_subtype: self={}, other={}, union_type={}'.format(self, other, union_type))
#        print('(B) self.shape={}, other.shape={}, union_type.shape={}'.format(self.shape, other.shape, union_type.shape))
        return other.equal(union_type, flex_shape)
    
    def sorted_key_list(self):
        """
        Sorted key list, valid only if dict type (that is, self.is_dict()).
        """
        return sorted(self._cython_type.keys())
    
    def cython_type_str(self, force_non_buffer=False):
        """
        Convert to a single string suitable for cdef type declaration in Cython.
        
        If program_info is a compiler.ProgramInfo instance then adds any relevant Cython class or header declarations
        to program_info.cython_headers. Specifically, cython_headers is a dict, the key is the class name string, and the
        value is the string of the Cython class declaration.
        
        If force_non_buffer is True then the return type string is forced to not be a buffer type (e.g. 'numpy.ndarray'
        instead of 'numpy.ndarray[numpy.float64_t, ndim=2]').
        """
        if self.is_tuple():
            return 'tuple'
        elif self.is_dict():
            if program_info is not None:
                if self.cython_nickname not in program_info.cython_headers:
                    header_str = ['cdef class ' + self.cython_nickname + ':']
                    keys = self.sorted_key_list()
                    for key in keys:
                        assert isinstance(key, str), 'not string key: {}'.format(key)
                        header_str.append('  cdef public ' + self._cython_type[key].cython_type_str(force_non_buffer=True) + ' ' + key)
                    header_str.append('')

                    header_str.append('  def __init__(self, ' + ','.join(keys) + '):')
                    for key in keys:
                        header_str.append('    self.{} = {}'.format(key, key))
                    header_str.append('')

                    header_str.append('  def __getitem__(self, key):')
                    for (i, key) in enumerate(keys):
                        header_str.append('    {} key == {}: return self.{}'.format('if' if i == 0 else 'elif', repr(key), key))
                    header_str.append('    else: raise KeyError(key)')
                    header_str.append('')

                    header_str.append('  def __setitem__(self, key, value):')
                    for (i, key) in enumerate(keys):
                        header_str.append('    {} key == {}: self.{} = value'.format('if' if i == 0 else 'elif', repr(key), key))
                    header_str.append('    else: raise KeyError(key)')
                    header_str.append('')

                    header_str.append('  def __len__(self):')
                    header_str.append('    return {}'.format(self.shape[0]))

                    header_str = '\n'.join(header_str)
                    program_info.cython_headers[self.cython_nickname] = header_str
        
            return self.cython_nickname
        elif self.is_list():
            return 'list'
        if force_non_buffer and isinstance(self._cython_type, str) and '[' in self._cython_type:
            return self._cython_type[:self._cython_type.index('[')]
        def convert_bool(s):
            if s == 'bool':
                return '_cbool'
            return s
        return self.rewrite_float32_str(convert_bool(self._cython_type)) #self.cython_type

    def rewrite_float32_str(self, s):
        """
        Get cython_type string after optionally rewriting arrays of float64 type to float32 if this has been specified by an ArrayStorage transform.
        """
        if self.is_rewrite_float32():
            if self.is_array():
                if self.primitive_type(rewrite=False) == 'double':
                    return s.replace('float64', 'float32')
        return s
    
    def __init__(self, *args):
        self.known_value = None
        if len(args):
            raise ValueError('Use either CythonType.from_value() or CythonType.from_cython_type()')

    def small_constant_shape(self, max_size=constant_shape_max_size):
        """
        True if has a small and constant shape.
        """
        return len(self.shape) and not any(v is None for v in self.shape) and (max_size is None or all(v <= max_size for v in self.shape))
    
    @staticmethod
    def dim_has_small_constant_shape(v, max_size=constant_shape_max_size):
        """
        Returns True if v is small constant shape size. Here v is a given element of self.shape (e.g. self.shape[0], or self.shape[1], etc).
        """
        return v is not None and (max_size is None or v <= max_size)
    
    numpy_to_scalar_type = {'float': 'float64'}
    
    scalar_type_to_c = {'float': 'float', 'double': 'double', 'float32': 'float', 'float64': 'double', 'bool': 'bool', 'int': 'int', 'str': 'str', 'int64': 'int64_t'}
    scalar_type_to_numpy = {'float': 'float32', 'double': 'float64', 'float32': 'float32', 'float64': 'float64', 'bool': 'bool', 'int': 'int', 'str': 'str', 'int64': 'int64'}

    def bind_4channel_type(self, other):
        self._shape_4channel = other._shape
        if self.is_list():
            self.cython_type[0].bind_4channel_type(other.cython_type[0])
        elif self.is_dict():
            for key in self.cython_type:
                self.cython_type[key].bind_4channel_type(other.cython_type[key])
        elif self.is_tuple():
            for i in range(len(self.cython_type)):
                self.cython_type[i].bind_4channel_type(other.cython_type[i])

    @property
    def shape(self):
        if self.is_use_4channel() and hasattr(self, '_shape_4channel'):
            return self._shape_4channel
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def set_shape(self, shape):
        """
        Modifies shape.
        """
        if not isinstance(shape, tuple):
            shape = tuple(shape)
        dims_changed = (len(shape) != len(self._shape))
        self._shape = shape
        if dims_changed:
            self.set_primitive_type(self.primitive_type())

    def set_primitive_type(self, ptype, is_numpy=False):
        """
        Set primitive C type string such as 'float', 'double', 'int', for scalar and array types only.
        
        If is_numpy is True then interpret the type as a numpy type. For example, numpy dtype 'float' actually corresponds to C type 'double'.
        """
        if is_numpy:
            ptype = self.numpy_to_scalar_type.get(ptype, ptype)
        
        try:
            ptype_numpy = self.scalar_type_to_numpy[ptype]
            ptype_c = self.scalar_type_to_c[ptype]
        except KeyError:
            raise KeyError('Unknown primitive type {}'.format(ptype))
        if self.shape == ():
            self._cython_type = ptype_c
        elif self.is_array():
            self._cython_type = 'numpy.ndarray[numpy.' + ptype_numpy + '_t, ndim={}]'.format(len(self.shape))
        else:
            raise ValueError('not scalar or array type: {}'.format(self))

    def primitive_type(self, is_numpy=False, allow_complex=True, rewrite=True, cython_type=False):
        """
        Get primitive C type string such as 'float' or 'double', or if is_numpy is True, a numpy type string such as 'float32'.
        
        If cython_type is True then return a CythonType instance instead of a str.
        
        If self is a tuple type then returns a tuple of such primitive C type strings, if allow_complex is True.
        If allow_complex is False, then returns the first primitive C type string from the tuple.
        
        If self is a dict type then returns a dict mapping the original keys to primitive_types, if allow_complex is True.
        If allow_complex is False, then returns the primitive C type string from the smallest key.
        
        If self is a list type then returns a list of a single primitive C type string, if allow_complex is True.
        If allow_complex is False, then returns the first primitive C type string.
        """
        if cython_type:
            return CythonType.from_cython_type(self.primitive_type(is_numpy, allow_complex, rewrite), self.program_info)
        
        def parse_type(t):
            try:
                if not is_numpy:
                    return self.scalar_type_to_c[t]
                else:
                    return self.scalar_type_to_numpy[t]
            except KeyError:
                raise KeyError('Unknown primitive type {}, is_numpy={}, allow_complex={}, rewrite={}, self._cython_type={}, self._cython_nickname={}, self.shape={}'.format(t, is_numpy, allow_complex, rewrite, self._cython_type, self._cython_nickname, self.shape))
        cython_type = self.cython_type if rewrite else self._cython_type
        
        if self.is_array():
            return parse_type(parse_cython_array(cython_type)[1])
        elif self.is_tuple():
#            with open('t.txt', 'wt') as f:
#                f.write(repr(self.cython_type))
            t = tuple([x.primitive_type() for x in cython_type])
            if allow_complex:
                return t
            else:
                return t[0]
        elif self.is_dict():
            t = [(_k, _v.primitive_type()) for (_k, _v) in cython_type.items()]
            if allow_complex:
                return dict(t)
            else:
                return t[0][1]
        elif self.is_list():
            t = [cython_type[0].primitive_type()]
            if allow_complex:
                return t
            else:
                return t[0]
        elif self.is_object():
            return self
        else:
            return parse_type(cython_type)

    def is_tuple(self):
        """
        Returns True if a tuple type.
        """
        return isinstance(self._cython_type, tuple)

    def is_array(self):
        """
        Returns True if a numpy array type.
        """
        return not self.is_tuple() and not self.is_dict() and not self.is_list() and not self.is_str() and not self.is_object() and len(self.shape)

    def is_dict(self):
        """
        Returns True if a dict type.
        """
        return isinstance(self._cython_type, dict)

    def is_list(self):
        """
        Return True if a list type.
        """
        return isinstance(self._cython_type, list)

    def is_str(self):
        return self._cython_type == 'str'
    
    def is_object(self):
        return self._cython_type == 'object'

    def is_scalar(self):
        return len(self.shape) == 0
    
#    def is_numeric(self):
#        return self._cython_type in ['bool', 'int', 'float', 'double'] and len(self.shape) == 0
    
    def c_array_type_suffix(self):
        """
        Get suffix of C array such as '[3]' or '[3][3]'. Raise error if non-constant shape or 0-dimensional.
        """
        assert len(self.shape), 'expected non-zero dimension'
        assert not any(v is None for v in self.shape), 'expected shape to be constant for all dims'
        return ''.join('[{}]'.format(v) for v in self.shape)

    def _set_shape_list(self):
        self.shape_list = [self.shape] if track_shape_list else []

    def _set_dict_info(self):
        self._cython_nickname = '_'.join(value_to_nickname(_k) + '_' + _v.cython_nickname for (_k, _v) in sorted(self._cython_type.items()))
        self.shape = (len(self._cython_type),)
        self._set_shape_list()

    def _set_list_info(self, nelems):
        self._cython_nickname = 'list_' + self._cython_type[0].cython_nickname
        self.shape = (nelems,)
        self._set_shape_list()

    def is_rewrite_float32(self):
        return self.program_info is not None and self.program_info.is_rewrite_float32()

    def is_use_4channel(self):
        return self.program_info is not None and self.program_info.is_use_4channel()

    @property
    def cython_nickname(self):
        return self.rewrite_float32_str(self._cython_nickname)
    
    @cython_nickname.setter
    def cython_nickname(self, value):
        self._cython_nickname = value
    
    @property
    def cython_type(self):
        t = self._cython_type
        if isinstance(t, str):
            return self.rewrite_float32_str(t)
        return t

    @cython_type.setter
    def cython_type(self, value):
        self._cython_type = value
    
    @staticmethod
    def from_known_value(value, program_info, error_variable=''):
        self = CythonType.from_value(value, program_info, error_variable='')
        self.known_value = value
        return self

    @staticmethod
    def from_value(value, program_info, error_variable=''):
        """
        Construct from value.
        """
        self = CythonType()
        self.program_info = program_info
        
        if isinstance(value, numpy.ndarray):
            dtype_str = str(value.dtype)
            self._cython_type = 'numpy.ndarray[numpy.{}_t, ndim={}]'.format(dtype_str, value.ndim)
            self._cython_nickname = 'array{}{}'.format(value.ndim, dtype_str)
            self.shape = value.shape
            self._set_shape_list()
            if cython_type_check:
                self.check()
        elif isinstance(value, tuple):
            self.shape = (len(value),)
            L = [CythonType.from_value(x, program_info) for x in value]
            self._cython_nickname = '_'.join(x.cython_nickname for x in L)
            self._cython_type = tuple(L) #tuple([x.cython_type for x in L])
            self._set_shape_list()
            if cython_type_check:
                self.check()
        elif isinstance(value, dict):
            self._cython_type = {_k: CythonType.from_value(_v, program_info) for (_k, _v) in value.items()}
            if not all(isinstance(_k, str) for _k in self.cython_type):
                raise NotImplementedError('CythonType from dict with non-string keys not currently supported: {}'.format(value))
            self._set_dict_info()
            if cython_type_check:
                self.check()
        elif isinstance(value, list):
            if len(value) == 0:
                raise NotImplementedError('unsupported type: empty list')
            self._cython_type = [CythonType.from_value(value[0], program_info)]
            for element in value[1:]:
                self._cython_type[0].union_inplace(CythonType.from_value(element, program_info))
            self._set_list_info(len(value))
        else:
            if isinstance(value, (float, numpy.float64)):
                self._cython_type = 'double'
            elif isinstance(value, (numpy.float32)):
                self._cython_type = 'float'
            elif isinstance(value, (bool, numpy.bool_)):  # This test must be before the int test because, cryptically, isinstance(False, int) is True.
                self._cython_type = 'bool'
            elif isinstance(value, (int, numpy.int64)):
                self._cython_type = 'int'
            elif isinstance(value, str):
                self._cython_type = 'str'
            elif value.__class__ is object().__class__ or value is None:
                self._cython_type = 'object'
            else:
                raise NotImplementedError('unsupported type: {!r} (type: {!r}, error_variable: {})'.format(value, type(value), error_variable))
            self._cython_nickname = self.cython_type
            self.shape = ()
            self.shape_list = []
            if cython_type_check:
                self.check()
        return self
    
    def check(self):
        """
        Run sanity checks for debugging purposes.
        """
#        if len(self.shape) == 1 and self.shape[0] is None:
#            raise ValueError((self.cython_type, self.cython_nickname, self.shape, self.shape_list))
        
        if isinstance(self._cython_type, tuple):
            for x in self._cython_type:
                if isinstance(x._cython_type, str):#, (self.cython_type, self.cython_nickname, self.shape, self.shape_list)
                    if 'shape=' in x._cython_type:
                        raise ValueError((self.cython_type, self.cython_nickname, self.shape, self.shape_list))
        elif isinstance(self._cython_type, list):
            if len(self._cython_type) != 1:
                raise ValueError((self.cython_type, self.cython_nickname, self.shape, self.shape_list))
        else:
            if 'shape=' in self._cython_type:
                raise ValueError((self.cython_type, self.cython_nickname, self.shape, self.shape_list))

    @staticmethod
    def from_cython_type(cython_type_shapeinfo, program_info):
        """
        Construct from cython_type str or tuple (can also include shape info, in the format returned by CythonType.__repr__()).
        """
#        print('from_cython_type({!r} (type={})'.format(cython_type_shapeinfo, type(cython_type_shapeinfo)))
        if isinstance(cython_type_shapeinfo, CythonType):
            ans = copy.deepcopy(cython_type_shapeinfo)
            ans.program_info = program_info
            return ans
#            cython_type_shapeinfo = str(cython_type_shapeinfo)
        is_str = isinstance(cython_type_shapeinfo, str)
        if isinstance(cython_type_shapeinfo, tuple) or (is_str and cython_type_shapeinfo.startswith('(')):
            if isinstance(cython_type_shapeinfo, tuple):
                L_args = cython_type_shapeinfo
            else:
                L_args = eval(cython_type_shapeinfo)
            L = [CythonType.from_cython_type(x, program_info) for x in L_args]
            ans = CythonType()
            ans.program_info = program_info
            ans._cython_type = tuple([x for x in L])
            ans._cython_nickname = '_'.join(x.cython_nickname for x in L)
            ans.shape = (len(L),)
            ans._set_shape_list()
            if cython_type_check:
                ans.check()
            return ans
        elif isinstance(cython_type_shapeinfo, dict) or (is_str and cython_type_shapeinfo.startswith('{')):
            if not isinstance(cython_type_shapeinfo, dict):
                cython_type_shapeinfo = eval(cython_type_shapeinfo)
            L_args = sorted(cython_type_shapeinfo.items())
            ans = CythonType()
            ans.program_info = program_info
            ans._cython_type = {_k: CythonType.from_cython_type(_v, program_info) for (_k, _v) in L_args}
            ans._set_dict_info()
            if cython_type_check:
                ans.check()
            return ans
        elif isinstance(cython_type_shapeinfo, list) or (is_str and (cython_type_shapeinfo.startswith('[') or cython_type_shapeinfo.startswith('"['))):
            if isinstance(cython_type_shapeinfo, list):
                nelems = len(cython_type_shapeinfo)
            elif is_str and (cython_type_shapeinfo.startswith('[') or cython_type_shapeinfo.startswith('"[')):
                if cython_type_shapeinfo.startswith('"') and cython_type_shapeinfo.endswith('"'):
                    cython_type_shapeinfo = cython_type_shapeinfo[1:-1]
                nelems = None
                if 'shape=(' in cython_type_shapeinfo:
                    idx_start0 = cython_type_shapeinfo.rindex('shape=(')
                    idx_start = idx_start0 + len('shape=(')
                    try:
                        comma = cython_type_shapeinfo.index(',', idx_start0)
                        comma_found = True
                    except ValueError:
                        comma_found = False
                    if comma_found:
                        sub = cython_type_shapeinfo[idx_start:comma]
                        try:
                            nelems = int(sub)
                        except ValueError:
                            warnings.warn('could not parse shape field of list in CythonType.from_cython_type() constructor: substring is {}'.format(sub))
                    else:
                        warnings.warn('could not parse shape field of list in CythonType.from_cython_type() constructor')
                    cython_type_shapeinfo = cython_type_shapeinfo[:idx_start0-1]
            else:
                raise ValueError
            #print('cython_type_shapeinfo={}'.format(cython_type_shapeinfo))
            if not isinstance(cython_type_shapeinfo, list):
                cython_type_shapeinfo = eval(cython_type_shapeinfo)
            if len(cython_type_shapeinfo) == 0:
                raise ValueError('In CythonType.from_cython_type({!r}), cannot parse length zero list type'.format(cython_type_shapeinfo))
            ans = CythonType()
            ans.program_info = program_info
            ans._cython_type = [CythonType.from_cython_type(cython_type_shapeinfo[0], program_info)]
            ans._set_list_info(nelems)
            if cython_type_check:
                ans.check()
            return ans
        
        cython_type_shapeinfo = cython_type_shapeinfo.strip("'\"")
        self = CythonType()
        self.program_info = program_info
        self._cython_type = cython_type_shapeinfo
        self.shape = None
        self.shape_list = []
        if '(' in self._cython_type and ')' in self._cython_type:
            lparen = self._cython_type.index('(')
            rparen = self._cython_type.rindex(')')
            shapeinfo = self._cython_type[lparen+1:rparen]
            self._cython_type = self._cython_type[:lparen]
#            print('shapeinfo:', shapeinfo)
            try:
                paren_comma = shapeinfo.index('),')
            except ValueError:
                raise ValueError('In CythonType.from_cython_type({!r}), could not find ")," in {!r}'.format(cython_type_shapeinfo, shapeinfo))
            if shapeinfo.startswith('shape=') and shapeinfo[paren_comma+2:].startswith('shape_list='):
                shape_listinfo = shapeinfo[paren_comma+2+len('shape_list='):]
                shapeinfo = shapeinfo[len('shape='):paren_comma+1]
                self.shape = eval(shapeinfo)
                self.shape_list = eval(shape_listinfo) if track_shape_list else []
        if self._cython_type.startswith('numpy.ndarray'):
            (ndim_val, primitive_type) = parse_cython_array(self._cython_type)
            self._cython_nickname = 'array{}{}'.format(ndim_val, primitive_type)
            if self.shape is None:
                self.shape = tuple([None for i in range(int(ndim_val))])
                self.shape_list = []
        else:
            self._cython_nickname = self._cython_type
            if self.shape is None:
                self.shape = ()
                self.shape_list = []

        if cython_type_check:
            self.check()
        return self

    def __repr__(self):
#        print('in __repr__, shape={}, cython_type type: {}'.format(self.shape, type(self.cython_type)))
        if cython_type_check:
            self.check()
        r = self.cython_type
        if isinstance(self.cython_type, tuple):
            return str(r)
        elif isinstance(self.cython_type, dict):
            return '{' + ', '.join(repr(key) + ': ' + repr(value) for (key, value) in sorted(self.cython_type.items())) + '}'
        elif isinstance(self.cython_type, list):
            str_r = str(r)
            if str_r.startswith('["') and str_r.endswith('"]'):
                str_r = '[' + str_r[2:-2] + ']'
            return '"' + str_r + '(shape={})'.format(self.shape) + '"'
        return "'" + r + ('(shape={},shape_list={})'.format(self.shape, self.shape_list) if (len(self.shape) or len(self.shape_list)) else '') + "'"

    def to_immutable(self):
        """
        Return cython_type attribute which has been converted to an immutable (hashable) form that is unique.
        """
        t = self.cython_type
        if isinstance(t, dict):
            t = tuple(sorted(t.items()))
        elif isinstance(t, list):
            t = tuple([t[0], None])
        return t

    def __hash__(self):
        return hash(self.to_immutable())
    
    def __eq__(self, other):
        return self.cython_type == other.cython_type

    def __lt__(self, other):
        return self.cython_type < other.cython_type
    
    def __gt__(self, other):
        return self.cython_type > other.cython_type

    def __le__(self, other):
        return self.cython_type <= other.cython_type

    def __ge__(self, other):
        return self.cython_type >= other.cython_type
    
    def __ne__(self, other):
        return self.cython_type != other.cython_type

    def isinstance_check(self, arg):
        """
        Convert to a code string that checks whether the string arg is an instance of the current Cython type.
        """

        s = self.cython_type #self.cython_type

        if s == 'double':
            return 'isinstance({}, (float, numpy.float64))'.format(arg)
        elif s == 'float':
            return 'isinstance({}, numpy.float32)'.format(arg)
        elif s == 'int':
            return 'isinstance({}, (int, numpy.int64))'.format(arg)
        elif s == 'bool':
            return 'isinstance({}, (bool, numpy.bool_))'.format(arg)
        elif s == 'str':
            return 'isinstance({}, str)'.format(arg)
        elif isinstance(s, str) and s.startswith('numpy.ndarray'):
            (ndim_val, primitive_type) = parse_cython_array(s)

            result = \
                ('(isinstance(%s, numpy.ndarray) and' % arg) + \
                (' %s.dtype == numpy.%s and' % (arg, primitive_type)) + \
                (' %s.ndim == %s' % (arg, ndim_val))

            for i in range(len(self.shape)):
                if self.dim_has_small_constant_shape(self.shape[i]):
                    result += (' and %s.shape[%d] == %d' % (arg, i, self.shape[i]))

            result += ')'

            return result
        elif self.is_tuple():
            return 'isinstance({}, tuple)'.format(arg)
        elif self.is_dict():
            return 'isinstance({}, dict)'.format(arg)
        elif self.is_list():
            return 'isinstance({}, list)'.format(arg)
        else:
            raise ValueError

    def assign_inplace(self, t):
        """
        In-place assignment operator: overwrites properties of self with those of t.
        """
        self._cython_type = t._cython_type
        self._cython_nickname = t._cython_nickname
        self.shape = t.shape
        self.shape_list = t.shape_list
    
    def union_inplace(self, t, warn=True, numeric_promotion=True, numpy_promotion=False):
        """
        Deprecated type union. Please use the function union_cython_types() instead.
        
        Attempt to union in place self with CythonType t, including unioning array shapes and promoting numeric types if needed.
        
        On success, return True. On failure, due nothing, issue a warning (if warn is True), and return False.
        """
#        print('union_inplace({}, {})'.format(self, t))
        p_s = self.primitive_type()
        p_t = t.primitive_type()

        if isinstance(p_s, str) and isinstance(p_t, str):
            try:
                p_promoted = CythonType.promote_primitive_types[(p_s, p_t)]
            except (UnionFailure, KeyError):
                if warn:
                    warnings.warn('could not union types: {}, {}'.format(self, t))
                return False
            if not numeric_promotion:
                if p_promoted != p_s or p_promoted != p_t:
                    if warn:
                        warnings.warn('could not union types in numeric_promotion=False mode: {}, {}'.format(self, t))
                    return False
            if p_promoted == p_t:
                self._cython_type = t._cython_type
                self._cython_nickname = t._cython_nickname
            try:
                self.shape = union_shapes(self.shape, t.shape, numpy_promotion=numpy_promotion)
            except UnionFailure:
                if warn:
                    warnings.warn('could not union shapes: {}, {}'.format(self.shape, t.shape))
                return False
            self.set_primitive_type(p_promoted)
            if track_shape_list:
                self.shape_list.extend(t.shape_list)
        elif isinstance(p_s, tuple) and isinstance(p_t, tuple) and len(p_s) == len(p_t):
            L_s = list(self._cython_type)
            L_t = list(t._cython_type)
            for i in range(len(L_s)):
                L_s[i].union_inplace(L_t[i])
            self._cython_type = tuple(L_s)
        elif isinstance(p_s, dict) and isinstance(p_t, dict) and p_s.keys() == p_t.keys():
            for key in self._cython_type.keys():
                self._cython_type[key].union_inplace(t._cython_type[key])
        elif isinstance(p_s, list) and isinstance(p_t, list) and len(p_s) >= 1 and len(p_t) >= 1:
            self._cython_type[0].union_inplace(t._cython_type[0])
        elif self.is_array() and t.is_list():
            pass
        elif self.is_list() and t.is_array():
            self.assign_inplace(t)
        else:
#            raise ValueError('unknown types for union_inplace: {}, {}'.format(self, t))
            if warn:
                warnings.warn('unknown types for union_inplace: {}, {}'.format(self, t))
            return False
        return True
#        def promote(current_s, current_t):
#            if isinstance(current_s, str) and isinstance(current_t, str):
#                return CythonType.promote_primitive_types[(current_s, current_t)]
#            elif isinstance(current_s, tuple) and isinstance(current_t, tuple) and len(current_s) == len(current_t):
#                return tuple([promote(current_s[i], current_t[i]) for i in range(len(current_s))])
#            elif isinstance(current_s, dict) and isinstance(current_t, dict) and current_s.keys() == current_t.keys():
#                return {_k: promote(current_s[_k], current_t[_k]) for _k in current_s.keys()}
#            else:
#                raise UnionFailure
#        try:
#            p_promoted = promote(p_s, p_t)
#        except UnionFailure:
#            warnings.warn('could not union types: {}, {}'.format(self, t))
#            return

def union_cython_types(a, b, numeric_promotion=True, numpy_promotion=False, strict=False):
    """
    Union CythonType instances a and b, returning the unioned type.
    """
    if a.is_object():
        return copy.deepcopy(a)
    if b.is_object():
        return copy.deepcopy(b)
    c = copy.deepcopy(a)
    c.known_value = None
    res = c.union_inplace(b, warn=False, numeric_promotion=numeric_promotion, numpy_promotion=numpy_promotion)
    if res:
        return c
    return CythonType.from_value(object(), a.program_info)

def union_cython_types_list(L, *args, **kw):
    """
    Union at least one CythonType instances in list L, returning the unioned type.
    """
    if len(L) == 1:
        return copy.deepcopy(L[0])
    return functools.reduce(lambda a, b: union_cython_types(a, b, *args, **kw), L)

class TypeSignature(dict):
    pass
    #def __hash__(self):
    #    return hash(frozenset(self.items()))
    #
    #def __lt__(self, other):
    #    return sorted(self.items()) < sorted(other.items())

class TypeSignatureSet:
    """
    A sorted set of type signatures for a given function.
    
    A function's type signature is a dict mapping variable name keys to CythonType instances.
    
    When constructed, the TypeSignatureSet needs a list of variable names that are arguments for the function. If a given
    type signature is missing a function variable name then it will not be added.
    """
    def __init__(self, arg_names, L=[]):
        self.arg_names = arg_names
        assert isinstance(self.arg_names, (list, tuple))
        for x in self.arg_names:
            assert isinstance(x, str), 'expected {} to be string'.format(x)
        
        self.s = {}                         # Mapping from argument types to type signatures
        
        for x in L:
            self.add(x)

    def add(self, type_sig, verbose=util_verbose):
#    def add(self, type_sig, verbose=True):
        """
        Add type signature to set. If it already exists then update shape of given type signature.
        """
        type_sig = TypeSignature(type_sig)
        #type_sig_key = tuple(sorted([(key, value.cython_type) for (key, value) in type_sig.items()]))
        type_sig_key = []
        for key in self.arg_names:
            if key in type_sig:
                type_sig_key.append(type_sig[key].to_immutable())
            else:
                warnings.warn('type signature is missing function argument {}, so not adding'.format(key))
                return
        type_sig_key = tuple(type_sig_key)
        if verbose or 1:
            print_log('TypeSignatureSet.add')
            print_log('  add: %s' % type_sig)
            print_log('  current: %s' % self.s)
            print_log('  type_sig_key: {}'.format(type_sig_key))
            typeL = list(type_sig.items())
            for j in range(len(typeL)):
                if cython_type_check:
                    typeL[j][1].check()
                print_log('  type_sig[{}].cython_type: {}'.format(j, typeL[j][1].cython_type))
        
        if type_sig_key not in self.s:
            if verbose:
                print_log('  not in self.s, adding new type signature')
            self.s[type_sig_key] = type_sig
        else:
            if verbose:
                print_log('  in self.s, updating shape')
            d = self.s[type_sig_key]
            for key in set(d.keys()) & set(type_sig.keys()):
                d[key].union_inplace(type_sig[key])

        if verbose:
            print_log('  after add: %s' % self.s)
            print_log('')
    
    def __len__(self):
        if util_verbose:
            print_log('TypeSignatureSet.__len__, len={}'.format(len(self.s)))
        return len(self.s)
        
    def __iter__(self):
        if util_verbose:
            print_log('TypeSignatureSet.__iter__, len={}'.format(len(self.s)))
        return iter([value for (key, value) in sorted(self.s.items(), key=lambda item: item[0])])
#        return iter(sorted(self.s.values()))

    def __repr__(self):
        if util_verbose:
            print_log('TypeSignatureSet.__repr__, len={}'.format(len(self.s)))
        return 'TypeSignatureSet({}, [{}])'.format(self.arg_names, ','.join(repr(typesig) for typesig in self.s.values()))

def union_shapes(s1, s2, numpy_promotion=False):
    if len(s1) != len(s2):
        #warnings.warn('shapes of two different lengths unioned: {} and {}'.format(len(s1), len(s2)))
        #return ()
        if numpy_promotion:
            return s2 if len(s1) == 0 else s1
        raise UnionFailure
    ans = tuple([None if s1[i] != s2[i] else s1[i] for i in range(len(s1))])
    if util_verbose:
        print_log('union_shapes({}, {}) => {}'.format(s1, s2, ans))
    return ans

def image_filename(filename, check_exists=True):
    """
    Get full path to image with given filename.
    """
    ans = os.path.abspath(os.path.join(util_dir, '../apps/images/', filename))
    if check_exists and not os.path.exists(ans):
        ans = os.path.abspath(os.path.join(util_dir, '../images/', filename))
        if check_exists and not os.path.exists(ans):
            raise ValueError('file {} not found (util_dir={})'.format(ans, util_dir))
    return ans

def print_twocol(a, b, n=35):
    """
    Print strings in two columns with n width of left column.
    """
    print(a.ljust(n) + b)

util_dir = os.path.abspath(os.path.split(__file__)[0])

def is_type_variable(name):
    return not name.startswith(types_non_variable_prefix)

def randrange(seed, start, stop):
    """
    A substitute for random.randrange(start, stop) which does not have side-effects
    """
    rand2_u = (seed & (2 ** 32 - 1)) ^ ((seed & 65535) << 16)
    rand2_v = (~seed) & (2 ** 32 - 1)
    rand2_v = 36969 * (rand2_v & 65535) + (rand2_v >> 16)
    rand2_u = 18000 * (rand2_u & 65535) + (rand2_u >> 16)
    rand_result = ((rand2_v << 16) + (rand2_u & 65535)) & (2 ** 32 - 1)
    return start + rand_result % (stop - start)
    
