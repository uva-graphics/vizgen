
import type_utils
import util
import ast
import astor
import numpy
import copy
import warnings
import macros
#import functools

class TypeWithAliases:
    def __init__(self, type, aliases):
        assert isinstance(type, util.CythonType)
        assert isinstance(aliases, list)
        for alias in aliases:
            assert isinstance(alias, tuple)
            assert len(alias) == 2
            assert isinstance(alias[0], str)
            assert isinstance(alias[1], str)
        self.type = type
        self.aliases = aliases
        
verbose = False

def type_from_value(ctx, v):
    return util.CythonType.from_value(v, ctx.program_info)

def ObjectType(ctx):
    return type_from_value(ctx, object())

def typefunc_float(ctx, x):
    if x.known_value is not None:
        return util.CythonType.from_known_value(float(x.known_value), ctx.program_info)
    else:
        return util.CythonType.from_value(1.0, ctx.program_info)

def typefunc_int_base(func):
    def f(ctx, x):
        if x.known_value is not None:
            return util.CythonType.from_known_value(func(x.known_value), ctx.program_info)
        else:
            return util.CythonType.from_value(1, ctx.program_info)
    return f

def typefunc_int(ctx, x):
    return typefunc_int_base(int)(ctx, x)
    
def typefunc_len(ctx, x):
    return typefunc_int_base(len)(ctx, x)

def numpy_type_promote_narg_types(nargs, float_type=False, ignore_additional_args=False):
    def f(ctx, *arg_types):
        if verbose:
            print('numpy_type_promote_narg_types', arg_types)
        if len(arg_types) > nargs and not ignore_additional_args:
            return util.CythonType.from_value(object(), ctx.program_info)
        arg_types = list(arg_types[:nargs])
        if float_type:
            arg_types.append(util.CythonType.from_value(1.0, ctx.program_info))
        return util.union_cython_types_list(arg_types, numpy_promotion=True)
    return f

def typefunc_min_base(func):
    def f(ctx, *arg_types):
        if len(arg_types) == 0:
            return util.CythonType.from_value(object(), ctx.program_info)
        if len(arg_types) > 1:
            ans_type = numpy_type_promote_narg_types(len(arg_types))(ctx, *arg_types)
        else:
            primitive_types = arg_types[0].primitive_type(cython_type=True)
            ans_type = numpy_type_promote_narg_types(len(primitive_types))(ctx, *primitive_types)
        known_values = [arg_type.known_value for arg_type in arg_types]
        if all([known_value is not None for known_value in known_values]):
            ans_type.known_value = func(known_value for known_value in known_values)
        return ans_type
    return f

def typefunc_min(ctx, *arg_types):
    return typefunc_min_base(min)(ctx, *arg_types)

def typefunc_max(ctx, *arg_types):
    return typefunc_min_base(max)(ctx, *arg_types)

def typefunc_sum(ctx, *arg_types):
    return typefunc_min_base(sum)(ctx, *arg_types)

# TODO: numpy.sum() actually promotes uint8 to uint64 for the result. Need some integer promotion logic here.

def typemethod_min_base(ctx, func):
    def f(ctx, self_type, *arg_types):
        if len(args) == 0:
            ans_type = self_type.primitive_type(cython_type=True)
            if self_type.known_value is not None:
                ans_type.known_value = self_type.known_value.func()
            return ans_type
        return util.CythonType.from_value(object(), ctx.program_info)
    return f

def typemethod_min(ctx, self_type, *arg_types):
    return typemethod_min_base(min)(ctx, self_type, *arg_types)

def typemethod_max(ctx, self_type, *arg_types):
    return typemethod_min_base(max)(ctx, self_type, *arg_types)

def typemethod_sum(ctx, self_type, *arg_types):
    return typemethod_min_base(sum)(ctx, self_type, *arg_types)

def typemethod_shape(ctx, self_type, *arg_types):
    if len(arg_types) > 0:
        return ObjectType(ctx)
    return util.CythonType.from_known_value(self_type.shape, ctx.program_info)

# TODO: a.sum() actually promotes uint8 to uint64 for the result. Need some integer promotion logic here.

class typefunc_math:
    pass

class typefunc_copy:
    @staticmethod
    def copy(ctx, obj_type):
        return obj_type

def cast_to_type(ctx, arg_type, typestr):
    t = copy.deepcopy(arg_type)
    t.set_primitive_type(typestr, is_numpy=True)
    return t

class typefunc_numpy:

    @staticmethod
    def min(ctx, *arg_types):
        return typemethod_min(ctx, *arg_types)
    
    @staticmethod
    def max(ctx, *arg_types):
        return typemethod_max(ctx, *arg_types)
    
    @staticmethod
    def sum(ctx, *arg_types):
        return typemethod_sum(ctx, *arg_types)
    
    # TODO: a.sum() actually promotes uint8 to uint64 for the result. Need some integer promotion logic here.
    
    @staticmethod
    def float32(ctx, arg_type):
        return cast_to_type(ctx, arg_type, 'float32')
    
    @staticmethod
    def float64(ctx, arg_type):
        return cast_to_type(ctx, arg_type, 'float64')
    
    @staticmethod
    def ones(ctx, shape_type, dtype_type=None, order_type=None):
        if verbose:
            print('  => Called ones/zeros/empty with type:', ctx, shape, dtype, order)
        if dtype_type is not None:
            if dtype_type.known_value is None:
                return util.CythonType.from_value(object(), ctx.program_info)
            dtype = dtype_type.known_value
        else:
            dtype = None
        shape = shape_type.known_value
        if shape is not None:
            if ((isinstance(shape, (tuple, list)) and all(isinstance(x, int) for x in shape)) or isinstance(shape, int)):
                return util.CythonType.from_value(numpy.ones(shape, dtype), ctx.program_info)
        if verbose:
            print(' => typefunc numpy.ones, shape type is not known:', shape)
            print(' => typefunc numpy.ones, dtype_val could be evaluated:', dtype)
        if isinstance(shape, (tuple, list)):
            if verbose:
                print(' => typefunc numpy.ones, isinstance is Tuple/List')
            ans = util.CythonType.from_value(numpy.ones([1]*len(shape), dtype), ctx.program_info)
            ans.shape = tuple(shape)
            return ans
        return util.CythonType.from_value(object(), ctx.program_info)
    
    zeros = empty = ones
    
    abs = square = numpy_type_promote_narg_types(1)
    clip = numpy_type_promote_narg_types(1, ignore_additional_args=True)

    @staticmethod
    def array(ctx, obj_type, dtype_type=None):
        if dtype_type is not None:
            if dtype_type.known_value is None:
                return util.CythonType.from_value(object(), ctx.program_info)
            else:
                dtype = dtype_type.known_value
        else:
            dtype = None
            
        obj = obj_type.known_value
                
        if obj_type.is_list() or obj_type.is_tuple():
            shape = []
            current_type = obj_type
            while current_type.is_list() or current_type.is_tuple():
                shape.append(current_type.shape[0])
                current_type = util.union_cython_types_list(current_type.cython_type)
            primitive_type = current_type.primitive_type()
            ans_type = type_from_value(ctx, numpy.zeros((0,)))
            ans_type.set_primitive_type(primitive_type)
            ans_type.set_shape(shape)
            if obj_type.known_value is not None:
                ans_type.known_value = obj_type.known_value
            if dtype is not None:
                ans_type.set_primitive_type(dtype, is_numpy=True)
            return ans_type
        
        if obj is not None:
            if isinstance(obj, (list, tuple)):
                shape = []
                current = obj
                primitive_type = ObjectType(ctx)
                while isinstance(current, (list, tuple)):
                    shape.append(len(current))
                    max_len = 1
                    new_current = current[0]
                    for item in current:
                        if isinstance(item, (list, tuple)):
                            if len(item) > new_current:
                                max_len = len(item)
                                new_current = item
                        elif item is not None and (not primitive_type.is_object()):
                            primitive_type = util.CythonType.from_value(item)
                    current = new_current
                ans_type = type_from_value(ctx, numpy.zeros((0,)))
                ans_type.set_primitive_type(primitive_type)
                ans_type.set_shape(shape)
                ans_type.known_value = obj
                if dtype is not None:
                    ans_type.set_primitive_type(dtype, is_numpy=True)
                return ans_type
            else:
                primitive_type = obj_type.primitive_type()
                try:
                    return util.CythonType.from_known_value(numpy.array(obj, dtype))
                except:
                    pass
        else:
            ans_type = copy.deepcopy(obj_type)
            if dtype is not None:
                try:
                    ans_type.set_primitive_type(dtype, is_numpy=True)
                    return ans_type
                except:
                    pass
            else:
                return ans_type
        return util.CythonType.from_value(object(), ctx.program_info)
    
    @staticmethod
    def asarray(ctx, obj, dtype=None):
        # TODO: need workaround here
        return TypeWithAliases(typefunc_numpy.array(ctx, obj, dtype), obj.aliases)

    @staticmethod
    def dot(ctx, a, b):
        a = typefunc_numpy.array(ctx, a)
        b = typefunc_numpy.array(ctx, b)
        if a.is_array() and b.is_array() and len(a.shape) == 1 and len(b.shape) == 1:
            return a.primitive_type(cython_type=True)
        return util.CythonType.from_value(object(), ctx.program_info)

    class linalg:
        @staticmethod
        def norm(ctx, a, ord=None):
            return util.CythonType.from_value(1.0, ctx.program_info)

for (_math_func, _math_nargs, _math_source_func, _numpy_source_func) in macros.unpack_math_funcs():
    if _math_func in ['floor', 'ceil']:
        setattr(typefunc_math, _math_source_func, numpy_type_promote_narg_types(_math_nargs, float_type=False))
    else:
        setattr(typefunc_math, _math_source_func, numpy_type_promote_narg_types(_math_nargs, float_type=True))
    setattr(typefunc_numpy, _numpy_source_func, numpy_type_promote_narg_types(_math_nargs, float_type=True))
    # TODO: fix: math functions should not actually return numpy arrays, and math.floor(), ceil() return ints

class typefunc_random:
    def randrange(ctx, start, stop=None, step=None):
        return util.CythonType.from_value(1, ctx.program_info)

class typefunc_util:
    def randrange(ctx, seed, start, stop):
        return util.CythonType.from_value(1, ctx.program_info)

class typefunc_time:
    def time(ctx):
        return util.CythonType.from_value(1.0, ctx.program_info)

def typefunc_pow(ctx, left, right):
    if left.shape == () and left.cython_type == 'int':
        if right.shape == () and right.cython_type == 'int':
            if right.known_value is not None and right.known_value > 0:
                ans_type = util.CythonType.from_value(1, ctx.program_info)
            else:
                ans_type = util.CythonType.from_value(1.0, ctx.program_info)
            try:
                ans_type.known_value = left.known_value ** right.known_value
            except:
                pass
            return ans_type
    return util.union_cython_types(left, right, numpy_promotion=True)

typefunc_abs = numpy_type_promote_narg_types(1)

def typefunc_range(ctx, start, stop=None, step=None):
    return util.CythonType.from_value([1], ctx.program_info)