from transforms_util import *
from transforms_base import BaseTransform

class ArrayStorage(BaseTransform):
    def __init__(self, program_info, line=None, use_float32=None, use_4channel=None):
        """
        Attempt to modify storage of arrays throughout the program.
        
        If use_float32 is True then rewrite to use float32 numpy dtype.
        If use_4channel is True then rewrite 3 channel arrays to 4 channel.
        If both are False then do nothing.
        """
        self.use_float32 = False        # Initialize to make sure args() will work if called on a partially-completed object
        self.use_4channel = False
        BaseTransform.__init__(self, program_info, line)
        if use_float32 is not None:
            self.use_float32 = use_float32
        if use_4channel is not None:
            self.use_4channel = use_4channel
        if program_info.safe:
            self.use_float32 = False
        self.check()
    
    def check(self):
        if not self.program_info.safe:
            assert self.use_float32 or self.use_4channel
    
    def args(self):
        return (self.line, self.use_float32, self.use_4channel)

    def apply(self, s):
#        for transform in self.program_info.transformL:
#            if isinstance(transform, TypeSpecialize):
#                for types in transform.typesL:
#                    for value in types.values():
#                        value.
        return s
        # use_float32 mode is handled by the macro facility
        """
        verbose = get_verbose()
    
        r = RedBaron(s)
    
        for node in redbaron_util.find_all(r, 'AtomtrailersNode'):
            if len(node.value) >= 3 and isinstance(node.value[0], redbaron.NameNode) and isinstance(node.value[1], redbaron.NameNode) and isinstance(node.value[2], redbaron.CallNode):
                s0 = node.value[0].value
                s1 = node.value[1].value
                call = node.value[2]
                if s0 in macros.numpy_modules and s1 in macros.numpy_array_storage_funcs:
                    if len(call) == 1:
                        key = util.types_non_variable_prefix + macro_match.node_str
        """
    
    def mutate(self):
        self.line = 1
        self.orig_num = 1
        r = random.randrange(2)
        if r == 0:
            self.use_float32 = True
            self.use_4channel = False
        elif r == 1:
            self.use_float32 = True
            self.use_4channel = True
        else:
            self.use_float32 = False
            self.use_4channel = True
        self.check()

