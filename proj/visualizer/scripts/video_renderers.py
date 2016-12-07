import sys; sys.path += ["../../compiler", "../../apps"]

import benchmark_naive_c
import numba
import numpy as np
import os
import subprocess
import shlex
import time
import util

class Renderer():
    """Abstract class for the renderers
    """

    def __init__(self, output_directory):
        """Constructor for the class

        This will check if the output frame directory exists (if it doesn't, it
        will create the folder). This will also create a logging file for 
        tracking average framerate.
        """

        self.output_directory = output_directory

        if not os.path.isdir(self.output_directory):
            print("Output directory \"%s\" doesn't existing; creating it..." % 
                self.output_directory)
            os.makedirs(self.output_directory)

        self.log_filename = os.path.join(self.output_directory, 
            "framerate_stats.csv")

        with open(self.log_filename, "w") as f:
            f.write("filename,time spent rendering (sec), fps\n")

    def render(self, path_to_frame):
        """Abstract function for rendering
        """

        print("%s.render(): UNIMPLEMENTED" % (self.__class__.__name__))

class BlurOneStageRGBCompiledRenderer(Renderer):
    """This renderer uses our compiled version of the blur_one_stage filter

    Most likely during this renderer's init, it will call the compiler to 
    compile the python
    """

    def __init__(self, output_directory):
        """Constructor for the class

        We call the super class's constructor, then we call our compiler to 
        compile the Python version of blur_one_stage
        """

        # call parent class's constructor:
        super(BlurOneStageRGBCompiledRenderer, self).__init__(output_directory)

        # # compile Python version of blur_one_stage:
        # if not os.path.isdir("compiler_output/blur_one_stage"):
        #     print("Compiler output directory doesn't exist; creating it...")
        #     os.makedirs("compiler_output/blur_one_stage")

        # starting_dir = os.getcwd()
        # os.chdir("../../compiler")

        # print("changing directory")

        # proc = subprocess.Popen(
        #     shlex.split("python3 compiler.py ../apps/blur_one_stage/blur_one_stage_4channel.py --out-dir ../visualizer/compiler_output/blur_one_stage --no-comparisons"),
        #     stdout=subprocess.PIPE)

        # print("created subprocess")

        # # hack to get live stdout output from the subprocess
        # # from: http://blog.endpoint.com/2015/01/getting-realtime-output-using-python.html
        # output_lines_iterator = iter(proc.stdout.readline, b"")
        # print("created iterator")

        # for line in proc.stdout:
        #     print(line.decode(), end='')

        # os.chdir(starting_dir)
        # exit()

class BlurOneStageRGBNaiveCRenderer(Renderer):
    """This renderer uses the naive C version of blur_one_stage

    Similar to BlurOneStageRGBCompiledRenderer, this will likely compile the C
    code in the constructor of this class
    """

    def __init__(self, output_directory):
        """Constructor for the class

        We call the super class's constructor, then we compile the C code
        """

        super(BlurOneStageRGBNaiveCRenderer, self).__init__(output_directory)
        compilation_result = benchmark_naive_c.compile_c_implementation(
            "../../apps/blur_one_stage/c")

        if not compilation_result:
            print("Error when compiling C code")

class BlurOneStageRGBNumbaRenderer(Renderer):
    """This renderer uses Numba to speed up the Python version of blur_one_stage
    """

    def render(self, path_to_frame):
        """This renders the frame at path_to_frame

        It will time only the rendering (no I/O), log the time, and write the 
        output image to disk.
        """

        input_img = util.read_img(path_to_frame)
        output_img = np.zeros(input_img.shape)

        # load module
        sys.path.append("../../apps/blur_one_stage")
        import blur_one_stage
        sys.path.remove("../../apps/blur_one_stage")

        # this is equivalent to using the @jit function decorator:
        jitted_func = numba.jit(blur_one_stage.gaussian_blur)

        t1 = time.time()
        jitted_func(input_img, output_img)
        t2 = time.time()

        print("Elapsed time: %f" % (t2 - t1))

        output_filename = path_to_frame.split(os.sep)[-1]
        util.write_img(output_img, 
            os.path.join(self.output_directory, output_filename))

        # append timing data to log file:
        with open(self.log_filename, "a") as f:
            f.write("%s, %f, %f\n" % (
                os.path.join(self.output_directory, output_filename), 
                t2 - t1, 
                1./(t2 - t1)))

class BlurOneStageRGBHalideRenderer(Renderer):
    """This renderer uses the Halide version of blur_one_stage
    """