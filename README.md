# VizGen: Accelerating Visual Computing Prototypes in Dynamic Languages

This is the source code for the compiler associated with the paper "VizGen: Accelerating Visual Computing Prototypes in Dynamic Languages" by Yuting Yang, Sam Prestwood, Connelly Barnes, ACM SIGGRAPH Asia 2016.

The [project page](http://www.cs.virginia.edu/~connelly/project_pages/vizgen/) has the paper, video, and other materials. The goal of this project is to accelerate visual programs written in Python.

# Platforms

We have developed the compiler on Linux and Mac. For Windows machines we simply run it in a Linux VM.

# Installation

We suggest to use Anaconda Python which has some dependencies already installed.

Currently requires Python 3 version between 3.0 and 3.4 due to the Astor library not presently supporting Python 3.5.

Required dependencies (install libraries via [PIP](http://pip.readthedocs.org/en/stable/installing/) or [conda](http://conda.pydata.org/docs/get-started.html) for Anaconda Python, except z3, which must be installed from source):
 - Cython
 - Scikit-image
 - Numpy
 - Astor
 - [z3](https://github.com/Z3Prover/z3) (build with Python support: use --python argument to mk_make.py)
 - On Mac, GCC is needed for OpenMP support (use `brew install gcc5` to install) 

Optional dependencies:
 - For the arcade app ("pacman"), Pygame is required. On Mac this can be installed using [Homebrew](http://brew.sh/):

    > brew install sdl sdl_image sdl_mixer sdl_ttf portmidi
    >
    > sudo pip install hg+http://bitbucket.org/pygame/pygame

 - For optional speed comparisons, Numba and PyPy need to be installed

# Usage

How to run an example application with ordinary Python:

    > cd proj/apps/mandelbrot
    > python mandelbrot.py

How to run an example application with our autotuning compiler (takes minutes to 10s of minutes to complete):

    > cd proj/compiler
    > python compiler.py ../apps/mandelbrot/mandelbrot.py

How to get help for the compiler:

    > cd proj/compiler
    > python compiler.py -h

How to run unit tests:

    > cd proj/compiler
    > python some_test.py       # Run most unit tests (takes about 10 minutes), or...
    > python all_test.py        # Run all unit tests (takes about 30 minutes).

# Security

The compiler presently uses the eval() function in various places. Therefore, do not compile programs that may contain potentially untrustworthy (e.g. malicious) code or strings. See more about the [security implications of eval](http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html).

# Bug reports and patches

Please submit bug reports via the GitHub issue tracking. Code submitted to GitHub (e.g. commits, pull requests) should pass all unit tests, and you agree to license code you submit to us under the below (MIT) license.

# License

This project is licensed under the [MIT license](https://opensource.org/licenses/MIT):

Copyright (c) 2015-2016 University of Virginia, Yuting Yang, Sam Prestwood, Connelly Barnes, and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

# Frequently Asked Questions

## What are some limitations?

This is a research compiler. It may not succeed for all input programs, scale to long input programs, and currently it compiles only one module at a time. It uses autotuning so it may take a while to finish for a given input program. It presently supports only CPU.

There is currently not support for accelerating user-defined classes (only functions),  type inference may not work in all cases, and some scenarios may not be optimized (e.g. code that returns matrices currently is not optimized).

## How can I make my module work with the compiler?

Implement a `test()` method in your module, which when run with no arguments, calls all of the functions that are to be sped up (it can also implement unit tests if you like, but this is not required). See the numerous example applications in the `proj/apps` directory (particularly simple applications are `mandelbrot` and `blur_two_stage`). 

## I called a builtin or library function X and my code is not fast. How do I make it fast?

One possibility is that type inference did not succeed. When the compiler is run you should see a printout of inferred types for each variable. If a variable has type `object` then it did not have type inferred. You can either implement step (3) below to make type inference succeed, or work around this in your program by using [MyPy](http://mypy-lang.org/)-style type annotations (see `proj/compiler/test_programs/type_annotation.py` for an example of type annotations).
 
The full solution for making an arbitrary library function X efficient is to do all three of (the source files referred to are all in the directory `proj/compiler`):
 
1. Implement a C (Cython) variant of your function in `macro_funcs.pyx`, following the other examples in that file. Alternatively, if the function takes vectors of constant length, then you can "template" your function in a manner similar to the other functions in `macro_funcs_templated.py`.
2. Add a translation "macro" to `macros.py`, which uses pattern matching to translate Python calls to corresponding C implementations. Specifically, add it to the `macros` list in `macros.py`, based on examples of other macros.
3. Make type inference work through your function, by adding a function named `typefunc_X` to `typefuncs.py`, following the examples of other type functions (or `typemethod_X` if you implemented a method).

# Code style

Please follow the [Google Python Style Guide](https://google-styleguide.googlecode.com/svn/trunk/pyguide.html). In particular, indent using 4 spaces. If your editor is producing tabs then change your editor settings to use spaces.
