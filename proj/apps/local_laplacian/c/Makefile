CC=gcc

OS=$(shell uname)

ifeq ($(OS),Darwin) # Mac
	CC=gcc-5
endif

CFLAGS=-Dreal=double \
	-I../../../compiler \
	-fopenmp

GCC_OPT_FLAGS=-O3
MATH_FLAGS=-lm

all: local_laplacian_fuse.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) local_laplacian_fuse.c $(MATH_FLAGS)
	
orig: local_laplacian.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) local_laplacian.c $(MATH_FLAGS)