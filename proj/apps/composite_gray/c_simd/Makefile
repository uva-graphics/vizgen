CC=gcc

OS=$(shell uname)

ifeq ($(OS),Darwin) # Mac
	CC=gcc-5
endif

CFLAGS= \
	-I../../../compiler \
	-msse3 \
	-mavx \
	-mtune=native \
	-fno-tree-vectorize

GCC_OPT_FLAGS=-O3
MATH_FLAGS=-lm

all: vectorize_x.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) vectorize_x.c $(MATH_FLAGS)
