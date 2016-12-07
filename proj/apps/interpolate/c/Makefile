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

all: interpolate_new.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) interpolate_new.c $(MATH_FLAGS)
	
orig: interpolate.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) interpolate.c $(MATH_FLAGS)