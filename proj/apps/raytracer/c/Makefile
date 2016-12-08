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

all: raytracer.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) raytracer.c $(MATH_FLAGS)
