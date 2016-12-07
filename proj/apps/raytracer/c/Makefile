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

all: raytracer_short_simplified.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) raytracer_short_simplified.c $(MATH_FLAGS)
