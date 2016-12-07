CC=gcc

OS=$(shell uname)

ifeq ($(OS),Darwin) # Mac
	CC=gcc-5
endif

CFLAGS= \
	-I../../../compiler -Dreal=double \
	-fopenmp

GCC_OPT_FLAGS=-O3
MATH_FLAGS=-lm

all: composite.c
	$(CC) $(GCC_OPT_FLAGS) $(CFLAGS) composite.c $(MATH_FLAGS)
