# Makefile for VEGAS

# C compiler and flags
CC = g77
CFLAGS = -std=gnu99 -pedantic -Wall -g -I/export/home/spec-hpc-01/jayanth/fftw3/include -I/export/home/spec-hpc-01/jayanth/pgplot
# NVCC compiler and flags
NVCC = nvcc
NVFLAGS = --ptxas-options=-v --compiler-bindir=/usr/bin/gcc -g

# linker options
LFLAGS_FFTW3 = -L/export/home/spec-hpc-01/jayanth/fftw3/lib -lfftw3f
LFLAGS_CUFFT = -L/opt/vegas/cuda/lib64 -lcufft
LFLAGS_PGPLOT = -L/export/home/spec-hpc-01/jayanth/pgplot -lpgplot -lcpgplot
LFLAGS_MATH = -lm

# bin directory
BINDIR = ./bin

all: s6_gpu_standalone s6_gpu_standalone.tex s6_gentestdata tags

# SERENDIP 6
s6_gpu_standalone: s6_gpu_standalone.cu s6_gpu_standalone.h
	$(NVCC) $(NVFLAGS) s6_gpu_standalone.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -lX11 -c
	$(CC) $(NVFLAGS) s6_gpu_standalone.o -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -lX11 -o $(BINDIR)/$@

s6_gpu_standalone.tex: s6_gpu_standalone.tex.cu s6_gpu_standalone.tex.h
	$(NVCC) $(NVFLAGS) s6_gpu_standalone.tex.cu -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -lX11 -c
	$(CC) $(NVFLAGS) s6_gpu_standalone.tex.o -lm $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -lX11 -o $(BINDIR)/$@

# test-data-generation program
s6_gentestdata: s6_gentestdata.c
	$(CC) $(CFLAGS) s6_gentestdata.c $(LFLAGS_MATH) -o $(BINDIR)/$@

tags: *.c *.cu *.h
	ctags --language-force=C s6*

clean:
	rm -f *.o

