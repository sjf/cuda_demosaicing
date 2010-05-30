# Makefile
SUFFIXES = .cu .c .cpp

CUDA_INSTALL_PATH ?= /usr/local/cuda/
CUDA_EXTRA_LIB ?=  $(HOME)/cuda/

all : proj

APP = proj
CC  = gcc
CXX = g++
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
LINK = g++ -fPIC

ARCH = -D_LINUX_ -D_GNU_SOURCE
DEBUG = -g -D_DEBUG_
PERF = 
LIBS = -lm
#INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc
COMMONDIR = common
#LINK_FLAGS = -rdynamic

emu ?= 1

# Kingfisher does not need to use emulation
ifeq ($(shell hostname), kingfisher)
  emu = 0
endif

PERF = -O3
TEST =
ifeq ($(test),1)
  #Turn on optimisation, set _TEST flag
  PERF = -O3
  TEST = -D_TEST
  DEBUG = 
endif
ifeq ($(dbg),1)
  #turn off optimisation for debugging
  PERF = 
else
  PREF = -O3 
endif

ifeq ($(shell uname -m), x86_64)
  BIN_DIR = obj64
  CU_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -L$(CUDA_EXTRA_LIB)/lib64  -lcutil_x86_64D 
else
  BIN_DIR = obj
  CU_LIBS = -L$(CUDA_INSTALL_PATH)/lib  -L$(CUDA_EXTRA_LIB)/lib -lcutil_i386D
endif

ifeq ($(emu), 1)
  CU_LIBS := $(CU_LIBS) -D__CUDAEMU__ -lcudartemu
  EMU_FLAGS = -deviceemu
else
  CU_LIBS := $(CU_LIBS) -lcudart
endif

ifeq ($(prof),1) 
 PROF = -pg 
endif



CU_INCLUDES = $(INCLUDES) -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc

CFLAGS = -Wshadow -std=c99 -Wall $(DEBUG) $(ARCH) $(PERF) $(TEST) $(INCLUDES) $(LIBS)

CXXFLAGS = -Wall  $(DEBUG) $(ARCH) $(PERF) $(TEST) $(CU_INCLUDES) $(LIBS) $(CU_LIBS)

# Use -ptx to generate assembly
NVCCFLAGS = $(DEBUG) $(PERF) $(TEST) $(EMU_FLAGS) --ptxas-options=-v 

LINK_FLAGS := $(LINK_FLAGS) $(CXXFLAGS)

#------------------------------------------------------------------------------
# base src files (except main)
BASE_SRCS = 
#bayer.c image.c base.c main.c util.c colorspace.c

CPP_SRCS = cuda_utils.cpp bayer.cpp image.cpp base.cpp util.cpp colorspace.cpp ahd_mask.cpp bilinear.cpp ahd.cpp mask.cpp ahd_mask2.cpp ahd_mask3.cpp ahd_mask4.cpp

CU_SRCS = ahd.cu bilinear.cu

BASE_OBJS = $(patsubst %.c,$(BIN_DIR)/%.o,$(BASE_SRCS))

CPP_OBJS = $(patsubst %.cpp,$(BIN_DIR)/%.cpp.o,$(CPP_SRCS))

CU_OBJS = $(patsubst %.cu,$(BIN_DIR)/%.cu.o,$(CU_SRCS))

#------------------------------------------------------------------------------
# Patterns:

#$(BIN_DIR)/ahd.cu.o : ahd.cu ahd_kernel.cu
#	$(NVCC) $(NVCCFLAGS) $(CU_INCLUDES) $(CU_LIBS) -o $@ -c $*.cu

$(BIN_DIR)/%.cu.o : *.cu
	$(NVCC) $(NVCCFLAGS) $(CU_INCLUDES) $(CU_LIBS) -o $@ -c $*.cu

$(BIN_DIR)/%.cpp.o : %.cpp
	$(CXX) $(CXXFLAGS) $(PROF) -o $@ -c $<

$(BIN_DIR)/%.o : %.c
	$(CC) $(CFLAGS) $(INCLUDES) $(PROF) $(LIBS) -o $@ -c $<

proj : cleantemps $(BASE_OBJS) $(CPP_OBJS) $(CU_OBJS) main.cpp
	$(LINK) $(LINK_FLAGS) $(PROF) -o $@ $(BASE_OBJS) $(CPP_OBJS) $(CU_OBJS) main.cpp $(LIBS) $(CU_LIBS) 
compare : $(BASE_OBJS) $(CPP_OBJS)  main_compare.cpp
	$(LINK) $(LINK_FLAGS) -o zipcompare $^ $(LIBS) $(CU_LIBS) 
 
bilinear_tests : bilinear_tests.cu
	$(NVCC) $(NVCCFLAGS) $(CU_INCLUDES) $(CU_LIBS) -o bilinear_tests bilinear_tests.cu  $(LIBS) $(CU_LIBS)
#bilinear_tests : bilinear_tests.cu
#------------------------------------------------------------------------------

DCRAW_SRC = dcraw.c
DCRAW_APP = dcraw

$(DCRAW_APP): $(DCRAW_SRC)
	$(CC) $(DEBUG) $(PROF) $< -o $@ -O4 -lm -DNO_JPEG -DNO_LCMS 2>&1

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY : clean clean-deps dist cleantemps

# Avoid the awful consequences of trying to compile these temp files
cleantemps:
	-rm *.cu.cpp 2>/dev/null

clean-byproducts: cleantemps
	rm -f *.o *~ core.* *.dump *.tar tags *.a *.ii *.cudafe1.* *.cu.c $(BIN_DIR)/* img/*.out.ppm img/interpolation*.ppm img/pre_noise.ppm img/score_*.ppm img/homo_map*.ppm img/direction.ppm img/interp_g_*.ppm *.sibling* img/edges.ppm img/mask.ppm img/bilinear_grayscale.ppm

clean: clean-byproducts
	rm -f $(APP) $(DCRAW_APP) bilinear_tests

clean-deps:
	rm -f .*.d

minclean:
	echo $(host)
	rm -f *.cu.o $(BIN_DIR)/*.cu.o

ETAGS=etags
tags:
	rm -f TAGS
	find . -name '*.h' -o -name '*.c' -print0 | xargs --null $(ETAGS) --append

