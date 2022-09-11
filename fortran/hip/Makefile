.PHONY: clean

HIP_COMPILER = gcc
HIPFORT_COMPILER = mpif90
HIPFC = hipfc
HIPCC = hipcc -fPIC
FC = $(HIPFC)
CC = $(HIPCC)
INCLUDE   = 
FCLIBS    = 

build.hip: 
	$(HIPCC) $(INCLUDE) -c heat_kernel.cpp
	$(HIPFC) $(INCLUDE) heat.F90 -c 
	$(HIPFC) $(INCLUDE) -o heat heat.o heat_kernel.o  $(FCLIBS)
clean:
	rm -rf heat *.o *.mod soln* 
