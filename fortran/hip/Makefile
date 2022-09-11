.PHONY: build.hip clean

HIP_COMPILER = gcc
HIPFORT_COMPILER = mpif90
HIPFC = /nfs_home/schowdad/hipfort/INSTALL/bin/hipfc
HIPCC = hipcc -fPIC
FC = $(HIPFC)
CC = $(HIPCC)

build.hip: 
	$(HIPCC) -c heat_kernel.cpp
	$(HIPFC) heat.F90 -c 
	$(HIPFC) -o heat heat.o heat_kernel.o 
clean:
	rm -rf heat *.o *.mod soln* 
