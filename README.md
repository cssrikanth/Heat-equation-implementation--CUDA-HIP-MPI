# 2D Heat Equation
Simple 2D Heat equation implementation using different programming methods. Below are instructions for fortran compilation.

# General Instructions
* Go into the code directory
* Edit the input.dat file
* The values in input.dat correspond to: grid size, sigma, nu, domain length, time iterations
* Read below for specific compilation instructions

# For fortran/serial
* You need gfortran to compile the code

# For fortran/cuda_cuf and fortran/cuda_kernel
* You need nvfortran to compile code

# For fortran/mpi+cuda
* You need nvfortran and mpif90 from Openmpi
* Please check the makefile in this directory for specific compilation options which include both CUDA Aware and CUDA Not aware MPI implementations

# For fortran/hip
* HIP implementation works for both CUDA and AMD GPUs
* For installing hip, I recommend using spack and installing the following packages:
>spack install hip; spack install hipfort
* For more instructions to change the GPU platforms, please check this [repo](https://github.com/ROCmSoftwarePlatform/hipfort)


***Note: I am almost sure that the implementations should work. If in case you find any bugs, please send me a pull request.***
