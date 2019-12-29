# CUDA-Python-Fortran-Heat-equation
Simple 2D Heat equation implementation in CUDA Python and CUDA Fortran

# For CUDA Fortran
* You need a pgi complier to run this code
* Go into heat_fortran_cuda and edit the input.dat file
* The values in input.dat correspond to: grid size, sigma, nu, domain length, time step
* type `make main` to compile the code with device memory and `make managed` to compile the code with managed memory
* run the code by typing `./heat` 
* type `make init` to see the initial profile and `make out` to see the final profile
