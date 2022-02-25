mpi:
	mpif90 heat.F90 -o heat
mpicuda-na:
	mpif90 -Mcuda=ccall -DUSE_CUDA -DNO_AWARE heat.F90 -o heat
mpicuda-a:
	mpif90 -Mcuda=ccall -DUSE_CUDA heat.F90 -o heat
out:
	python3 out.py
clean:
	rm -rf heat soln* *.mod *.eps
