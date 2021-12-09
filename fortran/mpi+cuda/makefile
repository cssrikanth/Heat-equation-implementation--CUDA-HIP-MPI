mpi:
	mpif90 heat.F90 -o heat
mpicuda:
	mpif90 -Mcuda=ccall -DUSE_CUDA -DNO_AWARE heat.F90 -o heat
out:
	python3 out.py
clean:
	rm heat soln* *.mod *.eps
