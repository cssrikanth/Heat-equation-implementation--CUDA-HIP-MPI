main:
	nvfortran -Mcuda heat.F90 -o heat
init:
	python3 init.py
out:
	python3 out.py
clean:
	rm -rf heat int.dat soln.dat *.mod
