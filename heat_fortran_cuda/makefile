main:
	pgf90 -Mcuda=cuda10.1 heat.cuf -o heat
managed:
	pgf90 -Mcuda=cuda10.1 heat_managed.cuf -o heat
init:
	python3 init.py
out:
	python3 out.py
clean:
	rm heat int.dat soln.dat *.mod
