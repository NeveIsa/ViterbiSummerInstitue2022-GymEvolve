gascratch:
	python -W ignore ga4mscratch.py --POPSIZE 500 --GENS 3 --NWORKERS 100 --RENDER=0

pso:
	python -W ignore pso.py --POPSIZE 100 --GENS 7

deps:
	pip install -r tools/req.txt
