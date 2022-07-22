# ga:
	# python -W ignore ga4mlib.py --POPSIZE 20 --GENS 3 --RENDER=0
ga:
	python -W ignore ga4mscratch.py --POPSIZE 500 --GENS 10 --NWORKERS 100 --RENDER=0

pso:
	python -W ignore pso.py --POPSIZE 500 --GENS 10


clean:
	rm store/ga/*
	
deps:
	pip install -r tools/req.txt
