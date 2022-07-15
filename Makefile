default:
	python -W ignore simplepolicy.py --POPSIZE 500 --GENS 3 --NWORKERS 100 --RENDER=0

deps:
	pip install -r tools/req.txt
