sequencial:
	gcc n-corpos-sequencial.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/sequencial -lm
	time ./bin/sequencial 100 1000 1 "Log/Sequencial/LogSequencial"

multi:
	gcc n-corpos-multithread.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/multi -lm -fopenmp
	time ./bin/multi 100 1000 1 "Log/Parallel/LogParallel"