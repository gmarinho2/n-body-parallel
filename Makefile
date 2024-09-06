sequencial:
	gcc n-corpos-sequencial.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/sequencial -lm
	time ./bin/sequencial 100 1000 1 "Log/Sequencial/LogSequencial"

parallel:
	gcc n-corpos-multithread.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/parallel -lm -fopenmp
	time ./bin/parallel 100 1000 1 "Log/Parallel/LogParallel"