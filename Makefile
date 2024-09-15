sequencial:
	gcc n-corpos-sequencial.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/sequencial -lm
	time ./bin/sequencial 2 10 1 "Log/Sequencial/LogSequencial"

omp_parallel:
	gcc n-corpos-multithread.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/parallel -lm -fopenmp
	time ./bin/parallel 2 10 1 "Log/Parallel/LogParallel"

cuda_parallel: 
	nvcc n-corpos-multithread.cu -o bin/cuda_parallel
	time ./bin/cuda_parallel 2 10 1 "Log/ParallelCuda/LogParallelCuda"
