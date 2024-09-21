sequencial:
	gcc n-corpos-sequencial.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/sequencial.exec -lm
	time ./bin/sequencial.exec 100 1000 1 "Log/Sequencial/LogSequencial.csv"

omp_parallel:
	gcc n-corpos-multithread.c domain/Calculate.c domain/Log.c domain/Update.c -o bin/parallel.exec -lm -fopenmp
	time ./bin/parallel.exec 100 1000 1 "Log/Parallel/LogParallel.csv"

cuda_parallel: 
	nvcc n-corpos-parallel.cu -o bin/cuda_parallel.exec
	time ./bin/cuda_parallel.exec 100 1000 1 "Log/ParallelCuda/LogParallelCuda.csv"
