sequencial:
	gcc n-corpos-sequencial.c -o sequencial -lm
	time ./sequencial 100 1000 1 "log_sequencial"

multi:
	gcc n-corpos-multithread.c -o multi -lm -fopenmp
	time ./multi 100 1000 1 "log_multi"