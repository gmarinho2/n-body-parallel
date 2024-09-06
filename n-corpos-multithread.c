#include "auxiliares.c"
#include <omp.h>


int main (int ac, char **av)
{
    int timesteps = atoi(av[1]), quantParticulas = atoi(av[2]), flagSave = atoi(av[3]);

    clock_t t;
    t = clock();

    char logFile[1024];
    float       dt        =  0.01f;
    PARTICULA *particulas = NULL;

    strcpy(logFile, av[4]);

    fprintf(stdout, "\nSistema de particulas P2P(particula a particula)\n");
    fprintf(stdout, "Memória utilizada %lu bytes \n", quantParticulas * sizeof(PARTICULA));
    fprintf(stdout, "Arquivo %s \n", logFile);

    particulas = (PARTICULA *) aligned_alloc(ALING, quantParticulas * sizeof(PARTICULA));
    assert(particulas != NULL);

    inicializador(particulas, quantParticulas);
    simulacao(particulas, quantParticulas, timesteps, dt);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    fprintf(stdout, "Elapsed time: %lf (s) \n", time_taken);

    FILE *ptr = fopen(logFile, "a+");
    assert(ptr != NULL);
    fclose(ptr);

    if (flagSave == 1)
          printLog(particulas, quantParticulas, timesteps);

    free(particulas);
}
