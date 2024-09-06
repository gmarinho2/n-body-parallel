#include "headers/Definitions.h"

void simulacao(PARTICULA* particula, int quantParticulas, int timesteps, double dt)
{
    double forca;
    double dx, dy, dz;
    int i, j, k;
   

    for(i = 0; i < timesteps; i++)
    {
        omp_set_num_threads(12);
        
        #pragma omp parallel shared(particula, quantParticulas, dt)
        {
            #pragma omp for
            for(j = 0; j < quantParticulas; j++)
            {
                for(k = 0; k < quantParticulas; k++)
                {
                    if(j != k)
                    {
                        dx = 0.0f, dy = 0.0f, dz = 0.0f;
                        forca = calculaForca(particula[j], particula[k], &dx, &dy, &dz);
                        particula[j].forca_sofrida.x = dx * forca;
                        particula[j].forca_sofrida.y = dy * forca;
                        particula[j].forca_sofrida.z = dz * forca;
                    }
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (j=0; j<quantParticulas; j++)
            {
                atualizaVelocidade(&particula[j], dt);
                atualizaCoordenada(&particula[j], dt);
            }
        }
        return;
    }
}


int main (int ac, char **av)
{
    int timesteps = atoi(av[1]), quantParticulas = atoi(av[2]), flagSave = atoi(av[3]);

    clock_t t;
    t = clock();

    char logFile[1024];
    double       dt        =  0.01f;
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
          printLog(particulas, quantParticulas, timesteps, "Parallel");
    free(particulas);
}
