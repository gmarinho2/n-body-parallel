#include "headers/Definitions.h"

void simulacao(PARTICULA *particula, int quantParticulas, int timesteps, double dt)
{
    for (int i = 0; i < timesteps; i++)
    {
        #pragma omp parallel shared(particula, quantParticulas, dt)
        {
            #pragma omp for
            for (int j = 0; j < quantParticulas; j++)
            {
                for (int k = 0; k < quantParticulas; k++)
                {
                    if (j != k)
                    {
                        double forca;
                        double dx, dy, dz;
                        dx = 0.0f, dy = 0.0f, dz = 0.0f;

                        forca = calculaForca(particula[j], particula[k], &dx, &dy, &dz);

                        particula[j].forca_sofrida.x += dx * forca;
                        particula[j].forca_sofrida.y += dy * forca;
                        particula[j].forca_sofrida.z += dz * forca;
                    }
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (int j = 0; j < quantParticulas; j++)
            {
                atualizaVelocidade(&particula[j], dt);
                atualizaCoordenada(&particula[j], dt);
            }
        }
    }
}

int main(int ac, char **av)
{
    int timesteps = atoi(av[1]), quantParticulas = atoi(av[2]), flagSave = atoi(av[3]);

    clock_t t;
    t = clock();

    char logFile[1024];
    double dt = 0.01f;
    PARTICULA *particulas = NULL;

    strcpy(logFile, av[4]);

    fprintf(stdout, "\nSistema de particulas P2P(particula a particula)\n");
    fprintf(stdout, "Memória utilizada %lu bytes \n", quantParticulas * sizeof(PARTICULA));
    fprintf(stdout, "Arquivo %s \n", logFile);

    particulas = (PARTICULA *)aligned_alloc(ALING, quantParticulas * sizeof(PARTICULA));
    assert(particulas != NULL);

    inicializador(particulas, quantParticulas);

    simulacao(particulas, quantParticulas, timesteps, dt);

    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC;
    fprintf(stdout, "Tempo gasto: %lf (s) \n\n", time_taken);

    FILE *log = fopen(logFile, "a+");
    assert(log != NULL);
    fprintf(log, "Timesteps: %d\nNúmero de Particulas: %d\nMemória em bytes:%lu\nTempo em segundos:%lf\n-----------------------------\n", timesteps, quantParticulas, quantParticulas * sizeof(particulas), time_taken);
    fclose(log);

    if (flagSave == 1)
        printLog(particulas, quantParticulas, timesteps, "Parallel");
    free(particulas);
}
