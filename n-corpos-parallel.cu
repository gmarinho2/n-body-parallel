
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASSA 1
#define EPSILON 1E-9

typedef struct vetor
{
    double x, y, z;
} VETOR;

typedef struct posicao
{
    double x, y, z;
} POSICAO;

typedef struct particula
{
    POSICAO coord;
    VETOR forca_sofrida;
    VETOR velocidade;
} PARTICULA;

void inicializador(PARTICULA *particula, int quantidade)
{
    srand(42);
    memset(particula, 0x00, quantidade * sizeof(PARTICULA));
    for (int i = 0; i < quantidade; i++)
    {
        particula[i].coord.x = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
        particula[i].coord.y = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
        particula[i].coord.z = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
    }
}

void printLog(PARTICULA *particles, int quantParticulas, int timestep, char *type)
{
    char path[100] = "../";
    sprintf(path, "Log/%s/Log%d-log.txt", type, timestep);
    fprintf(stdout, "Saving file [%s] ", path);
    fflush(stdout);
    FILE *arquivo = fopen(path, "w+");
    for (int i = 0; i < quantParticulas; i++)
    {
        fprintf(arquivo, "%d \t %.10f %.10f %.10f \t %.10f %.10f %.10f \t %.10f %.10f %.10f \n",
                i,
                particles[i].coord.x, particles[i].coord.y, particles[i].coord.z,
                particles[i].velocidade.x, particles[i].velocidade.y, particles[i].velocidade.z,
                particles[i].forca_sofrida.x, particles[i].forca_sofrida.y, particles[i].forca_sofrida.z);
    }
    fclose(arquivo);
    fprintf(stdout, "[OK]\n");
    fflush(stdout);
}

__device__ 
void atualizaVelocidade(PARTICULA *particula, double dt)
{
    particula->velocidade.x += dt * particula->forca_sofrida.x;
    particula->velocidade.y += dt * particula->forca_sofrida.y;
    particula->velocidade.z += dt * particula->forca_sofrida.z;
}

__device__ 
void atualizaCoordenada(PARTICULA *particula, double dt)
{
    particula->coord.x += dt * particula->velocidade.x;
    particula->coord.y += dt * particula->velocidade.y;
    particula->coord.z += dt * particula->velocidade.z;
}

__global__ 
void simulacao(PARTICULA *particula, int quantParticulas, double dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < quantParticulas)
    {

        for (int j = 0; j < quantParticulas; j++)
        {
            if (i != j)
            {
                double dx = particula[i].coord.x - particula[j].coord.x;
                double dy = particula[i].coord.y - particula[j].coord.y;
                double dz = particula[i].coord.z - particula[j].coord.z;
                double dist = dx * dx + dy * dy + dz * dz + EPSILON;
                double distSqrRoot = sqrt(dist);
                double invDist = 1.0 / pow(distSqrRoot, 2);

                particula[i].forca_sofrida.x += dx * invDist;
                particula[i].forca_sofrida.y += dy * invDist;
                particula[i].forca_sofrida.z += dz * invDist;
            }
        }
        atualizaVelocidade(&particula[i], dt);
        __syncthreads();

        atualizaCoordenada(&particula[i], dt);
        __syncthreads();
    }
}

int main(int ac, char **av)
{
    int timesteps = atoi(av[1]), quantParticulas = atoi(av[2]), flagSave = atoi(av[3]);

    clock_t t;
    t = clock();

    char logFile[1024];
    double dt = 0.01;
    PARTICULA *particulas = NULL;
    PARTICULA *d_particula;

    strcpy(logFile, av[4]);

    fprintf(stdout, "\nSistema de particulas P2P(particula a particula)\n");
    fprintf(stdout, "Memória utilizada %lu bytes \n", quantParticulas * sizeof(PARTICULA));
    fprintf(stdout, "Arquivo %s \n", logFile);

    particulas = (PARTICULA *)malloc(quantParticulas * sizeof(PARTICULA));
    assert(particulas != NULL);

    inicializador(particulas, quantParticulas);

    int block_size = 1024;
    int grid_size = ((quantParticulas + block_size - 1) / block_size);

    cudaMalloc(&d_particula, sizeof(PARTICULA) * quantParticulas);

    cudaMemcpy(d_particula, particulas, sizeof(PARTICULA) * quantParticulas, cudaMemcpyHostToDevice);

    for (int j = 0; j < timesteps; j++)
    {
        simulacao<<<grid_size, block_size>>>(d_particula, quantParticulas, dt);
    }

    cudaMemcpy(particulas, d_particula, sizeof(PARTICULA) * quantParticulas, cudaMemcpyDeviceToHost);

    cudaFree(d_particula);

    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC;
    fprintf(stdout, "Tempo gasto: %lf (s) \n\n", time_taken);

    FILE *log = fopen(logFile, "a+");
    assert(log != NULL);
    fprintf(log, "Timesteps: %d\nNúmero de Particulas: %d\nMemória em bytes:%lu\nTempo em segundos:%lf\n-----------------------------\n", timesteps, quantParticulas, quantParticulas * sizeof(particulas), time_taken);
    fclose(log);

    if (flagSave == 1)
        printLog(particulas, quantParticulas, timesteps, "ParallelCuda");
    free(particulas);
}
