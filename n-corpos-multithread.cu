
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASSA 1
#define EPSILON 1E-9
#define ALING 1024

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
    for (int i = 0; i < quantidade ; i++){
        particula[i].coord.x =  2.0 * (rand() / (double)RAND_MAX) - 1.0;
        particula[i].coord.y =  2.0 * (rand() / (double)RAND_MAX) - 1.0;
        particula[i].coord.z =  2.0 * (rand() / (double)RAND_MAX) - 1.0;
      }
}

void printLog(PARTICULA *particles, int quantParticulas, int timestep, char* type)
{
    char path[100] = "../";
    sprintf(path, "Log/%s/Log%d-log.txt", type, timestep);
    fprintf(stdout, "Saving file [%s] ", path); fflush(stdout);
    FILE *arquivo = fopen(path, "w+");
    for(int i = 0; i < quantParticulas; i++){
        fprintf(arquivo, "%d \t %.10f %.10f %.10f \t %.10f %.10f %.10f \t %.10f %.10f %.10f \n",
         i,  
            particles[i].coord.x, particles[i].coord.y, particles[i].coord.z,

            particles[i].velocidade.x, particles[i].velocidade.y, particles[i].velocidade.z, 
            
            particles[i].forca_sofrida.x, particles[i].forca_sofrida.y, particles[i].forca_sofrida.z);
    }
    fclose(arquivo);
    fprintf(stdout, "[OK]\n"); fflush(stdout);
}

__global__ void simulacao(PARTICULA* particula, int quantParticulas, int timesteps, double dt);

__device__ void calcula_forca(PARTICULA* particula, int quantParticulas, double dt, int tid) {
    int j, k;
    double dx, dy, dz;

    for(j = 0; j < quantParticulas; j++) {

        if(j != tid)
        {   
            dx = 0.0f, dy = 0.0f, dz = 0.0f;
            double G = 1; //constante gravitacional
            double distancia = sqrt(pow(particula[j].coord.x - particula[tid].coord.x, 2) + pow(particula[j].coord.y - particula[tid].coord.y, 2) + pow(particula[j].coord.z - particula[tid].coord.z, 2));
            double forca = G * MASSA * MASSA / distancia + EPSILON; //para nao dar zero

            dx = particula[j].coord.x - particula[tid].coord.x; // influencia da força no 
            dy = particula[j].coord.y - particula[tid].coord.y; // vetor decomposto, em cada eixo
            dz = particula[j].coord.z - particula[tid].coord.z;

            particula[j].forca_sofrida.x = dx * forca;
            particula[j].forca_sofrida.y = dy * forca;
            particula[j].forca_sofrida.z = dz * forca;
        }
    }

    for (j=0; j<quantParticulas; j++)
    {
        particula[j].velocidade.x += dt * particula[j].forca_sofrida.x;
        particula[j].velocidade.y += dt * particula[j].forca_sofrida.y;
        particula[j].velocidade.z += dt * particula[j].forca_sofrida.z;
        
        particula[j].coord.x += dt * particula[j].velocidade.x;
        particula[j].coord.y += dt * particula[j].velocidade.y;
        particula[j].coord.z += dt * particula[j].velocidade.z;
    }
}

__global__ void simulacao(PARTICULA* particula, int quantParticulas, int timesteps, double dt)
{
    int i, j;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(i = 0; i < timesteps; i++)
    {
        calcula_forca(particula, quantParticulas, dt, tid);
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
    PARTICULA* d_particula;

    strcpy(logFile, av[4]);

    fprintf(stdout, "\nSistema de particulas P2P(particula a particula)\n");
    fprintf(stdout, "Memória utilizada %lu bytes \n", quantParticulas * sizeof(PARTICULA));
    fprintf(stdout, "Arquivo %s \n", logFile);

    particulas = (PARTICULA *) aligned_alloc(ALING, quantParticulas * sizeof(PARTICULA));
    assert(particulas != NULL);

    inicializador(particulas, quantParticulas);

    int block_size = 256;
    int grid_size = ((quantParticulas + block_size) / block_size);


    cudaMalloc((void**)&d_particula, sizeof(PARTICULA) * quantParticulas);
    cudaMemcpy(d_particula, particulas, sizeof(PARTICULA) * quantParticulas, cudaMemcpyHostToDevice);

    simulacao<<<grid_size,block_size>>>(d_particula, quantParticulas, timesteps, dt);

    cudaMemcpy(particulas, d_particula, sizeof(PARTICULA) * quantParticulas, cudaMemcpyDeviceToHost);
    cudaFree(d_particula);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    fprintf(stdout, "Tempo gasto: %lf (s) \n\n", time_taken);

    FILE *log = fopen(logFile, "a+");
    assert(log != NULL);
    fprintf(log, "Timesteps: %d\nNúmero de Particulas: %d\nMemória em bytes:%lu\nTempo em segundos:%lf\n-----------------------------\n",timesteps,quantParticulas,quantParticulas * sizeof(particulas), time_taken);
    fclose(log);

    if (flagSave == 1)
          printLog(particulas, quantParticulas, timesteps, "ParallelCuda");
    free(particulas);
}

