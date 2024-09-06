#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define MASSA 1
#define EPSILON 1E-9
#define ALING 1024

typedef struct vetor
{
    float x, y, z;
} VETOR;

typedef struct posicao
{
    float x, y, z;
} POSICAO;

typedef struct particula
{
    POSICAO coord;
    VETOR forca_sofrida;
    VETOR velocidade;
} PARTICULA;

float calculaDistancia(PARTICULA a, PARTICULA b)
{
    float distancia;
    distancia = sqrt(pow(a.coord.x - b.coord.x, 2) + pow(a.coord.y - b.coord.y, 2) + pow(a.coord.z - b.coord.z, 2));
    return distancia;
}

void atualizaVelocidade(PARTICULA* particula, float dt)
{
    particula->velocidade.x += dt * particula->forca_sofrida.x;
    particula->velocidade.y += dt * particula->forca_sofrida.y;
    particula->velocidade.z += dt * particula->forca_sofrida.z;
}

void atualizaCoordenada(PARTICULA* particula, float dt)
{
    particula->coord.x += dt * particula->velocidade.x;
    particula->coord.y += dt * particula->velocidade.y;
    particula->coord.z += dt * particula->velocidade.z;
}

void inicializador(PARTICULA *particula, int quantidade)
{
    srand(42);
    memset(particula, 0x00, quantidade * sizeof(PARTICULA));
    for (int i = 0; i < quantidade ; i++){
        particula[i].coord.x =  2.0 * (rand() / (float)RAND_MAX) - 1.0;
        particula[i].coord.y =  2.0 * (rand() / (float)RAND_MAX) - 1.0;
        particula[i].coord.z =  2.0 * (rand() / (float)RAND_MAX) - 1.0;
      }
}

double calculaForca(PARTICULA a, PARTICULA b, double* dx, double* dy, double* dz)
{
    float G = 1; //constante gravitacional
    float distancia = calculaDistancia(a, b);
    float intensidadeForca = G * MASSA * MASSA / distancia + EPSILON; //para nao dar zero
    
    *dx = a.coord.x - b.coord.x; // influencia da força no 
    *dy = a.coord.y - b.coord.y; // vetor decomposto, em cada eixo
    *dz = a.coord.z - b.coord.z;

    return intensidadeForca;
}

void printLog(PARTICULA *particles, int quantParticulas, int timestep)
{
    char fileName[128];
    sprintf(fileName, "%s-%d-log.txt", __FILE__,  timestep);
    fprintf(stdout, "Saving file [%s] ", fileName); fflush(stdout);
    FILE *ptr = fopen(fileName, "w+");
    for(int i = 0; i < quantParticulas; i++){
        fprintf(ptr, "%d \t %.10f %.10f %.10f \t %.10f %.10f %.10f \t %.10f %.10f %.10f \n",
         i,  
            particles[i].coord.x, particles[i].coord.y, particles[i].coord.z,

            particles[i].velocidade.x, particles[i].velocidade.y, particles[i].velocidade.z, 
            
            particles[i].forca_sofrida.x, particles[i].forca_sofrida.y, particles[i].forca_sofrida.z);
    }
    fclose(ptr);
    fprintf(stdout, "[OK]\n"); fflush(stdout);
}

void simulacao(PARTICULA* particula, int quantParticulas, int timesteps, double dt)
{
    int i, j, k;
    for(i = 0; i < timesteps; i++)
    {
        for(j = 0; j < quantParticulas; j++)
        {
            for(k = 0; k < quantParticulas; k++)
            {
                if(j != k)
                {
                    double dx = 0.0f, dy = 0.0f, dz = 0.0f;
                    double forca = calculaForca(particula[j], particula[k], &dx, &dy, &dz);
                    particula[j].forca_sofrida.x = dx * forca;
                    particula[j].forca_sofrida.y = dy * forca;
                    particula[j].forca_sofrida.z = dz * forca;
                }
            }
        }

        for (j=0; j<quantParticulas; j++)
        {
            atualizaVelocidade(&particula[j], dt);
            atualizaCoordenada(&particula[j], dt);
        }
    }
    return;
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
          printLog(particulas, quantParticulas, timesteps);
    free(particulas);
}
