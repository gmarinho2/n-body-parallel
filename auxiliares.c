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

float calculaForca(PARTICULA a, PARTICULA b, float* dx, float* dy, float* dz)
{
    float G = 1; //constante gravitacional
    float distancia = calculaDistancia(a, b);
    float intensidadeForca = G * MASSA * MASSA / distancia + EPSILON; //para nao dar zero
    
    *dx = a.coord.x - b.coord.x; // influencia da força no 
    *dy = a.coord.y - b.coord.y; // vetor decomposto, em cada eixo
    *dz = a.coord.z - b.coord.z;

    return intensidadeForca;
}

void atualizaParticulas(PARTICULA* particula, int numero_de_particulas, int iteracoes, float dt)
{
    int i, j, k;
    for(i = 0; i < iteracoes; i++)
    {
        for(j = 0; j < numero_de_particulas; j++)
        {
            for(k = 0; k < numero_de_particulas; k++)
            {
                if(j != k)
                {
                    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
                    float forca = calculaForca(particula[j], particula[k], &dx, &dy, &dz);
                    particula[i].forca_sofrida.x = dx * forca;
                    particula[i].forca_sofrida.y = dy * forca;
                    particula[i].forca_sofrida.z = dz * forca;
                }
            }
        }

        for (j = 0; j<numero_de_particulas; j++)
        {
            atualizaVelocidade(&particula[j], dt);
            atualizaCoordenada(&particula[j], dt);
        }
    }
    return;
}

void printLog(PARTICULA *particles, int nParticles, int timestep){
    char fileName[128];
    sprintf(fileName, "%s-%d-log.txt", __FILE__,  timestep);
    fprintf(stdout, "Saving file [%s] ", fileName); fflush(stdout);
    FILE *ptr = fopen(fileName, "w+");
    for(int i = 0; i < nParticles; i++){
        fprintf(ptr, "%d \t %.10f %.10f %.10f \t %.10f %.10f %.10f \t %.10f %.10f %.10f \n", i,  particles[i].coord.x, particles[i].coord.y, particles[i].coord.z,  particles[i].velocidade.x, particles[i].velocidade.y, particles[i].velocidade.z, particles[i].forca_sofrida.x, particles[i].forca_sofrida.y, particles[i].forca_sofrida.z);
    }
    fclose(ptr);
    fprintf(stdout, "[OK]\n"); fflush(stdout);
}

int main (int ac, char **av){
    int timesteps  = atoi(av[1]),
        nParticles = atoi(av[2]),
        flagSave = atoi(av[3]);

    char logFile[1024];
    float       dt        =  0.01f;
    PARTICULA *particles = NULL;

    //Stopwatch stopwatch;

    strcpy(logFile, av[4]);

    fprintf(stdout, "\nP2P particle system \n");
    fprintf(stdout, "Memory used %lu bytes \n", nParticles * sizeof(PARTICULA));
    fprintf(stdout, "File %s \n", logFile);

    particles = (PARTICULA *) aligned_alloc(ALING, nParticles * sizeof(PARTICULA));
    assert(particles != NULL);

    inicializador(particles, nParticles);
    //START_STOPWATCH(stopwatch);
    atualizaParticulas(particles, nParticles, timesteps, dt);
    //STOP_STOPWATCH(stopwatch);

    //fprintf(stdout, "Elapsed time: %lf (s) \n", stopwatch.mElapsedTime);
    FILE *ptr = fopen(logFile, "a+");
    assert(ptr != NULL);
    //printf(ptr, "%lu;%lf\n", nParticles * sizeof(tpParticle), stopwatch.mElapsedTime);

    fclose(ptr);

    if (flagSave == 1)
          printLog(particles, nParticles, timesteps);

    free(particles);
}
