#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASSA 1
#define EPSILON 1E-9

void leParticulas(FILE arquivo)
{
    return;
}

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

float calculaDistanciaR3(PARTICULA a, PARTICULA b)
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
    return;
}

void atualizaCoordenada(PARTICULA* particula, float dt)
{
    particula->coord.x += dt * particula->velocidade.x;
    particula->coord.y += dt * particula->velocidade.y;
    particula->coord.z += dt * particula->velocidade.z;
    return;
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