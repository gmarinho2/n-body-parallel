#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct vetor
{
    float x, y, z;
    float intensidade;
} VETOR;

typedef struct posicao
{
    float x, y, z;
} POSICAO;

typedef struct particula
{
    POSICAO coord;
    float massa;
    VETOR acelaracao;
    VETOR velocidade;
} PARTICULA;

void leParticulas(FILE arquivo)
{
    return;
}

float calculaDistanciaR3(PARTICULA a, PARTICULA b)
{
    float distancia;
    distancia = sqrt(pow(a.coord.x - b.coord.x, 2) + pow(a.coord.y - b.coord.y, 2) + pow(a.coord.z - b.coord.z, 2));
    return distancia;
}

void atualizaAceleracao(PARTICULA a, PARTICULA b)
{
    return;
}

void atualizaVelocidade(PARTICULA a, PARTICULA b)
{
    return;
}

float calculaForca(PARTICULA a, PARTICULA b)
{
    float G = 1; //igual a 1 para testes
    float distancia = calculaDistanciaR3(a, b);
    float intensidadeForca = G * (a.massa * b.massa * distancia) / pow(distancia, 3);
    return intensidadeForca;
}

void atualizaParticulas(PARTICULA* particula, int numero_de_particulas, int iteracoes)
{
    int i, j, k;

    for(i = 0; i < iteracoes; i++)
    {
        for(j = 0; j < numero_de_particulas; j++)
        {
            for(k = 0; k < numero_de_particulas; k++)
            {
                if(j == k)
                {
                    continue;
                }
                else
                {
                    atualizaAceleracao(particula[j], particula[k]);
                    atualizaVelocidade(particula[j], particula[k]);
                }
            }

        }
    }
    return;
}