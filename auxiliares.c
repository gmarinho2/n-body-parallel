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
    float distancia = calculaDistanciaR3(a, b);
    float intensidadeForca = (a.massa * b.massa * distancia) / pow(distancia, 3);
    return intensidadeForca;
}

void atualizaParticulas(PARTICULA* particula, int numero_de_particulas)
{
    int i, j;
    for(i=0; i<numero_de_particulas; i++)
    {
        for(j= 0; j<numero_de_particulas; j++)
         {
            if(j == i)
            {
                continue;
            }
            else
            {
                atualizaVelocidade(particula[i], particula[j]);
                atualizaAceleracao(particula[i], particula[j]);
            }
        }
    }
    return;
}

int main()
{
    /* code */
    return 0;
}