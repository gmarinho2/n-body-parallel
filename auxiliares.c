#include <stdio.h>
#include <stdlib.h>

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
    POSICAO posicao;
    float massa;
    VETOR acelaracao;
    VETOR velocidade;
} PARTICULA;

void leParticulas(FILE arquivo)
{
    return;
}

void atualizaAceleracao(PARTICULA a, PARTICULA b)
{
    return;
}

void atualizaVelocidade(PARTICULA a, PARTICULA b)
{
    return;
}

void calculaForca(PARTICULA a, PARTICULA b)
{

    return;
}

void atualizaParticulas(PARTICULA* particula, int numero_de_particulas)
{
    int i, j;
    for(i=0; i<numero_de_particulas; i++)
    {
        for(j = 0; j<numero_de_particulas; j++)
         {
            if(j = i)
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