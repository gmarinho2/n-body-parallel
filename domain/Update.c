#include "../headers/Definitions.h"

void atualizaVelocidade(PARTICULA* particula, double dt)
{
    particula->velocidade.x += dt * particula->forca_sofrida.x;
    particula->velocidade.y += dt * particula->forca_sofrida.y;
    particula->velocidade.z += dt * particula->forca_sofrida.z;
}

void atualizaCoordenada(PARTICULA* particula, double dt)
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
        particula[i].coord.x =  2.0 * (rand() / (double)RAND_MAX) - 1.0;
        particula[i].coord.y =  2.0 * (rand() / (double)RAND_MAX) - 1.0;
        particula[i].coord.z =  2.0 * (rand() / (double)RAND_MAX) - 1.0;
      }
}