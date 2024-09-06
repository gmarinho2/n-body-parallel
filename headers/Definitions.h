#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

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

double calculaDistancia(PARTICULA a, PARTICULA b);

void atualizaVelocidade(PARTICULA* particula, double dt);

void atualizaCoordenada(PARTICULA* particula, double dt);

void inicializador(PARTICULA *particula, int quantidade);

double calculaForca(PARTICULA a, PARTICULA b, double* dx, double* dy, double* dz);

void printLog(PARTICULA *particles, int quantParticulas, int timestep, char* type);

void simulacao(PARTICULA* particula, int quantParticulas, int timesteps, double dt);

#endif