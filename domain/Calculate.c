#include "../headers/Definitions.h"

double calculaDistancia(PARTICULA a, PARTICULA b)
{
    double distancia;
    distancia = sqrt(pow(a.coord.x - b.coord.x, 2) + pow(a.coord.y - b.coord.y, 2) + pow(a.coord.z - b.coord.z, 2) + EPSILON);
    return distancia;
}

double calculaForca(PARTICULA a, PARTICULA b, double* dx, double* dy, double* dz)
{
    double G = 1.0; //constante gravitacional
    double distancia = calculaDistancia(a, b);
    double intensidadeForca = (G) / (distancia * distancia); //para nao dar zero
    
    *dx = a.coord.x - b.coord.x; // influencia da força no 
    *dy = a.coord.y - b.coord.y; // vetor decomposto, em cada eixo
    *dz = a.coord.z - b.coord.z;

    return intensidadeForca;
}
