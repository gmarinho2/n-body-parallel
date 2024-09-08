#include "../headers/Definitions.h"

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