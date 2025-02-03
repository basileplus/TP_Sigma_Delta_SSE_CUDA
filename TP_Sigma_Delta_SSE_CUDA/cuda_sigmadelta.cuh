#ifndef _CUDA_SIGMADELTA_CUH_
#define _CUDA_SIGMADELTA_CUH_

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define cudaCheckError(code,mess)	if (code != cudaSuccess) printf("Cuda erreur (%s): %s\n", mess, cudaGetErrorString(code))

void cuda_sigmadelta(unsigned char* Images, unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char N, unsigned long long* duree);
void cuda_erosion(unsigned char* Images, unsigned char* Erosions, unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned long long* duree);
void cuda_dilatation(unsigned char* Images, unsigned char* Dilatations, unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned long long* duree);

#endif
