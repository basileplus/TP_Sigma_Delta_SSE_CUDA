#include "cuda_sigmadelta.cuh"
#include <intrin.h>

#define	WHITE			0xFF
#define BLACK			0x00

__global__ void kernel_sigmadelta(unsigned char* cudaImages, unsigned int NbPixels, unsigned int NbImages, unsigned char N)
{
	// A COMPLETER
}

void cuda_sigmadelta(unsigned char* Images, unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char N, unsigned long long* duree)
{
	unsigned long long duree_tmp;
	unsigned int NbBytes = Width * Height * NbImages, NbPixels = Width * Height;;
	unsigned char* ptr_Cuda_Images;

	cudaCheckError(cudaMalloc(&ptr_Cuda_Images, NbBytes), "cuda_sigmadelta - malloc Images");
	cudaCheckError(cudaMemcpy(ptr_Cuda_Images, Images, NbBytes, cudaMemcpyHostToDevice), "cuda_sigmadelta - memcpy H->D");
	duree_tmp = __rdtsc();
	kernel_sigmadelta << <1, 1 /* MODIFIER SI NECESSAIRE */ >> > (ptr_Cuda_Images, NbPixels, NbImages, N);
	cudaDeviceSynchronize();
	*duree = __rdtsc() - duree_tmp;
	cudaCheckError(cudaGetLastError(), "kernel_sigmadelta");
	cudaCheckError(cudaMemcpy(Images, ptr_Cuda_Images, NbBytes, cudaMemcpyDeviceToHost), "cuda_sigmadelta - memcpy D->H");
	cudaCheckError(cudaFree(ptr_Cuda_Images), "cuda_sigmadelta - free Images");
}

__global__ void kernel_erosion(unsigned char* cudaImages, unsigned char *cudaErosions, unsigned int Width, unsigned int Height, unsigned int NbImages)
{
	// A COMPLETER
}

void cuda_erosion(unsigned char* Images, unsigned char* Erosions, unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned long long* duree)
{
	unsigned long long duree_tmp;
	unsigned int NbBytes = Width * Height * NbImages, NbPixels = Width * Height;;
	unsigned char *ptr_Cuda_Images, *ptr_Cuda_Erosions;
	
	cudaCheckError(cudaMalloc(&ptr_Cuda_Images, NbBytes), "cuda_erosion - malloc Images");
	cudaCheckError(cudaMalloc(&ptr_Cuda_Erosions, NbBytes), "cuda_erosion - malloc Erosions");
	cudaCheckError(cudaMemcpy(ptr_Cuda_Images, Images, NbBytes, cudaMemcpyHostToDevice), "cuda_erosion - memcpy H->D");
	duree_tmp = __rdtsc();
	kernel_erosion << <1, 1 /* MODIFIER SI NECESSAIRE */ >> > (ptr_Cuda_Images, ptr_Cuda_Erosions, Width, Height, NbImages);
	cudaDeviceSynchronize();
	*duree = __rdtsc() - duree_tmp;
	cudaCheckError(cudaGetLastError(), "kernel_erosion");
	cudaCheckError(cudaMemcpy(Erosions, ptr_Cuda_Erosions, NbBytes, cudaMemcpyDeviceToHost), "cuda_erosion - memcpy D->H");
	cudaCheckError(cudaFree(ptr_Cuda_Images), "cuda_erosion - free Images");
	cudaCheckError(cudaFree(ptr_Cuda_Erosions), "cuda_erosion - free Erosions");
}

__global__ void kernel_dilatation(unsigned char* cudaImages, unsigned char* cudaErosions, unsigned int Width, unsigned int Height, unsigned int NbImages)
{
	// A COMPLETER
}

void cuda_dilatation(unsigned char* Images, unsigned char* Dilatations, unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned long long* duree)
{
	unsigned long long duree_tmp;
	unsigned int NbBytes = Width * Height * NbImages, NbPixels = Width * Height;;
	unsigned char* ptr_Cuda_Images, * ptr_Cuda_Dilatations;

	cudaCheckError(cudaMalloc(&ptr_Cuda_Images, NbBytes), "cuda_dilatation - malloc Images");
	cudaCheckError(cudaMalloc(&ptr_Cuda_Dilatations, NbBytes), "cuda_dilatation - malloc Dilatations");
	cudaCheckError(cudaMemcpy(ptr_Cuda_Images, Images, NbBytes, cudaMemcpyHostToDevice), "cuda_dilatation - memcpy H->D");
	duree_tmp = __rdtsc();
	kernel_dilatation << <1, 1 /* MODIFIER SI NECESSAIRE */ >> > (ptr_Cuda_Images, ptr_Cuda_Dilatations, Width, Height, NbImages);
	cudaDeviceSynchronize();
	*duree = __rdtsc() - duree_tmp;
	cudaCheckError(cudaGetLastError(), "kernel_dilatation");
	cudaCheckError(cudaMemcpy(Dilatations, ptr_Cuda_Dilatations, NbBytes, cudaMemcpyDeviceToHost), "cuda_dilatation - memcpy D->H");
	cudaCheckError(cudaFree(ptr_Cuda_Images), "cuda_dilatation - free Images");
	cudaCheckError(cudaFree(ptr_Cuda_Dilatations), "cuda_dilatation - free Dilatations");
}