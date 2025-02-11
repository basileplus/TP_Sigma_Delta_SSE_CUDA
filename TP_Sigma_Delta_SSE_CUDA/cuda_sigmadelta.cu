#include "cuda_sigmadelta.cuh"
#include <intrin.h>

#define	WHITE			0xFF
#define BLACK			0x00

// Kernel SigmaDelta
// Chaque thread (correspondant à un pixel) parcourt l'ensemble des images (à partir de la 2ème image)
// et met à jour le modèle M, la variance V et l'image I en fonction de la différence Delta
__global__ void kernel_sigmadelta(unsigned char* d_I, unsigned char* d_M, unsigned short* d_V,
	int NbPixels, int NbImages, unsigned char N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < NbPixels)
	{
		// Pour chaque image (à partir de la deuxième, la première ayant servi à l'initialisation)
		for (int j = 1; j < NbImages; j++)
		{
			int offset = j * NbPixels;
			unsigned char I_val = d_I[offset + idx];
			unsigned char M_val = d_M[idx];
			unsigned short V_val = d_V[idx];

			// Mise à jour du modèle
			if (M_val < I_val)
				M_val++;
			else if (M_val > I_val)
				M_val--;

			// Calcul de Delta = |M - I|
			int Delta = abs((int)M_val - (int)I_val);
			int DeltaN = Delta * N; // Multiplication par le facteur N

			// Mise à jour de la variance V si Delta n'est pas nul
			if (Delta != 0)
			{
				if (V_val < DeltaN)
					V_val++;
				else if (V_val > DeltaN)
					V_val--;
			}

			// Détection du mouvement : si Delta < V, le pixel est WHITE, sinon BLACK
			I_val = (Delta < V_val) ? WHITE : BLACK;

			// Sauvegarde des résultats
			d_M[idx] = M_val;
			d_V[idx] = V_val;
			d_I[offset + idx] = I_val;
		}
	}
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

// Kernel pour l'érosion (MorphoErosion)
// Pour chaque image et pour chaque pixel (en excluant la bordure), on vérifie que les 9 pixels de la fenêtre 3x3 sont BLACK.
__global__ void kernel_erosion(unsigned char* d_Src, unsigned char* d_Dst, int Width, int Height, int NbImages)
{
	// Les threads couvrent l'image à l'intérieur des bordures : indices i de 1 à Width-2 et j de 1 à Height-2.
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z; // indice de l'image

	if (i < (Width - 1) && j < (Height - 1) && k < NbImages)
	{
		int offset = k * Width * Height;
		bool allBlack = true;
		// Vérification de la fenêtre 3x3
		for (int dj = -1; dj <= 1 && allBlack; dj++)
		{
			for (int di = -1; di <= 1; di++)
			{
				if (d_Src[offset + (j + dj) * Width + (i + di)] != BLACK)
				{
					allBlack = false;
					break;
				}
			}
		}
		d_Dst[offset + j * Width + i] = allBlack ? BLACK : WHITE;
	}
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

// Kernel pour la dilatation (MorphoDilatation)
// Pour chaque pixel (hors bordure) de chaque image, si au moins un pixel de la fenêtre 3x3 est BLACK, alors le résultat est BLACK.
__global__ void kernel_dilatation(unsigned char* d_Src, unsigned char* d_Dst, int Width, int Height, int NbImages)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z;

	if (i < (Width - 1) && j < (Height - 1) && k < NbImages)
	{
		int offset = k * Width * Height;
		bool hasBlack = false;
		for (int dj = -1; dj <= 1 && !hasBlack; dj++)
		{
			for (int di = -1; di <= 1; di++)
			{
				if (d_Src[offset + (j + dj) * Width + (i + di)] == BLACK)
				{
					hasBlack = true;
					break;
				}
			}
		}
		d_Dst[offset + j * Width + i] = hasBlack ? BLACK : WHITE;
	}
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