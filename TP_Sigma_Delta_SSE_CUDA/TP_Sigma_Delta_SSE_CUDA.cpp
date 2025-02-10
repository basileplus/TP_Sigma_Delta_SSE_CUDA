#include "cuda_sigmadelta.cuh"
#include "pgm.hpp"
#include <stdio.h>
#include <emmintrin.h>
#include <intrin.h>

#define	STRING_LENGTH	1024
#define	WHITE			0xFF
#define BLACK			0x00
#define SIMD 1

#define ALIGNMENT	16

#define vec_left1(v0, v1) _mm_or_si128(_mm_srli_si128 (v0, 15), _mm_slli_si128 (v1,  1))
#define vec_right1(v1, v2) _mm_or_si128(_mm_srli_si128 (v1,  1), _mm_slli_si128 (v2, 15))

#define	SCALAIRE
#define SIMD
//#define CUDA

void FirstStepSigmaDelta(unsigned int NbPixels, unsigned char* M, unsigned char* I, unsigned short* V)
{
	unsigned int	i;

	for (i = 0; i < NbPixels; i++)
	{
		M[i] = I[i];
		V[i] = 0;
		I[i] = BLACK;
	}
}

void FirstStepSigmaDelta_simd(unsigned int NbPixels, unsigned char* M, unsigned char* I, unsigned short* V)
{
	unsigned int i;

	__m128i vecZero = _mm_set1_epi8(0x00);
	__m128i vecBlack = _mm_set1_epi8(BLACK);

	for (i = 0; i < NbPixels; i += 16) {
		_mm_store_si128((__m128i*)&M[i], _mm_load_si128((__m128i*) & I[i])); //Comme les entiers peuvent etre codes sur differents nomrbe de bits pour load on cast en 128i
		_mm_store_si128((__m128i*)&V[i], vecZero);
		_mm_store_si128((__m128i*) & V[i+8], vecZero);
		_mm_store_si128((__m128i*) & I[i], vecBlack);
	}
}

void SigmaDelta(unsigned int NbPixels, unsigned int NbImages, unsigned char* M, unsigned char* I, unsigned short* V, unsigned char N)
{
	unsigned int	i, j, Offset = 0;
	unsigned char	tmp_M, tmp_I, Delta;
	unsigned short	tmp_V, DeltaN;

	for (j = 0, Offset = 0; j < NbImages; j++, Offset += NbPixels)
		for (i = 0; i < NbPixels; i++)
		{
			tmp_M = M[i];
			tmp_I = I[Offset + i];
			tmp_V = V[i];

			if (tmp_M < tmp_I) tmp_M++;
			if (tmp_M > tmp_I) tmp_M--;
			/* Autrement, tmp_M ne change pas */

			Delta = abs(tmp_M - tmp_I);
			DeltaN = Delta * N;

			if (Delta != 0)
			{
				if (tmp_V < DeltaN) tmp_V++;
				if (tmp_V > DeltaN) tmp_V--;
				/* Autrement, tmp_V ne change pas */
			}

			if (Delta < tmp_V)
				tmp_I = WHITE;
			else
				tmp_I = BLACK;

			M[i] = tmp_M;
			I[Offset + i] = tmp_I;
			V[i] = tmp_V;
		}
}

void SigmaDelta_simd(unsigned int NbPixels, unsigned int NbImages, unsigned char* M, unsigned char* I, unsigned short* V, unsigned char N)
{
	unsigned int	i, j, n,  Offset = 0;
	__m128i	vec_tmp_M, vec_tmp_I, vec_Delta; //char
	__m128i	vec_tmp_V, vec_DeltaN = _mm_set1_epi16(0x0000); //short
	__m128i vec_mask, vec_inc, vec_dec;
	__m128i vec_white = _mm_set1_epi8(WHITE);
	__m128i vec_zero = _mm_set1_epi8(0x00);
	__m128i vec_one = _mm_set1_epi8(0x01);

	for (j = 0, Offset = 0; j < NbImages; j++, Offset += NbPixels)
		for (i = 0; i < NbPixels; i+=16) {
			vec_tmp_M = _mm_load_si128((__m128i*) & M[i]);
			vec_tmp_I = _mm_load_si128((__m128i*) & I[Offset + i]);
			vec_tmp_V = _mm_load_si128((__m128i*) & V[i]);

			// if (tmp_M < tmp_I) tmp_M++; if (tmp_M > tmp_I) tmp_M--; else tmp_M;
			// On est pas sense avoir M qui depasse 255 car c'est la moyenne de valeur sur 8 bits donc adds ne devrait pas poser de pb
			// Pb : les comparaisons existent que sur les epi
			// on fait subs(a,b) on aura : 0 si b>a (valeur saturee) ; une valeur >=1 si a>b ; ça donne un masque interessant pour faire l'operation
			// si on prend min(subs(a,b);0x1) on aura 0 si b>a et 0x1 sinon
			vec_tmp_M = _mm_adds_epu8(vec_tmp_M,_mm_min_epu8(_mm_subs_epu8(vec_tmp_I, vec_tmp_M), vec_one)); // if (tmp_M < tmp_I) tmp_M++;
			vec_tmp_M = _mm_subs_epu8(vec_tmp_M, _mm_min_epu8(_mm_subs_epu8(vec_tmp_M, vec_tmp_I), vec_one)); // if (tmp_I < tmp_M) tmp_M--;

			vec_Delta = _mm_subs_epu8(_mm_max_epu8(vec_tmp_M,vec_tmp_I), _mm_min_epu8(vec_tmp_M,vec_tmp_I));

			// vec_DeltaN = _mm_vec_Delta * N;
			vec_DeltaN = vec_Delta;
			for (n = 1;  n < N; n++) {
				vec_DeltaN = _mm_adds_epu8(vec_DeltaN, vec_Delta);
			}

			vec_mask = _mm_cmpeq_epi8(vec_Delta, vec_zero); // if Delta==0
			vec_inc = _mm_andnot_si128(vec_mask, _mm_min_epu8(_mm_subs_epu8(vec_DeltaN, vec_tmp_V), vec_one)); // vec_inc = 1 if (tmp_V < DeltaN) & Delta!0
			vec_dec = _mm_andnot_si128(vec_mask, _mm_min_epu8(_mm_subs_epu8(vec_tmp_V, vec_DeltaN), vec_one)); // vec_inc = 1 if (tmp_V > DeltaN) & Delta!0

			vec_tmp_V = _mm_adds_epu8(vec_tmp_V, vec_inc); // tmp_V += vec_inc;
			vec_tmp_V = _mm_subs_epu8(vec_tmp_V, vec_dec); // tmp_V += vec_dec;

			vec_mask = _mm_cmpeq_epi8(_mm_subs_epu8(vec_Delta, vec_tmp_V), vec_zero); // if Delta < tmp_V
			vec_tmp_I = vec_mask; // I = 0xff  if Delta < tmp_V, 0x00 else
		
			_mm_store_si128((__m128i*) & M[i], vec_tmp_M);
			_mm_store_si128((__m128i*) & I[Offset + i], vec_tmp_I);
			_mm_store_si128((__m128i*) & V[i], vec_tmp_V);
		}
}


	void InitErosionDilatation(unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char* M)
	{
		unsigned int i, j, NbPixels = Width * Height, Offset;

		for (j = 0, Offset = 0; j < NbImages; j++, Offset += NbPixels)
		{
			for (i = 0; i < Width; i++)
			{
				M[Offset + i] = WHITE;							// <=> (*M)[j][0][i] = WHITE
				M[Offset + (Height - 1) * Width + i] = WHITE;	// <=> (*M)[j][Height-1][i] = WHITE
			}
			for (i = 0; i < Height; i++)
			{
				M[Offset + i * Width] = WHITE;					// <=> (*M)[j][i][0] = WHITE
				M[Offset + (i + 1) * Width - 1] = WHITE;		// <=> (*M)[j][i][Width-1] = WHITE
			}
		}
	}

void InitErosionDilatation_simd(unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char* M)
{
	unsigned int i, j, NbPixels = Width * Height, Offset;
	__m128i vec_white = _mm_set1_epi8(WHITE);

	for (j = 0, Offset = 0; j < NbImages; j++, Offset += NbPixels)
	{
		for (i = 0; i < Width; i+=16)
		{
			_mm_store_si128((__m128i*) & M[Offset+i], vec_white); // <=> (*M)[j][0][i] = WHITE
			_mm_store_si128((__m128i*) & M[Offset + (Height - 1) * Width + i], vec_white); // <=> (*M)[j][Height-1][i] = WHITE
		}
		for (i = 0; i < Height; i++)
		{
			_mm_store_si128((__m128i*) & M[Offset + i * Width], vec_white); // <=> (*M)[j][i][0] = WHITE sur 16 lignes
			_mm_store_si128((__m128i*) & M[Offset - 16 + (i + 1) * Width - 1], vec_white); // <=> (*M)[j][i][Width-1] = WHITE sur 16 lignes
		}
	}
}

void MorphoErosion(unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char* Src, unsigned char* Dst)
{
	unsigned int i, j, k, NbPixels = Width * Height, Offset;

	for (k = 0, Offset = 0; k < NbImages; k++, Offset += NbPixels)
		for (j = 1; j < (Height - 1); j++)	
			for (i = 1; i < (Width - 1); i++)
				if ((Src[Offset + (j - 1) * Width + i - 1] == BLACK) &&
					(Src[Offset + (j - 1) * Width + i] == BLACK) &&
					(Src[Offset + (j - 1) * Width + i + 1] == BLACK) &&
					(Src[Offset + j * Width + i - 1] == BLACK) &&
					(Src[Offset + j * Width + i] == BLACK) &&
					(Src[Offset + j * Width + i + 1] == BLACK) &&
					(Src[Offset + (j + 1) * Width + i - 1] == BLACK) &&
					(Src[Offset + (j + 1) * Width + i] == BLACK) &&
					(Src[Offset + (j + 1) * Width + i + 1] == BLACK))
					Dst[Offset + j * Width + i] = BLACK;
				else
					Dst[Offset + j * Width + i] = WHITE;
}

void MorphoErosion_simd(unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char* Src, unsigned char* Dst)
{
	unsigned int i, j, k, NbPixels, Offset;
	NbPixels = Width * Height;
	Offset = 0;

	__m128i v0, v1, vZero, vWhite;
	vZero = _mm_setzero_si128();
	vWhite = _mm_set1_epi8((char)WHITE);

	// Pointeurs pour parcourir les 3 lignes
	unsigned char* p0, * p1, * p2, * pDst;

	for (k = 0; k < NbImages; k++, Offset += NbPixels)
	{
		for (j = 1; j < Height - 1; j++)
		{
			// On définit les pointeurs pour la ligne du dessus, la ligne courante et la ligne du dessous.
			p0 = Src + Offset + (j - 1) * Width;
			p1 = Src + Offset + j * Width;
			p2 = Src + Offset + (j + 1) * Width;
			pDst = Dst + Offset + j * Width;
			// On parcourt les colonnes de 1 à Width-2 par blocs de 16 pixels.
			for (i = 1; i < Width - 1; i += 16)
			{
				// Ligne j-1
				v0 = _mm_loadu_si128((__m128i*)(p0 + i - 1)); // colonnes i-1 à i+14
				v1 = _mm_loadu_si128((__m128i*)(p0 + i));     // colonnes i à i+15
				v0 = _mm_or_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p0 + i + 1));   // colonnes i+1 à i+16
				v0 = _mm_or_si128(v0, v1);

				// Ligne j (courante)
				v1 = _mm_loadu_si128((__m128i*)(p1 + i - 1));
				v0 = _mm_or_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p1 + i));
				v0 = _mm_or_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p1 + i + 1));
				v0 = _mm_or_si128(v0, v1);

				// Ligne j+1
				v1 = _mm_loadu_si128((__m128i*)(p2 + i - 1));
				v0 = _mm_or_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p2 + i));
				v0 = _mm_or_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p2 + i + 1));
				v0 = _mm_or_si128(v0, v1);

				// Si v0 vaut zéro alors tous les 9 pixels sont BLACK.
				// _mm_cmpeq_epi8 renvoie 0xFF pour les octets égaux à zéro, 0 sinon.
				v0 = _mm_cmpeq_epi8(v0, vZero);
				// Pour obtenir le résultat final : si tous noirs, on veut BLACK (0x00) ; sinon WHITE (0xff).
				// Ainsi, on inverse le masque avec _mm_andnot_si128 : ~mask & 0xFF.
				v0 = _mm_andnot_si128(v0, vWhite);

				_mm_storeu_si128((__m128i*)(pDst + i), v0);
			}
		}
	}
}

void MorphoDilatation(unsigned int Width, unsigned int Height, unsigned int NbImages, unsigned char* Src, unsigned char* Dst)
{
	unsigned int i, j, k, NbPixels = Width * Height, Offset;

	for (k = 0, Offset = 0; k < NbImages; k++, Offset += NbPixels)
		for (j = 1; j < (Height - 1); j++)
			for (i = 1; i < (Width - 1); i++)
				if ((Src[Offset + (j - 1) * Width + i - 1] == BLACK) ||
					(Src[Offset + (j - 1) * Width + i] == BLACK) ||
					(Src[Offset + (j - 1) * Width + i + 1] == BLACK) ||
					(Src[Offset + j * Width + i - 1] == BLACK) ||
					(Src[Offset + j * Width + i] == BLACK) ||
					(Src[Offset + j * Width + i + 1] == BLACK) ||
					(Src[Offset + (j + 1) * Width + i - 1] == BLACK) ||
					(Src[Offset + (j + 1) * Width + i] == BLACK) ||
					(Src[Offset + (j + 1) * Width + i + 1] == BLACK))
					Dst[Offset + j * Width + i] = BLACK;
				else
					Dst[Offset + j * Width + i] = WHITE;
}

void MorphoDilatation_simd(unsigned int Width, unsigned int Height, unsigned int NbImages,
	unsigned char* Src, unsigned char* Dst)
{
	unsigned int i, j, k, NbPixels, Offset;
	NbPixels = Width * Height;
	Offset = 0;

	// Déclaration des variables SIMD
	__m128i v0, v1, vWhite;
	vWhite = _mm_set1_epi8((char)WHITE);

	// Pointeurs pour accéder aux 3 lignes
	unsigned char* p0, * p1, * p2, * pDst;

	for (k = 0; k < NbImages; k++, Offset += NbPixels) {
		for (j = 1; j < Height - 1; j++) {
			// Définir les pointeurs pour la ligne supérieure, la ligne courante et la ligne inférieure
			p0 = Src + Offset + (j - 1) * Width;
			p1 = Src + Offset + j * Width;
			p2 = Src + Offset + (j + 1) * Width;
			pDst = Dst + Offset + j * Width;

			// Parcours des colonnes intérieures par blocs de 16 pixels
			for (i = 1; i < Width - 1; i += 16) {
				// --- Ligne j-1 ---
				v0 = _mm_loadu_si128((__m128i*)(p0 + i - 1));  // colonnes i-1 à i+14
				v1 = _mm_loadu_si128((__m128i*)(p0 + i));       // colonnes i à i+15
				v0 = _mm_and_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p0 + i + 1));    // colonnes i+1 à i+16
				v0 = _mm_and_si128(v0, v1);

				// --- Ligne j (courante) ---
				v1 = _mm_loadu_si128((__m128i*)(p1 + i - 1));
				v0 = _mm_and_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p1 + i));
				v0 = _mm_and_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p1 + i + 1));
				v0 = _mm_and_si128(v0, v1);

				// --- Ligne j+1 ---
				v1 = _mm_loadu_si128((__m128i*)(p2 + i - 1));
				v0 = _mm_and_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p2 + i));
				v0 = _mm_and_si128(v0, v1);
				v1 = _mm_loadu_si128((__m128i*)(p2 + i + 1));
				v0 = _mm_and_si128(v0, v1);

				// À ce stade, v0 contient le AND bit-à-bit des 9 pixels de la fenêtre 3×3.
				// Si tous les pixels étaient WHITE (0xFF), alors v0 vaut 0xFF dans chaque octet.
				// Sinon (si au moins un pixel vaut BLACK, c'est-à-dire 0x00), v0 sera égal à 0.
				// On compare alors v0 à un vecteur rempli de WHITE pour obtenir :
				//    - 0xFF (WHITE) si tous les pixels étaient WHITE,
				//    - 0x00 (BLACK) sinon.
				v0 = _mm_cmpeq_epi8(v0, vWhite);

				// Stockage du résultat dans Dst
				_mm_storeu_si128((__m128i*)(pDst + i), v0);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	unsigned long long	Duree_SigmaDelta = 0, Duree_Erosion = 0, Duree_Dilatation = 0, 
						Duree_SigmaDelta_simd = 0, Duree_Erosion_simd = 0, Duree_Dilatation_simd = 0,
						Duree_SigmaDelta_Cuda = 0, Duree_Erosion_Cuda = 0, Duree_Dilatation_Cuda = 0, Duree_Tmp;
	unsigned int	Width, Height, NbPixels, i, j, k, indexFirst, indexLast, NbImages, Offset;
	char	String[STRING_LENGTH];
	unsigned char* Matrice_M, * Images, N, * Erosions, * Dilatations;
	unsigned short* Matrice_V;
	
	if (argc < 7)	// Vérification des arguments
	{
		printf("Arguments manquants :\n");
		printf("\t\tArg 1 :\tChemin du répertoire contenant les images à traiter\n");
		printf("\t\tArg 2 :\tNom des images sans extension ni numéro\n");
		printf("\t\tArg 3 :\tNuméro de la première image à traiter\n");
		printf("\t\tArg 4 :\tNuméro de la dernière image à traiter\n");
		printf("\t\tArg 5 :\tFacteur d'amplification N\n");
		printf("\t\tArg 6 :\tChemin du répertoire où les images traitées seront stockées\n");
		return 0;
	}

	indexFirst = (unsigned char)atoi(argv[3]);	// Indice de la première image
	indexLast = (unsigned char)atoi(argv[4]);	// Indice de la dernière image
	NbImages = indexLast - indexFirst + 1;		// Calcul du nombre d'images
	N = (unsigned char)atoi(argv[5]);			// Facteur N

	// Lecture de la taille des images
	sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[1], argv[2], atoi(argv[3]));
	if (ReadPGM_Size(String, &Width, &Height))
	{
		printf("Erreur...\n");
		return	1;
	}
	// Calcul du nombre de pixels par image
	NbPixels = Width * Height;

	// Allocation mémoire: matrices et images
	Images = (unsigned char*)_aligned_malloc(NbPixels * NbImages * sizeof(unsigned char), ALIGNMENT);
	Matrice_M = (unsigned char*)_aligned_malloc(NbPixels * sizeof(unsigned char), ALIGNMENT);
	Matrice_V = (unsigned short*)_aligned_malloc(NbPixels * sizeof(unsigned short), ALIGNMENT);
	Erosions = (unsigned char*)_aligned_malloc(NbPixels * NbImages * sizeof(unsigned char), ALIGNMENT);
	Dilatations = (unsigned char*)_aligned_malloc(NbPixels * NbImages * sizeof(unsigned char), ALIGNMENT);

#ifdef SCALAIRE
	// Lecture de toutes les images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset+=NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[1], argv[2], i);
		if (ReadPGM(String, Images + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}
	
	// Initialisation des matrices pour SigmaDelta
	FirstStepSigmaDelta(NbPixels, Matrice_M, Images, Matrice_V);
	// Traitement des images suivantes
	Duree_Tmp = __rdtsc();
	SigmaDelta(NbPixels, NbImages -1 , Matrice_M, Images + NbPixels, Matrice_V, N);
	Duree_SigmaDelta = __rdtsc() - Duree_Tmp;

	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSD", i);
		if (WritePGM(String, Width, Height, Images + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	// Initialisation des images pour l'érosion
	InitErosionDilatation(Width, Height, NbImages, Erosions);
	Duree_Tmp = __rdtsc();
	MorphoErosion(Width, Height, NbImages, Images, Erosions);
	Duree_Erosion = __rdtsc() - Duree_Tmp;
	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSDE", i);
		if (WritePGM(String, Width, Height, Erosions + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	// Initialisation des images pour l'érosion
	InitErosionDilatation(Width, Height, NbImages, Dilatations);
	Duree_Tmp = __rdtsc();
	MorphoDilatation(Width, Height, NbImages, Erosions, Dilatations);
	Duree_Dilatation = __rdtsc() - Duree_Tmp;
	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSDED", i);
		if (WritePGM(String, Width, Height, Dilatations + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}
#else
	Duree_SigmaDelta = 1;
	Duree_Erosion = 1;
	Duree_Dilatation = 1;
#endif

#ifdef SIMD
	// Relecture de toutes les images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[1], argv[2], i);
		if (ReadPGM(String, Images + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	// Initialisation des matrices pour SigmaDelta
	FirstStepSigmaDelta_simd(NbPixels, Matrice_M, Images, Matrice_V);
	// Traitement des images suivantes
	Duree_Tmp = __rdtsc();
	SigmaDelta_simd(NbPixels, NbImages - 1, Matrice_M, Images + NbPixels, Matrice_V, N);
	Duree_SigmaDelta_simd = __rdtsc() - Duree_Tmp;

	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSD_simd", i);
		if (WritePGM(String, Width, Height, Images + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	// Initialisation des images pour l'érosion
	InitErosionDilatation(Width, Height, NbImages, Erosions);
	Duree_Tmp = __rdtsc();
	MorphoErosion_simd(Width, Height, NbImages, Images, Erosions);
	Duree_Erosion_simd = __rdtsc() - Duree_Tmp;
	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSDE_simd", i);
		if (WritePGM(String, Width, Height, Erosions + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	// Initialisation des images pour l'érosion
	InitErosionDilatation_simd(Width, Height, NbImages, Dilatations);
	Duree_Tmp = __rdtsc();
	MorphoDilatation_simd(Width, Height, NbImages, Erosions, Dilatations);
	Duree_Dilatation_simd = __rdtsc() - Duree_Tmp;
	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSDED_simd", i);
		if (WritePGM(String, Width, Height, Dilatations + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}
#else
	Duree_SigmaDelta_simd = 1;
	Duree_Erosion_simd = 1;
	Duree_Dilatation_simd = 1;
#endif

#ifdef CUDA
	// Relecture de toutes les images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[1], argv[2], i);
		if (ReadPGM(String, Images + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	cuda_sigmadelta(Images, Width, Height, NbImages, N, &Duree_SigmaDelta_Cuda);
	
	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSD_cuda", i);
		if (WritePGM(String, Width, Height, Images + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	cuda_erosion(Images, Erosions, Width, Height, NbImages, &Duree_Erosion_Cuda);

	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSDE_cuda", i);
		if (WritePGM(String, Width, Height, Erosions + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}

	cuda_dilatation(Erosions, Dilatations, Width, Height, NbImages, &Duree_Dilatation_Cuda);

	// Ecriture des images
	for (i = indexFirst, Offset = 0; i <= indexLast; i++, Offset += NbPixels)
	{
		sprintf_s(String, STRING_LENGTH, "%s\\%s%d.pgm", argv[6], "ResultSDED_cuda", i);
		if (WritePGM(String, Width, Height, Dilatations + Offset))
		{
			printf("Erreur...\n");
			return	1;
		}
	}
#else
	Duree_SigmaDelta_Cuda = 1;
	Duree_Erosion_Cuda = 1;
	Duree_Dilatation_Cuda = 1;
#endif

	printf("Duree Sigma Delta      : %lld cycles\n", Duree_SigmaDelta);
	printf("Duree Erosion          : %lld cycles\n", Duree_Erosion);
	printf("Duree Dilatation       : %lld cycles\n", Duree_Dilatation);
	printf("Duree Sigma Delta simd : %lld cycles (gain = %2.2f)\n", Duree_SigmaDelta_simd, ((float)Duree_SigmaDelta)/Duree_SigmaDelta_simd);
	printf("Duree Erosion simd     : %lld cycles (gain = %2.2f)\n", Duree_Erosion_simd, ((float)Duree_Erosion) / Duree_Erosion_simd);
	printf("Duree Dilatation simd  : %lld cycles (gain = %2.2f)\n", Duree_Dilatation_simd, ((float)Duree_Dilatation) / Duree_Dilatation_simd);
	printf("Duree Sigma Delta cuda : %lld cycles (gain = %2.2f)\n", Duree_SigmaDelta_Cuda, ((float)Duree_SigmaDelta) / Duree_SigmaDelta_Cuda);
	printf("Duree Erosion cuda     : %lld cycles (gain = %2.2f)\n", Duree_Erosion_Cuda, ((float)Duree_Erosion) / Duree_Erosion_Cuda);
	printf("Duree Dilatation cuda  : %lld cycles (gain = %2.2f)\n", Duree_Dilatation_Cuda, ((float)Duree_Dilatation) / Duree_Dilatation_Cuda);
	printf("Appuyez sur une touche pour quitter...\n");
	//getchar();

	return 0;
}