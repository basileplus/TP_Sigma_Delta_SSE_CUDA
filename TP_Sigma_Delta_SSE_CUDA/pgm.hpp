#ifndef	_PGM_H_
#define	_PGM_H_

#include <stdio.h>
#include <stdlib.h>

#define	ALIGNMENT	16 // bytes

// La fonction suivante permet de lire la taille d'une image pgm en niveaux de gris codée sur 8 bits.
// FileName correspond au chemin du fichier à lire. La fonction écrit aux adresses Width et Height 
// les dimensions de l'image lue. La fonction suivante retourne 0 en cas de succès.
int ReadPGM_Size(char* FileName,
	unsigned int* Width,
	unsigned int* Height);

// La fonction suivante permet de lire une image pgm en niveaux de gris codée sur 8 bits.
// FileName correspond au chemin du fichier à lire. L'image est écrite à l'adresse Array.
// La fonction suivante retourne 0 en cas de succès.
int ReadPGM(	char *FileName ,		
				 unsigned char *Array	);

// La fonction suivante écrit une image pgm en niveaux de gris codée sur 8 bits. FileName correspond au chemin du fichier 
// à créer. Width et Height sont les dimensions de l'image. Array correspond au tableau contenant l'image.
// La fonction suivante retourne 0 en cas de succès.
int WritePGM(	char 	      *FileName ,		
				unsigned int  Width		,
				unsigned int  Height	,
				unsigned char *Array	);

#endif

