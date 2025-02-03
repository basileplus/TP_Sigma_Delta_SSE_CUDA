#ifndef	_PGM_H_
#define	_PGM_H_

#include <stdio.h>
#include <stdlib.h>

#define	ALIGNMENT	16 // bytes

// La fonction suivante permet de lire la taille d'une image pgm en niveaux de gris cod�e sur 8 bits.
// FileName correspond au chemin du fichier � lire. La fonction �crit aux adresses Width et Height 
// les dimensions de l'image lue. La fonction suivante retourne 0 en cas de succ�s.
int ReadPGM_Size(char* FileName,
	unsigned int* Width,
	unsigned int* Height);

// La fonction suivante permet de lire une image pgm en niveaux de gris cod�e sur 8 bits.
// FileName correspond au chemin du fichier � lire. L'image est �crite � l'adresse Array.
// La fonction suivante retourne 0 en cas de succ�s.
int ReadPGM(	char *FileName ,		
				 unsigned char *Array	);

// La fonction suivante �crit une image pgm en niveaux de gris cod�e sur 8 bits. FileName correspond au chemin du fichier 
// � cr�er. Width et Height sont les dimensions de l'image. Array correspond au tableau contenant l'image.
// La fonction suivante retourne 0 en cas de succ�s.
int WritePGM(	char 	      *FileName ,		
				unsigned int  Width		,
				unsigned int  Height	,
				unsigned char *Array	);

#endif

