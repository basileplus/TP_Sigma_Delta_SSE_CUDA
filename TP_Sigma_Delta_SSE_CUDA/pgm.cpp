#include "pgm.hpp"

int ReadPGM_Size(char* FileName,
    unsigned int* Width,
    unsigned int* Height)
{
    FILE* ptrFile = NULL;
    char tmp[200];

    // Ouverture du fichier en mode binaire
    if (fopen_s(&ptrFile, FileName, "rb"))
        return 1;	// Echec lors de l'ouverture

    // Lecture de la première ligne "P5\n"
    fgets(tmp, 200, ptrFile);
    // Lecture des commentaires
    do {
        fgets(tmp, 200, ptrFile);
    } while (tmp[0] == '#');
    // Récupération de la taille de l'image
    sscanf_s(tmp, "%d %d\n", Width, Height);

    fclose(ptrFile);

    return 0;
}

int ReadPGM(	char *FileName ,		
				 unsigned char *Array	)
{
    FILE* ptrFile = NULL;
    char tmp[200];
    unsigned int MaxGrey, Width, Height, NbPixels, i, value;

    // Ouverture du fichier en mode binaire
    if (fopen_s(&ptrFile, FileName, "rb"))
        return 1;	// Echec lors de l'ouverture

    // Lecture de la première ligne "P5\n"
    fgets(tmp, 200, ptrFile);
    // Lecture des commentaires
    do {
        fgets(tmp, 200, ptrFile);
    } while (tmp[0] == '#');
    // Récupération de la taille de l'image
    sscanf_s(tmp, "%d %d\n", &Width, &Height);
    NbPixels = Width * Height;

    // Lecture du niveau de gris maximal
    fscanf_s(ptrFile, "%d\n", &MaxGrey);

    // Lecture de l'image
    if (MaxGrey <= 255)
    {
        if (MaxGrey != 0)
            for (i = 0; i < NbPixels; i++)
            {
                value = 0;
                if (!fread(&value, 1, 1, ptrFile))
                {
                    fclose(ptrFile);
                    return 1;
                }
                value = value * 255 / MaxGrey;
                Array[i] = value;
            }
        else
            for (i = 0; i < NbPixels; i++)
            {
                value = 0;
                if (!fread(&value, 1, 1, ptrFile))
                {
                    fclose(ptrFile);
                    return 1;
                }
                Array[i] = value;
            }
    }
    else
    {
        fclose(ptrFile);
        return 1;
    }

    // Fermeture du fichier
    if (fclose(ptrFile))
        return 1;	// Echec lors de la fermeture

   return 0;
}

int WritePGM(	char 	      *FileName ,		
				unsigned int  Width		,
				unsigned int  Height	,
				unsigned char *Array	)
{
   char P5[] = "P5\n", tmp;
   unsigned int  Divider, MaxGrey, i;
   FILE *ptrFile;

   if( Array == NULL )
      return 1;

   // Ouverture du fichier en mode binaire
   if( fopen_s(&ptrFile,FileName,"wb") )
      return 1;	// Echec lors de l'ouverture

   // Ecriture de la ligne "P5"
   fprintf_s(ptrFile, "P5\n");
   // Ecriture de la largeur et de la hauteur de l'image
   fprintf_s(ptrFile, "%d %d\n", Width, Height);

   // Ecriture du niveau de gris maximal
   MaxGrey = 1;
   for (i = 0; i < (Height * Width); i++)
       MaxGrey = (MaxGrey > Array[i]) ? MaxGrey : Array[i];
   fprintf_s(ptrFile, "%d\n", MaxGrey);

   // Ecriture de l'image
   for(i = 0;i<(Height*Width);i++)
      if( !fwrite(&Array[i],1,1,ptrFile) )
      {
         fclose(ptrFile);
         return 1;
      }

   // Fermeture du fichier
   if( fclose(ptrFile) )
      return 1;	// Echec lors de la fermeture

   return 0;
}


