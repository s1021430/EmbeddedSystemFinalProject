// NImage.h: interface for the NImage class.
// Linux Version
//////////////////////////////////////////////////////////////////////

#pragma once

#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef		_NIMAGE_H
#define		_NIMAGE_H

#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4) 

struct PaletteEntry
{
    u_char b, g, r, a;
};

class NImage 
{
private:
	
 	u_char*    		lpBits;

	int				nWidth;
	int				nHeight;
	int				nBitCount;
    int				nBytesPerLine;
	int				nBytesPerPixel;
	int				nNumColors;
	int				nSize;

public:
	NImage();
	NImage(int Width, int Height);
 	~NImage();
	
public:

 	bool  IsNull();
	void  Destroy();

	int   GetWidth();
	int   GetHeight();
	int   GetBPP();
	int	  GetPitch();

  
	u_char* GetPixelAddress(int x,int y);


	bool  ReSize(int Width, int Height);

	bool  LoadBMP(char* filename);
	bool  SaveBMP(char* filename);
	
private:
	void  FillGrayPalette( PaletteEntry* palette, int bpp);
	
	int	  getByte(FILE *fp);
    int	  getWord(FILE *fp);
    int	  getDWord(FILE *fp);

    void  putByte(FILE *fp, int val);
    void  putBytes(FILE *fp, const char *buffer, int count);
    void  putWord(FILE *fp, int val);
    void  putDWord(FILE *fp, int val);


};

#endif		//!_NIMAGE_H

