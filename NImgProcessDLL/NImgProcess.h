// NImgProcess.h: interface for the NImage Process class.
// Linux Version
//////////////////////////////////////////////////////////////////////
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "NImage.h"

#ifndef		_NIMGPROCESS_H
#define		_NIMGPROCESS_H


class NImgProcess
{
private:
//member variables
	int		nWidth;
	int		nHeight;
	int		nBitCount;
    int		nBytesPerLine;
	int		nBytesPerPixel;
	int		nNumColors;
	int		nSize;

private:
//member functions
	void  		GetImageParament(NImage* pImg);

	u_char** 	Create2DList(NImage* pImg);
	void  		Release2DList(u_char** list);
	void    	ArrayCopy(u_char **list0, u_char **list1, int Dx, int Dy);
	
	void  		histog(u_char** list,long* pg,int x, int y, int Dx, int Dy); 
	void    	Sobel_core(u_char **list0, u_char **list1, int Dx, int Dy);
	void		Laplacian_core(u_char **list0, u_char **list1, int Dx, int Dy);
	void		Mean_core(u_char **list0, u_char **list1, int Dx, int Dy);
	int   		Otsu(long* pg);  	
	int   		MaxMin(double* tab,int flag);
	int   		Ptile(long* pg,double nn); 

	bool		toSmall(u_char **list0, double x1, double y1, double x2, double y2, u_char *result);

public:
	NImgProcess();
 	~NImgProcess();
public:
//member functions
	bool		Inverse(NImage *pImg);
	bool		SingleThresholding(NImage *pImg, int threshold);
	bool		OtsuThresholding(NImage *pImg, NImage *pImg2);

	bool		Sobel(NImage *pImg);
	bool		Laplacian(NImage *pImg);
	bool		Mean(NImage *pImg);

	bool		Subtract(NImage *pSrcImg, NImage *pRefImg);
	bool		BitwiseXOR(NImage *pSrcImg, NImage *pRefImg);

	bool		Dilation3x3(NImage *pImg, NImage *pImg2);
	bool		Erosion3x3(NImage *pImg, NImage *pImg2);

	bool		Small_Transform(NImage *pSrcImg, NImage *pOutImg);
	bool		FromImageToVector(NImage *pImg, u_char *pVector, int element_num);
	bool		Split_Image(NImage *pSrcImg, int start_x, int start_y, int split_wid,
							int split_hei, NImage *pSubImg);
};

#endif		//!_NImgProcess_H
