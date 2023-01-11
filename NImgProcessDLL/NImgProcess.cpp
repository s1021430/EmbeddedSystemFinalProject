// NImgProcess.cpp: implementation of the NImage Process class.
// Linux Version
//////////////////////////////////////////////////////////////////////
//#include <emmintrin.h>		// for SSE2
//#include <tmmintrin.h>		// for SSSE3
//#include <immintrin.h>


#include "NImgProcess.h"
#include <fstream>
#include <ostream>
#include <iostream>

using namespace std;

NImgProcess::NImgProcess()
{
	nWidth = 0;
	nHeight = 0;
	nBitCount = 0;
    nBytesPerLine = 0;
	nBytesPerPixel = 0;
	nNumColors = 0;
	nSize = 0;
}

NImgProcess::~NImgProcess()
{
	
}

bool NImgProcess::Inverse(NImage *pImg)
{
	u_char	*lpBuf;
 	int		x,y;
    
 	GetImageParament(pImg);              

 	for (y=0;y<nHeight;y++) 
	{
		lpBuf=(u_char*) pImg->GetPixelAddress(0,y);
		for (x=0;x<nWidth;x++,lpBuf++) 
			*lpBuf = 255 - *lpBuf;
	}

	return true;
}

bool NImgProcess::SingleThresholding(NImage *pImg, int threshold)
{
	u_char	*lpBuf;
 	int		x,y;
    
 	GetImageParament(pImg);              

 	for (y=0;y<nHeight;y++) 
	{
		lpBuf=(u_char*) pImg->GetPixelAddress(0,y);
		for (x=0;x<nWidth;x++,lpBuf++) 
		{
			if (*lpBuf >= threshold) *lpBuf=255;
			else		        	 *lpBuf=0;
		}
	}

	return true;
}

bool NImgProcess::OtsuThresholding(NImage *pImg, NImage *pImg2)
{
	u_char	**list, **list2;
	u_char	*lpBuf, *lpBuf2;
	long	pg[256];
 	int		x,y,thres;
    
 	GetImageParament(pImg);          

	if (pImg2->GetWidth() != nWidth || pImg2->GetHeight() != nHeight)
		return false;

	list=(u_char**) Create2DList(pImg);
	list2 = (u_char**)Create2DList(pImg2);

	histog(list,pg,0,0,nWidth,nHeight);
 
	thres = Otsu(pg);
	
 	for (y=0;y<nHeight;y++) 
	{
		lpBuf=(u_char*) pImg->GetPixelAddress(0,y);
		lpBuf2 = (u_char*)pImg2->GetPixelAddress(0, y);
		for (x=0; x<nWidth; x++, lpBuf++, lpBuf2++)
		{
			if (*lpBuf >= thres)*lpBuf2=255;
			else		        *lpBuf2=0;
		}
	}

	Release2DList(list);
	Release2DList(list2);

	return true;
}

bool NImgProcess::Sobel(NImage *pImg)
{
	u_char	**list1, **list0;
	
	GetImageParament(pImg);
	NImage  *Img1 = new NImage(nWidth, nHeight);
	
	list0 = (u_char**)Create2DList(pImg);
	list1 = (u_char**)Create2DList(Img1);

	Sobel_core(list0, list1, nWidth, nHeight);
	std::ofstream log_file("log.txt", std::ios::app);
	log_file << "Sobel_core!" << endl;
	ArrayCopy(list0, list1, nWidth, nHeight);
	log_file << "ArrayCopy!" << endl;
	Release2DList(list0);
	Release2DList(list1);


	delete Img1;

	return true;
}

bool NImgProcess::Laplacian(NImage *pImg)
{
	u_char	**list1, **list0;

	GetImageParament(pImg);
	NImage  *Img1 = new NImage(nWidth, nHeight);

	list0 = (u_char**)Create2DList(pImg);
	list1 = (u_char**)Create2DList(Img1);

	Laplacian_core(list0, list1, nWidth, nHeight);

	ArrayCopy(list0, list1, nWidth, nHeight);

	Release2DList(list0);
	Release2DList(list1);


	delete Img1;

	return true;
}

bool NImgProcess::Mean(NImage *pImg)
{
	u_char	**list1, **list0;

	GetImageParament(pImg);
	NImage  *Img1 = new NImage(nWidth, nHeight);

	list0 = (u_char**)Create2DList(pImg);
	list1 = (u_char**)Create2DList(Img1);

	Mean_core(list0, list1, nWidth, nHeight);

	ArrayCopy(list0, list1, nWidth, nHeight);

	Release2DList(list0);
	Release2DList(list1);


	delete Img1;

	return true;
}

bool NImgProcess::Subtract(NImage *pSrcImg, NImage *pRefImg)
{
	u_char	*lpSrcBuf, *lpRefBuf;	
	int		x, y;
	int		sub_value;

	GetImageParament(pSrcImg);

	if (pRefImg->GetWidth() != nWidth || pRefImg->GetHeight() != nHeight)
		return false;

	for (y = 0; y < nHeight; y++)
	{
		lpSrcBuf = (u_char*)pSrcImg->GetPixelAddress(0, y);
		lpRefBuf = (u_char*)pRefImg->GetPixelAddress(0, y);
		for (x = 0; x < nWidth; x++, lpSrcBuf++, lpRefBuf++)
		{
			sub_value = *lpSrcBuf - *lpRefBuf;
			
			if (sub_value < 0) sub_value = 0;

			*lpSrcBuf = sub_value;
		}
			
	}

	return true;
}

bool NImgProcess::BitwiseXOR(NImage *pSrcImg, NImage *pRefImg)
{
	u_char	*lpSrcBuf, *lpRefBuf;
	int		x, y;

	GetImageParament(pSrcImg);

	if (pRefImg->GetWidth() != nWidth || pRefImg->GetHeight() != nHeight)
		return false;

	for (y = 0; y < nHeight; y++)
	{
		lpSrcBuf = (u_char*)pSrcImg->GetPixelAddress(0, y);
		lpRefBuf = (u_char*)pRefImg->GetPixelAddress(0, y);
		for (x = 0; x < nWidth; x++, lpSrcBuf++, lpRefBuf++)
			*lpSrcBuf = (*lpSrcBuf) ^ (*lpRefBuf);

	}

	return true;
}

bool NImgProcess::Dilation3x3(NImage *pImg, NImage *pImg2)
{
	u_char	**list1, **list0;

	GetImageParament(pImg);
	
	if (pImg2->GetWidth() != nWidth || pImg2->GetHeight() != nHeight)
		return false;

	list0 = (u_char**)Create2DList(pImg);
	list1 = (u_char**)Create2DList(pImg2);

	for (int i = 1; i < nHeight - 1; i++)
	{
		for (int j = 1; j < nWidth - 1; j++)
		{
			if (list0[i - 1][j - 1] > 0 || list0[i - 1][j] > 0 ||
				list0[i - 1][j + 1] > 0 || list0[i][j - 1] > 0 ||
				list0[i][j] > 0 ||
				list0[i][j + 1] > 0 || list0[i + 1][j - 1] > 0 ||
				list0[i + 1][j] > 0 || list0[i + 1][j + 1] > 0)
			{

				list1[i][j] = 255;
			}
			else
			{
				list1[i][j] = 0;
			}
		}

	}

	Release2DList(list0);
	Release2DList(list1);


	return true;
}

bool NImgProcess::Erosion3x3(NImage *pImg, NImage *pImg2)
{
	u_char	**list1, **list0;

	GetImageParament(pImg);

	if (pImg2->GetWidth() != nWidth || pImg2->GetHeight() != nHeight)
		return false;

	list0 = (u_char**)Create2DList(pImg);
	list1 = (u_char**)Create2DList(pImg2);

	for (int i = 1; i < nHeight - 1; i++)
	{
		for (int j = 1; j < nWidth - 1; j++)
		{
			if (list0[i - 1][j - 1] < 255 || list0[i - 1][j] < 255 ||
				list0[i - 1][j + 1] < 255 || list0[i][j - 1] < 255 ||
				list0[i][j] < 255 ||
				list0[i][j + 1] < 255 || list0[i + 1][j - 1] < 255 ||
				list0[i + 1][j] < 255 || list0[i + 1][j + 1] < 255)
			{

				list1[i][j] = 0;
			}
			else
			{
				list1[i][j] = 255;
			}
		}
	}


	Release2DList(list0);
	Release2DList(list1);

	return true;
}

bool NImgProcess::Small_Transform(NImage *pSrcImg, NImage *pOutImg)
{
	int x, y;
	int x1, y1;
	int x2, y2;
	double fx1, fy1, fx2, fy2;

	int rx1, ry1, rx2, ry2;
	int h1, w1;
	int h2, w2;
	double fx, fy;

	u_char	result;
	u_char	**list1, **list0;

	GetImageParament(pSrcImg);

	list0 = (u_char**)Create2DList(pSrcImg);

	list1 = (u_char**)malloc(pOutImg->GetHeight() * sizeof(u_char*));
	for (int i = 0; i < pOutImg->GetHeight(); i++)
		list1[i] = (u_char*)pOutImg->GetPixelAddress(0, i);


	x1 = 0;
	y1 = 0;
	x2 = pOutImg->GetWidth() - 1;
	y2 = pOutImg->GetHeight() - 1;

	rx1 = 0;
	ry1 = 0;
	rx2 = pSrcImg->GetWidth() - 1;
	ry2 = pSrcImg->GetHeight() - 1;

	w1 = rx2 - rx1 + 1;
	h1 = ry2 - ry1 + 1;
	w2 = x2 - x1 + 1;
	h2 = y2 - y1 + 1;
	for (y = y1; y <= y2; y++) 
	{
		for (x = x1; x <= x2; x++) 
		{
			fx1 = (x - x1)*w1 / (double)w2 + rx1;
			fy1 = (y - y1)*h1 / (double)h2 + ry1;
			fx2 = (x + 1 - x1)*w1 / (double)w2 + rx1;
			fy2 = (y + 1 - y1)*h1 / (double)h2 + ry1;

			toSmall(list0, fx1, fy1, fx2, fy2, &result);
			
			list1[y][x] = result;
		}
	}

	Release2DList(list0);
	Release2DList(list1);

	return true;
}

bool NImgProcess::FromImageToVector(NImage *pImg, u_char *pVector, int element_num)
{
	GetImageParament(pImg);

	u_char **list0 = (u_char**)Create2DList(pImg);

	int k = 0;
	for (int i = 0; i < nHeight; i++)
	{
		for (int j = 0; j < nWidth; j++)
		{
			if (k >= element_num) break;
			
			pVector[k] = list0[i][j];

			k++;
		}
	}

	Release2DList(list0);

	return true;
}

bool NImgProcess::Split_Image(NImage *pSrcImg, int start_x, int start_y, int split_wid,
							  int split_hei, NImage *pSubImg)
{
	if (pSubImg->GetWidth() != split_wid || pSubImg->GetHeight() != split_hei)
		return false;
	
	GetImageParament(pSrcImg);

	if (start_x + split_wid > nWidth || start_y + split_hei > nHeight)
		return false;

	u_char **list0 = (u_char**)Create2DList(pSrcImg);

	u_char **list1 = (u_char**)malloc(pSubImg->GetHeight() * sizeof(u_char*));
	for (int i = 0; i < pSubImg->GetHeight(); i++)
		list1[i] = (u_char*)pSubImg->GetPixelAddress(0, i);

	for (int i = 0; i < split_hei; i++)
		for (int j = 0; j < split_wid; j++)
			list1[i][j] = list0[i+start_y][j+start_x];

	Release2DList(list0);
	Release2DList(list1);

	return true;

}

void NImgProcess::GetImageParament(NImage *pImg)
{
	if (pImg->IsNull()) return;

	nWidth = pImg->GetWidth();
	nHeight = pImg->GetHeight();
	nBitCount = pImg->GetBPP();
	nBytesPerLine = (pImg->GetWidth()*pImg->GetBPP()+31)/32*4;
	nBytesPerPixel = pImg->GetBPP()/8;
	if (pImg->GetBPP()<=8) 
		nNumColors = 1 << pImg->GetBPP();
	else 
		nNumColors = 0;
		
	nSize  = nBytesPerLine*nHeight;
}


u_char** NImgProcess::Create2DList(NImage *pImg)
{
	int			i;
	u_char		**list;

 	GetImageParament(pImg);
	list=(u_char**)malloc(nHeight*sizeof(u_char*));        
	for (i=0;i<nHeight;i++) 
		list[i]=(u_char*) pImg->GetPixelAddress(0,i); 

	return(list);
}

void NImgProcess::Release2DList(u_char** list)
{
	free(list);
}

void NImgProcess::ArrayCopy(u_char **list0, u_char **list1, int Dx, int Dy)
{
	int  i, j;

	for (i = 1; i < Dy - 1; i++)
		for (j = 1; j < Dx - 1; j++)
			list0[i][j] = list1[i][j];

}

void NImgProcess::histog(u_char** list,long* pg,int x, int y, int Dx, int Dy)
{                                                               
	int   i,j;

  	for (i=0;i<256;i++)  pg[i]=0;

	for(i=y;i<y+Dy;i++) 
 		for (j=x;j<x+Dx;j++)  
			pg[list[i][j]]++;	
}

void NImgProcess::Sobel_core(u_char **list0, u_char **list1, int Dx, int Dy)
{
	int  i, j, A, B, C;

	for (i = 1; i < Dy - 1; i++)
	{
		for (j = 1; j < Dx - 1; j++)
		{
			A = abs((list0[i - 1][j + 1] + 2 * list0[i][j + 1] + list0[i + 1][j + 1]) -
				(list0[i - 1][j - 1] + 2 * list0[i][j - 1] + list0[i + 1][j - 1]));
			B = abs((list0[i + 1][j - 1] + 2 * list0[i + 1][j] + list0[i + 1][j + 1]) -
				(list0[i - 1][j - 1] + 2 * list0[i - 1][j] + list0[i - 1][j + 1]));

			C = (int)sqrt((double)(A * A + B * B));

			C /= 4;

			if (C > 255) C = 255;

			list1[i][j] = (u_char)C;
		}
	}
}

void NImgProcess::Laplacian_core(u_char **list0, u_char **list1, int Dx, int Dy)
{
	int  i, j, C;

	for (i = 1; i < Dy - 1; i++)
	{
		for (j = 1; j < Dx - 1; j++)
		{
			C = -1 * list0[i - 1][j - 1] - 1 * list0[i - 1][j]
				- 1 * list0[i - 1][j + 1] - 1 * list0[i][j - 1]
				+ 8 * list0[i][j]
				- 1 * list0[i][j + 1] - 1 * list0[i + 1][j - 1]
				- 1 * list0[i + 1][j] - 1 * list0[i + 1][j + 1];


			if (C > 255) C = 255;
			else if (C < 0) C = 0;

			list1[i][j] = (u_char)C;
		}
	}
}

void NImgProcess::Mean_core(u_char **list0, u_char **list1, int Dx, int Dy)
{
	int  i, j, C;

	for (i = 1; i < Dy - 1; i++)
	{
		for (j = 1; j < Dx - 1; j++)
		{
			C = list0[i - 1][j - 1] + list0[i - 1][j]
				+ list0[i - 1][j + 1] + list0[i][j - 1]
				+ list0[i][j]
				+ list0[i][j + 1] + list0[i + 1][j - 1]
				+ list0[i + 1][j] + list0[i + 1][j + 1];


			C /= 9;

			list1[i][j] = (u_char)C;
		}
	}
}

int NImgProcess::Otsu(long *pg)
{
	int  i, j, p;
	double m0, m1, M0, M1, u, v, w[256];

	M0 = M1 = 0;
	for (i = 0; i < 256; i++)
	{
		M0 += pg[i];     
		M1 += pg[i] * i;
	}

	for (j = 0; j < 256; j++)
	{
		m0 = m1 = 0;
		for (i = 0; i <= j; i++)
		{
			m0 += pg[i];
			m1 += pg[i] * i;
		}

		if (m0) u = m1 / m0;
		else   u = 0;

		if (M0 - m0) v = (M1 - m1) / (M0 - m0);
		else      v = 0;

		w[j] = m0 * (M0 - m0)*(u - v)*(u - v);
	}

	p = MaxMin(w, 1);
	return(p);
}

int  NImgProcess::MaxMin(double* tab,int flag)
{
   double  max,min;
   int     i,p,q;

   max=min=tab[128];		
   p=q=128;
   for (i=0;i<256;i++) 
   {
   
      if (tab[i]>max) 
	  {
	     max=tab[i];	
		 p=i;
      }
	  
	  if (tab[i]<min) 
	  {
		 min=tab[i];	
		 q=i;
	  }
   }
   
   if (flag==1) 
   {
      for (i=0;i<256;i++) 
         tab[i]=120*(max-tab[i])/(max-min);
   }
   else 
   {         
      for (i=0;i<256;i++) 
         tab[i]=120*(tab[i]-min)/(max-min);
	 
	  p=q;
   } 

   return(p);
}

int NImgProcess::Ptile(long* pg,double nn)       
{
   int  i;
   double mm,kk;

   for (i=0,mm=0;i<256;i++) mm+=pg[i];
   
   kk=0;
   for (i=0;i<256;i++) 
   {
	  kk+=(double) pg[i]/mm;
      if (kk>=nn)  break;
   }
   return i;
}

bool NImgProcess::toSmall(u_char **list0, double x1, double y1, double x2, double y2, u_char *result)
{
	int sx, sy, ex, ey;
	int x, y;
	double e1, e2;
	double ph, pw;
	double gray;
	double s;
	double ss;
	
	u_char val;

	sx = (int)x1;
	sy = (int)y1;
	ex = (int)x2;
	ey = (int)y2;

	if ((double)ex != x2) ex++;
	if ((double)ey != y2) ey++;
	
	s = 0.0;
	gray = 0.0;
	for (y = sy; y < ey; y++) 
	{
		e1 = (double)y;
		e2 = (double)(y + 1);
		if (e1 < y1) e1 = y1;
		if (e2 > y2) e2 = y2;
		ph = e2 - e1;

		for (x = sx; x < ex; x++) 
		{
			val = list0[y][x];

			e1 = (double)x;
			e2 = (double)(x + 1);

			if (e1 < x1) e1 = x1;
			if (e2 > x2) e2 = x2;

			pw = e2 - e1;
			ss = pw * ph;
			s += ss;

			gray += (double)(val)*ss;
		}
	}

	if (s == 0.0) 
		return false;

	gray /= s;

	*result = (unsigned char)(gray + 0.5);

	return true;

}
