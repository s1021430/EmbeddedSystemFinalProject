// Image.cpp: implementation of the NImage class.
//
//////////////////////////////////////////////////////////////////////
#include "NImage.h"


NImage::NImage()
{
    lpBits = NULL;
    nSize = 0;
    nWidth = 0;
    nHeight = 0;
    nBitCount = 8;
}

NImage::NImage(int Width, int Height)
{
    lpBits = NULL;
    nSize = 0;
    nWidth = 0;
    nHeight = 0;
    nBitCount = 8;
}

NImage::~NImage()
{
	Destroy();
}

bool NImage::ReSize(int Width, int Height)
{
    nWidth = Width;
    nHeight = Height;

    nBytesPerPixel = 1;
    nBytesPerLine = (nWidth*nBitCount+31)/32*4;
    nSize = nBytesPerLine * nHeight;

    if (nBitCount == 8)
        nNumColors = 256;
    else
        nNumColors = 0;

    if(lpBits != 0)	{free(lpBits); lpBits = 0;}
    lpBits = (u_char*)malloc( nSize /*nWidth*nHeight*sizeof(CType)*/);

    memset(lpBits,0,nSize);

    return true;
}


bool NImage::IsNull()
{
	if (lpBits == 0)
        return true;
    else
        return false;
}


void  NImage::Destroy()
{
	if(lpBits != 0)		
	{
		free (lpBits); 
		lpBits = 0;
	}
}

u_char* NImage::GetPixelAddress(int x,int y)
{
	return(lpBits+(nHeight-1-y)*nBytesPerLine+x*nBytesPerPixel);
}


int   NImage::GetWidth()
{
	return(nWidth);
}

int   NImage::GetHeight()
{
	return(nHeight);
}

int   NImage::GetBPP()
{
	return(nBitCount);
}

int	  NImage::GetPitch()
{
	return(GetPixelAddress(0,1)-GetPixelAddress(0,0));
} 



//--------------------------------------------------------------
//  BMP

bool NImage::LoadBMP(char* filename)
{
	FILE *fp = fopen(filename, "rb");
    PaletteEntry colorTab[256];

    if(fp == 0)
        return false;

    fseek(fp,10,SEEK_SET);

    int m_offset = getDWord(fp);
    int size = getDWord(fp);
	
    if(size >= 36)
    {
        nWidth  = getDWord(fp);
        nHeight = getDWord(fp);
        nBitCount = getDWord(fp) >> 16;

        //nBitCount = nBytesPerPixel*8;
        if(nBitCount != 8)	{fclose(fp);return false;}

        if(nBitCount>8) nNumColors=0;
        else  nNumColors=1<<nBitCount;

        fseek(fp,54,SEEK_SET);
        memset(colorTab,0,sizeof(PaletteEntry)*4);
        fread(&colorTab,sizeof(u_char),4*nNumColors,fp);

        nBytesPerPixel = 1;
        nBytesPerLine = (nWidth*nBitCount+31)/32*4;
        nSize = nBytesPerLine * nHeight;

        fseek(fp,m_offset,SEEK_SET);

        if(lpBits != 0)	{free(lpBits); lpBits = 0;}
        lpBits = (u_char*)malloc(nSize);

        fread(&lpBits[0],sizeof(u_char),nSize,fp);

    }

    fclose(fp);
	
	return true;
}

bool NImage::SaveBMP(char* filename)
{
	int  fileStep = (nWidth + 3) & -4;
    int  bitmapHeaderSize = 40;
    int  paletteSize = 1024;
    int  headerSize = 14 /* fileheader */ + bitmapHeaderSize + paletteSize;
    int  fileSize = fileStep*nHeight + headerSize;
	
    PaletteEntry palette[256];

    FILE	*fp;
    fp=fopen(filename,"w+b");
    if (fp == 0)	return false;

    // write signature 'BM'
    const char* fmtSignBmp = "BM";
    putBytes(fp, fmtSignBmp, (int)strlen(fmtSignBmp) );

    // write file header
    putDWord(fp, fileSize ); // file size
    putDWord(fp, 0 );
    putDWord(fp, headerSize );
	
    // write bitmap header
    putDWord(fp, bitmapHeaderSize );

    putDWord(fp, nWidth );
    putDWord(fp, nHeight );
    putWord(fp, 1 );
    putWord(fp, 1 << 3 );
    putDWord(fp, 24 );
    putDWord(fp, 0 );
    putDWord(fp, 0 );
    putDWord(fp, 0 );
    putDWord(fp, 0 );
    putDWord(fp, 0 );

    if (nBitCount == 8)
    {
        FillGrayPalette(palette, 8);
        fwrite(palette,4,nNumColors,fp);
    }

    fwrite(lpBits,1,nSize,fp);
    fclose(fp);
	
    return true;
}
//  BMP

// Private member functions
void NImage::FillGrayPalette(PaletteEntry* palette, int bpp)
{
    int i, length = 1 << bpp;
    for( i = 0; i < length; i++ )
    {
        int val = (i * 255/(length - 1));
        palette[i].b = palette[i].g = palette[i].r = (u_char)val;
        palette[i].a = 0;
    }
}

int NImage::getByte(FILE *fp)
{
    int val = 0;
    fread(&val,sizeof(u_char),1,fp);
    return val;
}

int NImage::getWord(FILE *fp)
{
    int   val=0;
    val = getByte(fp);
    val|= getByte(fp) << 8;
    return val;

}

int NImage::getDWord(FILE *fp)
{
    int   val=0;
    val = getByte(fp);
    val |= getByte(fp) << 8;
    val |= getByte(fp) << 16;
    val |= getByte(fp) << 24;
    return val;
}

void NImage::putByte(FILE *fp, int val)
{
    fwrite( &val, sizeof(u_char), 1, fp );
}

void NImage::putBytes(FILE *fp, const char *buffer, int count)
{
    u_char *data = (u_char*)buffer;
    fwrite(&data[0], sizeof(u_char), count, fp );
}

void NImage::putWord(FILE *fp, int val)
{
    putByte(fp,val);
    putByte(fp,val >> 8);
}

void NImage::putDWord(FILE *fp, int val)
{
    putByte(fp, val);
    putByte(fp,val >> 8);
    putByte(fp,val >> 16);
    putByte(fp,val >> 24);
}

 
