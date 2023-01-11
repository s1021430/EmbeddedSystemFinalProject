import ctypes
import numpy as np
from ctypes import * 

lib2 = CDLL('NImgProcessDLL.so')

lib2.CreateNImgProcess.restype = POINTER(c_ulong)
lib2.DestroyNImgProcess.argtypes = [POINTER(c_ulong)]
lib2.CreateNImgProcess.restype = POINTER(c_ulong)
lib2.DestroyNImgProcess.argtypes = [POINTER(c_ulong)]
lib2.Inverse.argtypes = [POINTER(c_ulong), POINTER(c_ulong)]
lib2.Inverse.restype = c_bool
lib2.SingleThresholding.argtypes = [POINTER(c_ulong), POINTER(c_ulong), c_int]
lib2.SingleThresholding.restype = c_bool
lib2.OtsuThresholding.argtypes = [POINTER(c_ulong), POINTER(c_ulong), POINTER(c_ulong)]
lib2.OtsuThresholding.restype = c_bool

lib2.Sobel.argtypes = [POINTER(c_ulong), POINTER(c_ulong)]
lib2.Sobel.restype = c_bool
lib2.Laplacian.argtypes = [POINTER(c_ulong), POINTER(c_ulong)]
lib2.Laplacian.restype = c_bool
lib2.Mean.argtypes = [POINTER(c_ulong), POINTER(c_ulong)]
lib2.Mean.restype = c_bool

# Subtract(unsigned long m_SrcImg, unsigned long m_RefImg, unsigned long m_ImgPro)
lib2.Subtract.argtypes = [POINTER(c_ulong), POINTER(c_ulong), POINTER(c_ulong)]
lib2.Subtract.restype = c_bool
# Dilation3x3(unsigned long m_Img, unsigned long m_Img2, unsigned long m_ImgPro)
lib2.Dilation3x3.argtypes = [POINTER(c_ulong),POINTER(c_ulong), POINTER(c_ulong)]
lib2.Dilation3x3.restype = c_bool
# Erosion3x3(unsigned long m_Img, unsigned long m_Img2, unsigned long m_ImgPro)
lib2.Erosion3x3.argtypes = [POINTER(c_ulong),POINTER(c_ulong), POINTER(c_ulong)]
lib2.Erosion3x3.restype = c_bool
#Small_Transform(unsigned long m_SrcImg, unsigned long m_OutImg, unsigned long m_ImgPro)
lib2.Small_Transform.argtypes = [POINTER(c_ulong), POINTER(c_ulong), POINTER(c_ulong)]
lib2.Small_Transform.restype = c_bool
#FromImageToVector(unsigned long m_Img, u_char *m_Vector, int element_num, unsigned long m_ImgPro);
lib2.FromImageToVector.argtypes = [POINTER(c_ulong), POINTER(c_ubyte), c_int, POINTER(c_ulong)]
lib2.FromImageToVector.restype = c_bool
#Split_Image(unsigned long m_SrcImg, int start_x, int start_y, int split_wid, int split_hei, unsigned long m_SubImg, unsigned long m_ImgPro);
lib2.Split_Image.argtypes = [POINTER(c_ulong), c_int, c_int, c_int, c_int, POINTER(c_ulong),POINTER(c_ulong)]
lib2.Split_Image.restype = c_bool

class ImgProcessClass(object):

	def __init__(self):
		self.obj = lib2.CreateNImgProcess()

	def __del__(self):
		lib2.DestroyNImgProcess(self.obj)
	
	def Subtract(self, Img1, Img2):
		if(lib2.Subtract(Img1, Img2, self.obj)): return True
	
	def Dilation3x3(self, Img1, Img2):
		if(lib2.Dilation3x3(Img1, Img2, self.obj)): return True
	
	def Erosion3x3(self, Img1, Img2):
		if(lib2.Erosion3x3(Img1, Img2, self.obj)): return True

	def Inverse(self, Img): 
		if(lib2.Inverse(Img, self.obj)): return True
		
	def SingleThresholding(self, Img, thres):
		if(lib2.SingleThresholding(Img, self.obj, thres)): return True
		
	def OtsuThresholding(self, Img1, Img2):
		if(lib2.OtsuThresholding(Img1, Img2, self.obj)): return True
	
	def Sobel(self, Img):
		if lib2.Sobel(Img, self.obj): return True

	def Laplacian(self, Img):
		if lib2.Laplacian(Img, self.obj): return True

	def Mean(self, Img):
		if lib2.Mean(Img, self.obj): return True

#Small_Transform(BYTE* m_SrcImg, int srcWid, int srcHei,BYTE* m_OutImg, int outWid, int outHei,unsigned long m_ImgPro)
#lib2.Small_Transform.argtypes = [POINTER(c_uint8), c_int, c_int, POINTER(c_uint8), c_int, c_int, POINTER(c_ulong)]
	def Small_Transform(self, Img1, Img2):
		if(lib2.Small_Transform(Img1, Img2, self.obj)): return True
#FromImageToVector(BYTE* pImg, int wid, int hei,BYTE* m_Vector, int element_num, unsigned long m_ImgPro);
#lib2.FromImageToVector.argtypes = [POINTER(c_uint8), c_int, c_int, POINTER(c_uint8), c_int, POINTER(c_ulong)]
	def FromImageToVector(self, Img, vector, element_num):
		c_arr = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
		if lib2.FromImageToVector(Img, c_arr, element_num, self.obj) : 
			return True

#Split_Image(BYTE* m_SrcImg, int srcWid, int srcHei, int start_x, int start_y, int split_wid, int split_hei,BYTE*  m_SubImg, int subWid, int subHei, unsigned long m_ImgPro);
#lib2.Split_Image.argtypes = [POINTER(c_uint8), c_int, c_int, c_int, c_int, c_int, c_int, POINTER(c_uint8), c_int, c_int, POINTER(c_ulong)]
	def Split_Image(self, srcImg, splittedImg, start_x, start_y, split_w, split_h):
		if lib2.Split_Image(srcImg, start_x, start_y, split_w, split_h, splittedImg, self.obj): return True

