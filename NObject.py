import numpy as np
import ctypes
from ctypes import *

lib3 = CDLL('NObjectDLL.so')

lib3.CreateNObject.restype = POINTER(c_ulong)
lib3.DestroyNObject.argtypes = [POINTER(c_ulong)]

# Blob_Labelling(BYTE* m_Img, LONG_PTR m_Obj, int wid, int hei);
lib3.Blob_Labelling.argtypes = [POINTER(c_uint8), POINTER(c_ulong), c_int, c_int]
lib3.Blob_Labelling.restype = c_int

# Contour_Tracing(BYTE* m_Img, LONG_PTR m_Obj, int blob_num, int *ct_x, int  *ct_y, int wid, int hei);
lib3.Contour_Tracing.argtypes = [POINTER(c_uint8), POINTER(c_ulong), c_int, POINTER(c_int), POINTER(c_int), c_int, c_int]
lib3.Contour_Tracing.restype = c_int

# CreateMaskFromObject(BYTE* pMask_Img, LONG_PTR m_Obj, int blob_num, int wid, int hei)
lib3.CreateMaskFromObject.argtypes = [POINTER(c_uint8), POINTER(c_ulong), c_int, c_int, c_int]
lib3.CreateMaskFromObject.restype = c_bool

#Area(unsigned long m_Obj, int blob_num);
lib3.Area.argtypes = [POINTER(c_ulong),  c_int]
lib3.Area.restype = c_int

#Rect(unsigned long m_Obj, int blob_num, int *start_x, int *start_y, int *rect_w, int *rect_h);
lib3.Rect.argtypes = [POINTER(c_ulong),  c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]
lib3.Rect.restype = c_bool

#Blob_Count(unsigned long m_Obj)
lib3.Blob_Count.argtypes = [POINTER(c_ulong)]
lib3.Blob_Count.restype = c_int

class NObjectClass(object):

    def __init__(self):
        self.obj = lib3.CreateNObject()

    def __del__(self):
        lib3.DestroyNObject(self.obj)

    def Blob_Labelling(self, data):
        pro_data = data.copy()
        wid = pro_data.shape[1]
        hei = pro_data.shape[0]
        return lib3.Blob_Labelling(pro_data.ctypes.data_as(POINTER(c_uint8)), self.obj, wid, hei)

    def Contour_Tracing(self, data, i, ct_x, ct_y):
        pro_data = data.copy()
        wid = pro_data.shape[1]
        hei = pro_data.shape[0]
        return lib3.Contour_Tracing(pro_data.ctypes.data_as(POINTER(c_uint8)), self.obj, i, ct_x, ct_y, wid, hei)
    
    def CreateMaskFromObject(self, data, blob_num):
        pro_data = data.copy()
        wid = pro_data.shape[1]
        hei = pro_data.shape[0]
        if(lib3.CreateMaskFromObject(pro_data.ctypes.data_as(POINTER(c_uint8)), self.obj, blob_num, wid, hei)): 
            return pro_data
        else:
            return None

    def Area(self, blobNum):
        return lib3.Area(self.obj, blobNum)

    def Rect(self, blobNum, start_x, start_y, rech_w, rech_h):
        return lib3.Rect(self.obj, blobNum, start_x, start_y, rech_w, rech_h)

    def Blob_Count(self):
        return lib3.Blob_Count(self.obj)
