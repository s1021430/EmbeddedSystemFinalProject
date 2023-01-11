import ctypes
import numpy as np
from ctypes import * 

lib5 = CDLL('MLPDLL.so')

lib5.CreateMLP.restype = POINTER(c_ulong)
lib5.DestroyMLP.argtypes = [POINTER(c_ulong)]
lib5.DestroyMLP.restype = c_bool

lib5.LoadNetwork.argtypes = [POINTER(c_ulong), POINTER(c_char)]
lib5.LoadNetwork.restype = c_bool
lib5.SaveNetwork.argtypes = [POINTER(c_ulong), POINTER(c_char)]
lib5.SaveNetwork.restype = c_bool
lib5.Training.argtypes = [POINTER(c_ulong), POINTER(POINTER(c_uint8)), POINTER(c_char), c_int]
lib5.Training.restype = c_bool
lib5.Classify.argtypes = [POINTER(c_ulong), POINTER(c_ubyte)]
lib5.Classify.restype = c_char_p

class MLPClass(object):

	def __init__(self):
		self.obj = lib5.CreateMLP()

	def __del__(self):
		lib5.DestroyMLP(self.obj)
		
	def LoadNetwork(self, path):
		b_string1 = path.encode('utf-8')
		return lib5.LoadNetwork(self.obj, create_string_buffer(b_string1))
		# netWorkFileNameBuffer = ctypes.c_char_p(netWorkFileName.encode('utf-8'))
		# return lib5.LoadNetwork(self.obj, netWorkFileNameBuffer)

	def SaveNetwork(self, path):
		b_string1 = path.encode('utf-8')
		return lib5.SaveNetwork(self.obj, create_string_buffer(b_string1))
		# netWorkFileNameBuffer = create_string_buffer(netWorkFileName.encode('utf-8'))
		# return lib5.SaveNetwork(self.obj, netWorkFileNameBuffer)

	def Training(self, samples, trainer_string, num):
		return lib5.Training(self.obj, samples, trainer_string, num)

	def Classify(self, sample):
		c_arr = sample.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
		a = lib5.Classify(self.obj, c_arr)
		return a