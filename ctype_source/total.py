import ctypes
import numpy as np

_lib = ctypes.CDLL('./sample.so')

def totalCtypes(arr, n):
    return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)

n = 100

x = np.arange(n)
print(x)
print(totalCtypes(x,n))
