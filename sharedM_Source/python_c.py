import ctypes
import numpy as np

import time

np.random.seed(42)

_lib = ctypes.CDLL('./sample.so')

def totalCtypes(arr, n):
    return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)

for i in range(3):
    n = 10
    x = np.random.randint(1,100,n)
    print(x)
    totalCtypes(x, n)
    # time.sleep(7)

    print(type(x))

# print(totalCtypes(x,n))
 
