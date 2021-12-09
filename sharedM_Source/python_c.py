import ctypes
import numpy as np

import time

np.random.seed(42)

_lib = ctypes.CDLL('./sample.so')

def totalCtypes(arr, n):
    _lib = ctypes.CDLL('./sample.so')

    _lib.total_double.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

    array_type = ctypes.c_double * n

    # return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)

    return _lib.total_double(array_type(*arr), ctypes.c_int(n))



for i in range(3):
    # n = 10
    # x = np.random.randint(1,100,n)
    # print(x)

    n = int(input("How many numbers do you need ?"))
    x = []
    for i in range(n):
        value = float(input())
        x.append(value)

    print(x)
    # x = list((np.random.random_sample(size=10)))

    totalCtypes(x, len(x))
    time.sleep(4)

    print(type(x))

# print(totalCtypes(x,n))
 
