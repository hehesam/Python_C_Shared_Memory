import ctypes
import numpy as np
# import voice_text
import time

np.random.seed(42)


def totalCtypes(arr, n):
    _lib = ctypes.CDLL('./sample.so')

    _lib.total_double.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

    array_type = ctypes.c_double * n

    # return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)

    return _lib.total_double(array_type(*arr), ctypes.c_int(n))



def totalCtypes_struct(arr, n):
    _lib = ctypes.CDLL('./sample.so')

    _lib.total_struct.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

    array_type = ctypes.c_double * n

    # return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)

    return _lib.total_struct(array_type(*arr), ctypes.c_int(n))

def totalCtypes_string(text):
    



def text_to_shared(wordList: str):

    x = wordList.split(" ")

    if x[0] == 'number':
        x.pop(0)


    print(x)
    # x = list((np.random.random_sample(size=10)))

    # totalCtypes(x, len(x))
    totalCtypes_struct(x, len(x))
    time.sleep(4)

    print(type(x))

 
