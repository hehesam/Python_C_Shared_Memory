import ctypes
import numpy as np



def totalCtypes(arr, n):

    _lib = ctypes.CDLL('./sample.so')
    _lib.total.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    array_type = ctypes.c_double * n

    # return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)
    return _lib.total(array_type(*arr), ctypes.c_int(n))

n = 100


# x = np.arange(n, dtype=np.double)
# x = np.random.random((n,)).tolist()

# x = []
# for i in range(n):
#     x.append(i)

x = np.random.random((n,)).tolist()


print(x)
print(totalCtypes(x,n))



# print(our_function([1,2,-3,4,-5,6]))
