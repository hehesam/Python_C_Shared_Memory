import ctypes
import numpy as np

class SomeStructure(ctypes.Structure):
    _fields_ = [
            ("name", ctypes.c_int),
            ('c', ctypes.c_char_p),
            ('s', ctypes.c_char_p)
    ]
def Struct_c():
    lib = ctypes.CDLL('./sample.so')
    lib.someFunction.restype = ctypes.c_double
    lib.someFunction.argtypes = [ctypes.POINTER(SomeStructure)]

    s_obj = SomeStructure(3, b'q', b'hello')
    result = lib.someFunction(ctypes.byref(s_obj))
    print("result: %s, new value for text: %s" %(result, str(s_obj.s)))

def totalCtypes(arr, n):

    _lib = ctypes.CDLL('./sample.so')
    _lib.total.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    array_type = ctypes.c_double * n


    # return _lib.total(arr.ctypes.data_as(ctypes.c_void_p), n)
    return _lib.total(array_type(*arr), ctypes.c_int(n))


def sayMyName(name):

    name = bytes(name, 'utf-8')

    _lib = ctypes.CDLL('./sample.so')
    _lib.myName.argtypes = [ctypes.c_char_p]
    # name_type = ctypes.c_char * n
    # return _lib.myName(name_type(*name), ctypes.c_int(n))
    print()

    print(type(name))

    return _lib.myName(name)
    
n = 100


# x = np.arange(n, dtype=np.double)
# x = np.random.random((n,)).tolist()

# x = []
# for i in range(n):
#     x.append(i)

x = np.random.random((n,)).tolist()


# print(x)
# print(totalCtypes(x,n))

name = 'Hesam'
# name = bytes(name, 'utf-8')
print(type(name))


sayMyName(name)


# print(our_function([1,2,-3,4,-5,6]))
