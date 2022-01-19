## Python_C_Shared_Memory
Py_C shared memory rep
mudual need to run until now :
#c:
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/shm.h>
#include <string.h>

#python:
ctypes
speech_recognition
pyAudio

making shared library :
gcc -fPIC -c file.c
gcc -shared file.o -o sample.so