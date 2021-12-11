
#making sample.so from file.o for creating shared library
gcc -fPIC -c file.c
gcc -shared file.o -o sample.so
