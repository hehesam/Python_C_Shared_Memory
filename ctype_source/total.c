#include <stdio.h>

int total(int *x, int n){
	int i;
	double count = 0;
	for(i = 0 ; i<2*n;i++){
		count += x[i];
		printf("%d\n", x[i]);
	}

	return count;
}

