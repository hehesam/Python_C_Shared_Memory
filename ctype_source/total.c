#include <stdio.h>
struct SomeStructure
{
	int i;
	char* c;
	char* s;
};

double someFunction(struct SomeStructure *s){
	printf("int is %d, char is %s, string is %s\n",s->i, s->c, s->s );
	s->s = "goodbye";
	return 42;
}


int total(double *x, int n){
	int i;
	double count = 0;
	for(i = 0 ; i<n;i++){
		count += x[i];
		printf("%d : %f\n", i, x[i]);
	}

	return count;
}

int myName(char *name){
	
	printf("my nme is : %s\n", name);
	return 1;
}

