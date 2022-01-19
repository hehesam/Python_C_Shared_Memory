#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/shm.h>
#include <string.h>

#include "struct.h"

// #include <string>

// #include <iostream>

// using namespace std;

// struct Shared_Segment
// {
// 	int size;
// 	double * numbers;
	
// };

void array_printer(int * arr, int n){
	printf("this is the array in c_ptyhon file \n");
	for(int i=0;i<n*2;i++)
		printf("%d \n",arr[i]);

	// cout<<endl;
	printf("\n");
}


int total_int(int * arr, int n){
	array_printer(arr, n);
	int * shared_memory;

	int shmid;

	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);

	// printf("Key of shared memory is %d\n", shmid);

	shared_memory = (int *)shmat(shmid, NULL, 0); // procces attached to shared memory segment

	printf("Process attached at %p\n", shared_memory);

	printf("i will now copy the arr in shared memory\n");

	for(int i=0;i<n*2;i++)

		shared_memory[i] = arr[i];

	printf("the shared_memory should now be in full \n");

	array_printer(shared_memory, n);

	
	return 1;

}

double total_double(double * arr, int n){
	double * shared_memory;

	int shmid;

	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);

	// printf("Key of shared memory is %d\n", shmid);

	shared_memory = (double *)shmat(shmid, NULL, 0); // procces attached to shared memory segment

	printf("Process attached at %p\n", shared_memory);

	printf("i will now copy the arr in shared memory\n");

	for(int i=0;i<n;i++){


		shared_memory[i] = arr[i];
		printf("index : %d has the value : %f\n",i, shared_memory[i] );
	}

	printf("the shared_memory should now be in full \n");

	// array_printer(shared_memory, n);

	
	return 1;


}

double total_struct(double * arr, int n){

	struct Shared_Segment * shared_memory;

	int shmid;

	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);

	printf("Key of shared memory is %d\n", shmid);

	shared_memory = shmat(shmid, NULL, 0); // procces attached to shared memory segment

	printf("Process attached at %p\n", shared_memory);

	printf("i will now copy the arr in shared memory numbers\n");

	for(int i=0;i<n;i++){

		// printf("value of arr is : %f\n", arr[i]);

		shared_memory->numbers[i] = arr[i];


		printf("index : %d has the value : %f\n",i, shared_memory->numbers[i] );
	}
	shared_memory->size = n;

	printf("the shared_memory should now be in full \n");

	// array_printer(shared_memory, n);

	
	return 1;

}

int total_string(char *text){

	printf("text in c_ptyhon file is  : %s\n", text);

	char * shared_memory;

	int shmid;

	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);

	printf("Key of shared memory is %d\n", shmid);

	shared_memory = shmat(shmid, NULL, 0); // procces attached to shared memory segment

	printf("Process attached at %p\n", shared_memory);

	printf("i will now copy the arr in shared memory numbers\n");

	// shared_memory = text;

	printf("the length of text is : %ld\n", sizeof(text));

	strcpy(shared_memory, text);

	printf("shared_memory value is : %s\n", shared_memory);

	// for(int i=0;i<sizeof(text);i++){

	// 	// printf("value of arr is : %f\n", arr[i]);

	// 	shared_memory->numbers[i] = arr[i];


	// 	printf("index : %d has the value : %f\n",i, shared_memory->numbers[i] );
	// }


	return 1;
}

// int main(){
// 	// double  arr[3];
// 	// arr[0] = 1.01;
// 	// arr[1] = 3.03;
// 	// arr[2] = 2.02;
// 	// total_struct(arr, 3);
// }