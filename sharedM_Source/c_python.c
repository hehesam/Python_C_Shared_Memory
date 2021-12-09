#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/shm.h>
#include <string.h>

#include "struct.h"

// #include <string>

// #include <iostream>

// using namespace std;

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
// int main(){

// 	int * shared_memory;
// 	int buff[100];
// 	string buffs;

// 	int shmid;

// 	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);

// 	printf("Key of shared memory is %d\n", shmid);

// 	shared_memory = (int *)shmat(shmid, NULL, 0); // procces attached to shared memory segment

// 	printf("Process attached at %p\n", shared_memory);

// 	cout<<"input the string "<<endl;

// 	// getline(cin, buffs);

// 	for(int i=0;i<5;i++)
// 		cin>>shared_memory[i];

// 	cout<<"you have enter these numbers\n";

// 	for(int i=0;i<5;i++)
// 		cout<<shared_memory[i]<< "	";



// 	// read(0, buff, 100); // get some input from user



// 	// strcpy(shared_memory, buffs.c_str()); // data written to shared memory

// 	// printf("You wrote: %s\n", (char *)shared_memory);

// 	int a;
// 	cout << "enter the number : \n";
// 	cin>>a;

// }