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


void total_double(){
	int ii = 0;
	while(ii<20){
	printf("II value is %d\n", ii);
	double * shared_memory = NULL;
	double  flag;
	int shmid;
	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);
	printf("Key of shared memory is %d\n", shmid);

	shared_memory = (double *)shmat(shmid, NULL, 0); // procces attached to shared memory segment

	

	if(ii==0){
		flag = shared_memory[0];
		printf("HIHI\n");
	}

	else if(flag != shared_memory[0]){
		printf("Process attached at %p\n", shared_memory);

		flag = shared_memory[0];

		for(long unsigned int i=0;i<10;i++){
		printf("%f\n",shared_memory[i]);
			}

	printf("\n");

	}
	
	shmdt(shared_memory);
	ii++;
	sleep(2);
}

}

void total_struct(){
	int ii = 0;
	while(ii<60){
	printf("II value is %d\n", ii);
	struct Shared_Segment * shared_memory;
	double  flag;
	int shmid;

	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);
	printf("Key of shared memory is %d\n", shmid);

	shared_memory = shmat(shmid, NULL, 0); // procces attached to shared memory segment

	printf("Process attached at %p\n", shared_memory);


	if(ii==0){
		flag = shared_memory->numbers[0];
		// printf("HIHI\n");
	}

	else if(flag != shared_memory->numbers[0]){
		// printf("Process attached at %p\n", shared_memory);

		flag = shared_memory->numbers[0];

		for(long unsigned int i=0;i<shared_memory->size;i++){
		printf("%f\n",shared_memory->numbers[i]);
			}

	printf("\n");

	}
	
	shmdt(shared_memory);
	ii++;
	sleep(2);
}
}

int main(){

	// total_double();
	total_struct();

}