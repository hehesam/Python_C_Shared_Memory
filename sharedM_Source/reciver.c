#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/shm.h>
#include <string.h>

// #include <string>

// #include <iostream>

// using namespace std;


int main(){

	int * shared_memory;
	char buff[100];
	int shmid;

	shmid = shmget((key_t)1122, 1024, 0666|IPC_CREAT);

	printf("Key of shared memory is %d\n", shmid);

	shared_memory = shmat(shmid, NULL, 0); // procces attached to shared memory segment

	printf("Process attached at %p\n", shared_memory);

	printf("the shared memory in reviver file is : \n");
	
	for(int i=0;i<20;i++){
		printf("%d\n",shared_memory[i]);
		if(shared_memory[i]/2){
			printf("Even\n");
		}
		else
			printf("odd\n");
	}
	printf("\n");

	// printf("data read from shared memory is: %s\n", (char *)shared_memory);

	// printf("you have to write something\n");

	// read(0, buff, 100); // get some input from user

	// strcpy(shared_memory, buff); // data written to shared memory


}