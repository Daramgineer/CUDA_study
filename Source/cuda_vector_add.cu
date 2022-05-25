#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#define SIZE 1024
#define THREADS_PER_BLOCK 16

__global__ void VectorAdd(int* a, int* b, int* c, int n)  //global 사용을 통해  device(gpu)내 함수 선언
{
	for (int i = 0; i < n; ++i)
		c[i] = a[i] + b[i];
}

int main()
{
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;

	a = (int*)malloc(SIZE * sizeof(int));  // malloc을 통해 host(cpu)변수의 저장공간 확보
	b = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(SIZE * sizeof(int));

	cudaMalloc(&d_a, (SIZE * sizeof(int))); //cudaMalloc을 통해 device내 사용 변수의 저장공간 확보 
	cudaMalloc(&d_b, (SIZE * sizeof(int)));
	cudaMalloc(&d_c, (SIZE * sizeof(int)));

	for (int i = 0; i < SIZE; ++i) //host 변수 초기값 설정
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, (SIZE * sizeof(int)), cudaMemcpyHostToDevice); // malloc에서 설정한 host변수 저장공간 크기, device변수로 복제
	cudaMemcpy(d_b, b, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);


	VectorAdd << < SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, SIZE); //덧셈함수 device연산, <<< 사용블록수 , 블록당 사용쓰레드 수>>>

	cudaMemcpy(a, d_a, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);  //device연산값 host로 복사
	cudaMemcpy(b, d_b, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; ++i) //10회 덧셈 수행
		printf("c[%d] = %d\n", i, c[i]);

	free(a); //host 변수 저장공간 해제
	free(b);
	free(c);

	cudaFree(d_a); //device 변수 저장공간 해제
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}