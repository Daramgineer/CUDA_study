#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define _CRT_SECURE_NO_WARNINGS


#define SIZE 1024
#define THREADS_PER_BLOCK 16

__global__ void VectorDot(int* a, int* b, int* c)
{
	__shared__ int temp[THREADS_PER_BLOCK]; //block sheard mem에 temp값 공유
	int t_id = threadIdx.x + blockIdx.x * blockDim.x;
	temp[threadIdx.x] = a[t_id] * b[t_id]; //각 thread는 a, b 곱연산 수행
	__syncthreads(); //block내 thread의 연산 동기화

	int sum = 0;
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < THREADS_PER_BLOCK; i++) //block내 thread 곱연산결과 합산
		{
			sum += temp[i];
		}
		atomicAdd(c, sum);
	}
}

int main()
{
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;

	a = (int*)malloc(SIZE * sizeof(int));  // malloc을 통해 host(cpu)변수의 저장공간 확보
	b = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(sizeof(int));  // 저장공간 하나면 충분(a,b 곱연산의 합 저장).

	cudaMalloc(&d_a, (SIZE * sizeof(int))); //cudaMalloc을 통해 device내 사용 변수의 저장공간 확보 
	cudaMalloc(&d_b, (SIZE * sizeof(int)));
	cudaMalloc(&d_c, sizeof(int)); // 저장공간 하나면 충분(a,b 곱연산의 합 저장).

	for (int i = 0; i < SIZE; ++i) //host 변수 초기값 설정
	{
		a[i] = i;
		b[i] = i;
	}

	cudaMemcpy(d_a, a, (SIZE * sizeof(int)), cudaMemcpyHostToDevice); // malloc에서 설정한 host변수 저장공간 크기, device변수로 복제
	cudaMemcpy(d_b, b, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, sizeof(int));

	VectorDot << < 4, 16 >> > (d_a, d_b, d_c); //덧셈함수 device연산, <<< 사용블록수 , 블록당 사용쓰레드 수>>>
	cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost); //device연산결과 host변수에 복사

	printf("Final Sum : %d\n", *c);

	free(a); //host 변수 저장공간 해제
	free(b);
	free(c);

	cudaFree(d_a); //device 변수 저장공간 해제
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}