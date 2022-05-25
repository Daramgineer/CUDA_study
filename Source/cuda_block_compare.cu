#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void loop()
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; //thread num ����ȭ
	printf("This is iteration number %d\n", idx);
}

int main()
{
	clock_t st = clock(); //�����ð� üũ ����
	int N = 10;

	loop << <1, N >> > (); //single block (168ms)
	loop << <2, N / 2 >> > (); //multi block (152ms) 

	cudaDeviceSynchronize(); //thread���� ����ȭ

	clock_t ed = clock(); //�����ð� üũ ����
	printf("time %u ms\n", ed - st); //�����ð� out
}